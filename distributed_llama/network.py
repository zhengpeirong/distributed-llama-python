"""TCP network layer for distributed root-worker synchronization.

Port of src/nn/nn-network.cpp and src/nn/nn-network.hpp.
Uses pure Python ``socket`` and ``selectors`` -- no C extensions needed.
"""

import socket
import selectors
import struct
import time
import numpy as np
from typing import List, Optional, Tuple, Any

from .graph_builder import (
    NnNetConfig, NnNodeConfig, NnSegmentConfig,
    NnPipeConfig, NnSyncType, NnSize3D, size0,
    NnSyncConfig, NnOpConfig, NnPointerConfig, NnPreSyncConfig,
    NnBufferConfig,
)
from .quants import get_bytes, F_32, F_16, F_Q40, F_Q80

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT_SOCKET_INDEX = 0
ACK = 23571114
MAX_CHUNK_SIZE = 4096


# ---------------------------------------------------------------------------
# Helpers -- blocking / non-blocking send & recv
# ---------------------------------------------------------------------------

def _send_all(sock: socket.socket, data: bytes) -> None:
    """Send all bytes on a (possibly non-blocking) socket."""
    view = memoryview(data)
    total = len(view)
    sent = 0
    while sent < total:
        chunk = view[sent:sent + MAX_CHUNK_SIZE]
        try:
            n = sock.send(chunk)
        except BlockingIOError:
            # Busy-wait briefly when non-blocking
            time.sleep(0.0)
            continue
        if n == 0:
            raise ConnectionError("Socket closed during send")
        sent += n


def _recv_all(sock: socket.socket, size: int) -> bytes:
    """Receive exactly *size* bytes on a (possibly non-blocking) socket."""
    parts = []
    received = 0
    while received < size:
        chunk_size = min(size - received, MAX_CHUNK_SIZE)
        try:
            data = sock.recv(chunk_size)
        except BlockingIOError:
            time.sleep(0.0)
            continue
        if not data:
            raise ConnectionError("Socket closed during recv")
        parts.append(data)
        received += len(data)
    return b"".join(parts)


def _send_int32(sock: socket.socket, value: int) -> None:
    _send_all(sock, struct.pack("<i", value))


def _recv_int32(sock: socket.socket) -> int:
    return struct.unpack("<i", _recv_all(sock, 4))[0]


def _send_string(sock: socket.socket, s: str) -> None:
    data = (s + "\0").encode("utf-8")
    _send_int32(sock, len(data))
    _send_all(sock, data)


def _recv_string(sock: socket.socket) -> str:
    n_bytes = _recv_int32(sock)
    raw = _recv_all(sock, n_bytes)
    # Strip trailing null terminator
    if raw and raw[-1:] == b"\0":
        raw = raw[:-1]
    return raw.decode("utf-8")


# ---------------------------------------------------------------------------
# NnSocket -- thin wrapper around Python socket.socket
# ---------------------------------------------------------------------------

class NnSocket:
    """Thin wrapper around Python ``socket.socket``."""

    def __init__(self, sock: Optional[socket.socket] = None):
        self.fd: Optional[socket.socket] = None
        if sock is not None:
            self.assign(sock)

    def assign(self, sock: socket.socket) -> None:
        """Assign an existing socket (takes ownership)."""
        if sock is None:
            raise ValueError("Cannot assign None socket")
        if self.fd is not None:
            self.release()
        self.fd = sock

    def release(self) -> Optional[socket.socket]:
        """Give up ownership of the socket WITHOUT closing it.
        Returns the raw socket and clears the internal reference.
        """
        if self.fd is not None:
            sock = self.fd
            self.fd = None
            return sock
        return None

    def __del__(self):
        self.release()


# ---------------------------------------------------------------------------
# NnNetwork -- multi-socket connection manager
# ---------------------------------------------------------------------------

class NnNetwork:
    """Manages one or more connected TCP sockets for node-node communication."""

    def __init__(self, sockets: List[NnSocket]):
        n = len(sockets)
        # Move ownership: release each NnSocket and store the raw socket
        self._sockets: List[socket.socket] = []
        for s in sockets:
            raw = s.release()
            if raw is None:
                raise ValueError("NnSocket with no fd")
            self._sockets.append(raw)
        self.n_sockets: int = n
        self._sent_bytes: List[int] = [0] * n
        self._recv_bytes: List[int] = [0] * n

    # --- Factory methods ----------------------------------------------------

    @classmethod
    def serve(cls, host: str, port: int) -> "NnNetwork":
        """Create TCP server, accept connections using the full node-discovery
        protocol, return a fully-connected ``NnNetwork``."""

        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(socket.SOMAXCONN)
        print(f"Listening on {host}:{port}...")

        # Accept root connection (always socket 0)
        root_conn, root_addr = server.accept()
        print("Root node has connected")
        _set_no_delay(root_conn)
        _set_quick_ack(root_conn)

        # Root sends topology info
        n_sockets = _recv_int32(root_conn)
        n_nodes = n_sockets - 1  # minus root
        node_index = _recv_int32(root_conn)
        print(f"nNodes: {n_nodes}")
        print(f"NodeIndex: {node_index}")

        # Read host/port for each worker
        hosts: List[str] = [""] * n_nodes
        ports: List[int] = [0] * n_nodes
        for i in range(n_nodes):
            hosts[i] = _recv_string(root_conn)
            ports[i] = _recv_int32(root_conn)

        # Acknowledge topology
        _send_int32(root_conn, ACK)

        # Wait for root to indicate readiness
        ack = _recv_int32(root_conn)
        if ack != ACK:
            raise RuntimeError(f"Expected ACK from root, got {ack}")

        # Build socket list: index 0 is root, then workers
        sockets: List[NnSocket] = [NnSocket(root_conn)]

        for i in range(n_nodes):
            socket_index = i + 1
            if i >= node_index:
                print(f"  Socket[{socket_index}]: connecting to {hosts[i]}:{ports[i]} worker")
                conn = _connect_socket(hosts[i], ports[i])
                sockets.append(NnSocket(conn))
                print(f"  Socket[{socket_index}]: connected")
            else:
                print(f"  Socket[{socket_index}]: wait for {hosts[i]}:{ports[i]} worker")
                conn, addr = server.accept()
                _set_no_delay(conn)
                _set_quick_ack(conn)
                sockets.append(NnSocket(conn))
                print(f"  Socket[{socket_index}]: accepted")

        server.close()
        print("Network is initialized")
        return cls(sockets)

    @classmethod
    def connect(cls, n_sockets: int, hosts: List[str], ports: List[int]) -> "NnNetwork":
        """Connect to *n_sockets* workers, perform handshake, return ``NnNetwork``."""
        assert n_sockets > 0

        sockets: List[NnSocket] = []
        for i in range(n_sockets):
            print(f"  Socket[{i}]: connecting to {hosts[i]}:{ports[i]} worker")
            fd = _connect_socket(hosts[i], ports[i])

            # Send topology info
            _send_int32(fd, n_sockets)
            _send_int32(fd, i)
            for j in range(n_sockets):
                if j == i:
                    continue
                _send_string(fd, hosts[j])
                _send_int32(fd, ports[j])

            # Wait for ACK
            ack = _recv_int32(fd)
            if ack != ACK:
                raise RuntimeError(f"Expected ACK, got {ack}")
            sockets.append(NnSocket(fd))
            print(f"  Socket[{i}]: connected")

        # Signal readiness to all workers
        for ns in sockets:
            _send_int32(ns.fd, ACK)

        print("Network is initialized")
        return cls(sockets)

    # --- I/O on a single socket --------------------------------------------

    def write(self, socket_index: int, data: bytes) -> None:
        """Send *data* length-prefixed to one socket."""
        assert 0 <= socket_index < self.n_sockets
        sock = self._sockets[socket_index]
        _send_int32(sock, len(data))
        _send_all(sock, data)
        self._sent_bytes[socket_index] += 4 + len(data)

    def read(self, socket_index: int, size: int) -> bytes:
        """Receive exactly *size* bytes (length-prefixed) from one socket."""
        assert 0 <= socket_index < self.n_sockets
        sock = self._sockets[socket_index]
        prefix = _recv_int32(sock)  # read size header
        self._recv_bytes[socket_index] += 4
        data = _recv_all(sock, prefix)
        self._recv_bytes[socket_index] += prefix
        return data

    def _write_raw(self, socket_index: int, data: bytes) -> None:
        """Write raw bytes (no length prefix) to one socket."""
        assert 0 <= socket_index < self.n_sockets
        _send_all(self._sockets[socket_index], data)
        self._sent_bytes[socket_index] += len(data)

    def _read_raw(self, socket_index: int, size: int) -> bytes:
        """Read exactly *size* raw bytes (no length prefix)."""
        assert 0 <= socket_index < self.n_sockets
        data = _recv_all(self._sockets[socket_index], size)
        self._recv_bytes[socket_index] += size
        return data

    def try_read_with_max_attempts(self, socket_index: int, size: int,
                                   max_attempts: int) -> bytes:
        """Read *size* raw bytes with a maximum number of retry attempts.

        Port of C++ ``NnNetwork::tryReadWithMaxAttempts``.
        Used by ``WorkerLlmInference`` when the socket is in non-blocking /
        turbo mode.
        """
        assert 0 <= socket_index < self.n_sockets
        sock = self._sockets[socket_index]
        for _ in range(max_attempts):
            try:
                data = _recv_all(sock, size)
                self._recv_bytes[socket_index] += size
                return data
            except (BlockingIOError, OSError):
                continue
        return b""

    # --- ACK ---------------------------------------------------------------

    def write_ack(self, socket_index: int) -> None:
        """Send ACK packet (int32 == ACK constant)."""
        assert 0 <= socket_index < self.n_sockets
        _send_int32(self._sockets[socket_index], ACK)
        self._sent_bytes[socket_index] += 4

    def read_ack(self, socket_index: int) -> None:
        """Read and verify ACK packet."""
        assert 0 <= socket_index < self.n_sockets
        packet = _recv_int32(self._sockets[socket_index])
        self._recv_bytes[socket_index] += 4
        if packet != ACK:
            raise RuntimeError(f"Invalid ACK packet: {packet}")

    # --- Broadcast & many-at-once -----------------------------------------

    def write_all(self, data: bytes) -> None:
        """Broadcast *data* to ALL sockets via ``write_many``."""
        n = self.n_sockets
        if n == 0:
            return
        ios = [(i, data) for i in range(n)]
        self.write_many(ios)

    def write_many(self, ios: List[Tuple[int, bytes]]) -> None:
        """Write to multiple sockets in parallel.

        *ios* is a list of ``(socket_index, data_bytes)`` tuples.
        Each entry may be mutated in place -- do not reuse the list.
        """
        if not ios:
            return

        # Pre-account bytes so stats are correct before the send loop
        for idx, data in ios:
            if not data:
                continue
            self._sent_bytes[idx] += len(data)

        # Convert to mutable state: list of [socket_index, data_view, remaining]
        pending = []
        for idx, data in ios:
            if data:
                pending.append([idx, memoryview(data), len(data)])

        while pending:
            still_pending = []
            for item in pending:
                idx, view, remaining = item
                chunk_size = min(remaining, MAX_CHUNK_SIZE)
                offset = len(view) - remaining
                try:
                    n = self._sockets[idx].send(view[offset:offset + chunk_size])
                except BlockingIOError:
                    still_pending.append(item)
                    continue
                if n == 0:
                    raise ConnectionError(f"Socket {idx} closed during send")
                new_remaining = remaining - n
                if new_remaining > 0:
                    item[2] = new_remaining
                    still_pending.append(item)
            if len(still_pending) == len(pending):
                # No progress -- brief yield to avoid busy-wait
                time.sleep(0.0)
            pending = still_pending

    def read_many(self, ios: List[Tuple[int, int]]) -> List[bytes]:
        """Read from multiple sockets. Returns list of bytes in the same order.

        *ios* is a list of ``(socket_index, size)`` tuples.
        """
        if not ios:
            return []

        # Pre-account stats
        for idx, size in ios:
            self._recv_bytes[idx] += size

        # Allocate result buffers and track progress
        results: List[Optional[bytes]] = [None] * len(ios)
        pending = []
        for orig_idx, (sock_idx, size) in enumerate(ios):
            if size == 0:
                results[orig_idx] = b""
                continue
            buf = bytearray(size)
            pending.append([orig_idx, sock_idx, memoryview(buf), size])

        while pending:
            still_pending = []
            for item in pending:
                orig_idx, sock_idx, view, remaining = item
                chunk_size = min(remaining, MAX_CHUNK_SIZE)
                offset = len(view) - remaining
                try:
                    n = self._sockets[sock_idx].recv_into(
                        view[offset:offset + chunk_size], chunk_size)
                except BlockingIOError:
                    still_pending.append(item)
                    continue
                if n == 0:
                    raise ConnectionError(f"Socket {sock_idx} closed during recv")
                new_remaining = remaining - n
                if new_remaining > 0:
                    item[3] = new_remaining
                    still_pending.append(item)
                else:
                    results[orig_idx] = bytes(view)
            if len(still_pending) == len(pending):
                time.sleep(0.0)
            pending = still_pending

        # Fill any stragglers
        for item in pending:
            results[item[0]] = bytes(item[2])

        return [r for r in results]  # type: ignore[return-value]

    # --- Stats -------------------------------------------------------------

    def get_stats(self) -> Tuple[int, int]:
        """Return ``(total_sent, total_recv)`` bytes and reset counters."""
        sent = sum(self._sent_bytes)
        recv = sum(self._recv_bytes)
        self.reset_stats()
        return sent, recv

    def reset_stats(self) -> None:
        """Zero the per-socket byte counters."""
        for i in range(self.n_sockets):
            self._sent_bytes[i] = 0
            self._recv_bytes[i] = 0

    # --- Turbo mode --------------------------------------------------------

    def set_turbo(self, enabled: bool) -> None:
        """Enable / disable TCP_NODELAY (and non-blocking) on all sockets."""
        if enabled:
            for sock in self._sockets:
                _set_no_delay(sock)
                _set_quick_ack(sock)
                sock.setblocking(False)
        else:
            for sock in self._sockets:
                sock.setblocking(True)

    # --- Cleanup -----------------------------------------------------------

    def __del__(self):
        for sock in self._sockets:
            try:
                sock.close()
            except OSError:
                pass
        print("Network is closed")


# ---------------------------------------------------------------------------
# Low-level socket helpers
# ---------------------------------------------------------------------------

def _set_no_delay(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _set_quick_ack(sock: socket.socket) -> None:
    """Enable TCP_QUICKACK if available on this platform."""
    try:
        TCP_QUICKACK = getattr(socket, "TCP_QUICKACK", None)
        if TCP_QUICKACK is not None:
            sock.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)
    except OSError:
        pass


def _connect_socket(host: str, port: int) -> socket.socket:
    """Create socket, connect to *host:port*, set TCP options."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except OSError:
        sock.close()
        raise
    _set_no_delay(sock)
    _set_quick_ack(sock)
    return sock


# ===========================================================================
# Weight splitting utilities (pure-Python equivalents of the C++ split fns)
# ===========================================================================

def _block_size_for_type(ftype: int) -> int:
    if ftype in (F_32, F_16):
        return 1
    if ftype in (F_Q40, F_Q80):
        return 32  # both Q40 and Q80 use block_size=32
    raise ValueError(f"Unknown float type: {ftype}")


def _block_bytes_for_type(ftype: int) -> int:
    if ftype == F_32:
        return 4
    if ftype == F_16:
        return 2
    if ftype == F_Q40:
        return 2 + 16   # d (uint16) + qs (uint8[16])
    if ftype == F_Q80:
        return 2 + 32   # d (uint16) + qs (int8[32])
    raise ValueError(f"Unknown float type: {ftype}")


def split_row_matmul_weight(slice_obj: Any, node_index: int,
                            weight: bytes, temp: bytearray) -> None:
    """Copy the node's column-slice from the full quantised weight into *temp*.

    For row-matmul distribution each node owns columns
    [node_index*d0 .. (node_index+1)*d0) of the logical weight matrix
    ``(d output-rows, n/bs blocks-per-row)``.

    Q40 data is stored output-row-major: for each of d output rows,
    there are n/bs blocks (each covering bs=32 input elements).
    This function extracts d0 contiguous rows for the node.
    """
    ftype = slice_obj.type
    n_nodes = slice_obj.n_nodes
    d = slice_obj.size.x               # full output dimension (elements)
    d0 = slice_obj.slice_size.x        # this node's output dimension
    n = slice_obj.size.y               # full input dimension (elements)

    bs = _block_size_for_type(ftype)
    block_bytes = _block_bytes_for_type(ftype)
    blocks_per_row = n // bs           # blocks per output row

    src_row_start = node_index * d0    # first output row owned by this node
    row_bytes = blocks_per_row * block_bytes

    for row in range(d0):
        src_begin = (src_row_start + row) * row_bytes
        src_slice = weight[src_begin:src_begin + row_bytes]
        dst_begin = row * row_bytes
        temp[dst_begin:dst_begin + len(src_slice)] = src_slice


def split_col_matmul_weight(slice_obj: Any, node_index: int,
                            weight: bytes, temp: bytearray) -> None:
    """Copy the node's column-slice from the full quantised weight into *temp*.

    The file stores the full weight as *n* rows with *d/bs* blocks per row.
    The matmul expects *d* rows with *n0/bs* blocks per row (where n0=n/n_nodes).

    This function transposes on the fly: for each of the *d* output rows it
    copies the *n0/bs* blocks belonging to this node from the corresponding
    file row.
    """
    ftype = slice_obj.type
    n_nodes = slice_obj.n_nodes
    n = slice_obj.n             # full input dimension (file rows)
    n0 = slice_obj.n0           # per-node input dimension
    d = slice_obj.d             # full output dimension

    bs = _block_size_for_type(ftype)
    block_bytes = _block_bytes_for_type(ftype)
    blocks_per_file_row = n // bs           # blocks in one file row
    file_row_bytes = blocks_per_file_row * block_bytes

    blocks_per_output_row = n0 // bs        # blocks per output row for this node
    output_row_bytes = blocks_per_output_row * block_bytes

    block_offset = node_index * blocks_per_output_row

    for out_row in range(d):
        src_begin = out_row * file_row_bytes + block_offset * block_bytes
        dst_begin = out_row * output_row_bytes
        temp[dst_begin:dst_begin + output_row_bytes] = \
            weight[src_begin:src_begin + output_row_bytes]



# ===========================================================================
# NnNetworkNodeSynchronizer
# ===========================================================================

class NnNetworkNodeSynchronizer:
    """Implements ``NnNodeSynchronizer`` protocol for distributed tensor sync."""

    def __init__(self, network: NnNetwork, execution: Any,
                 net_config: NnNetConfig, node_config: NnNodeConfig):
        self.network = network
        self.execution = execution         # has .pipes and .batchSize
        self.net_config = net_config
        self.node_config = node_config

    # -------------------------------------------------------------------

    def sync(self, segment_index: int, n_threads: int, thread_index: int) -> None:
        """Synchronise pipe data between nodes for one segment.

        Called by ALL threads, but only *thread_index == 0* performs I/O.
        """
        segment = self.node_config.segments[segment_index]

        for sync_cfg in segment.syncs:
            pipe_index = sync_cfg.pipe_index
            pipe = self.execution.pipes[pipe_index]
            pipe_cfg = self.net_config.pipes[pipe_index]
            batch_bytes = get_bytes(pipe_cfg.size.float_type, pipe_cfg.size.x)

            for batch_idx in range(self.execution.batch_size):
                pipe_batch = bytearray(pipe[batch_idx * batch_bytes:
                                           (batch_idx + 1) * batch_bytes])

                if sync_cfg.sync_type == NnSyncType.SYNC_WITH_ROOT:
                    self._sync_with_root(pipe_batch, batch_bytes,
                                         n_threads, thread_index)
                elif sync_cfg.sync_type == NnSyncType.SYNC_NODE_SLICES:
                    self._sync_node_slices(pipe_batch, batch_bytes,
                                           n_threads, thread_index,
                                           only_worker_to_root=False)
                elif sync_cfg.sync_type == NnSyncType.SYNC_NODE_SLICES_EXCEPT_ROOT:
                    self._sync_node_slices(pipe_batch, batch_bytes,
                                           n_threads, thread_index,
                                           only_worker_to_root=True)
                else:
                    raise ValueError(f"Unknown sync type: {sync_cfg.sync_type}")

                # Copy modified data back to numpy pipe
                pipe[batch_idx * batch_bytes:
                     (batch_idx + 1) * batch_bytes] = np.frombuffer(
                    pipe_batch, dtype=pipe.dtype)


    # -------------------------------------------------------------------

    def _sync_with_root(self, buffer: bytearray, n_bytes: int,
                        n_threads: int, thread_index: int) -> None:
        node_index = self.node_config.node_index
        n_sockets = self.network.n_sockets

        if node_index == 0:
            # Root: send to all workers (distribute across threads)
            n_per_thread = (n_sockets // n_threads +
                            (1 if (n_sockets % n_threads) > thread_index else 0))
            if n_per_thread == 0:
                return
            ios = []
            for i in range(n_per_thread):
                sock_idx = thread_index + i * n_threads
                ios.append((sock_idx, buffer))  # buffer is bytes-like
            self.network.write_many(ios)
        else:
            # Worker: receive from root (thread 0 only)
            if thread_index != 0:
                return
            raw = self.network._read_raw(ROOT_SOCKET_INDEX, n_bytes)
            buffer[:n_bytes] = raw

    def _sync_node_slices(self, buffer: bytearray, n_bytes: int,
                          n_threads: int, thread_index: int,
                          only_worker_to_root: bool) -> None:
        node_index = self.node_config.node_index
        is_worker = node_index != 0
        n_nodes = self.net_config.n_nodes
        n_sockets = self.network.n_sockets
        slice_bytes = n_bytes // n_nodes

        # How many sockets this thread handles
        effective = (1 if (only_worker_to_root and is_worker) else n_sockets)
        n_per_thread = (effective // n_threads +
                        (1 if (effective % n_threads) > thread_index else 0))
        if n_per_thread == 0:
            return

        # ----- phase 1: send my slice to relevant sockets -----
        if not only_worker_to_root or is_worker:
            my_slice = bytes(buffer[node_index * slice_bytes:
                                    (node_index + 1) * slice_bytes])
            ios = []
            for i in range(n_per_thread):
                sock_idx = thread_index + i * n_threads
                if only_worker_to_root:
                    sock_idx = ROOT_SOCKET_INDEX  # workers only send to root
                ios.append((sock_idx, my_slice))
            self.network.write_many(ios)

        # ----- phase 2: receive slices from other nodes -----
        if not only_worker_to_root or not is_worker:
            ios = []
            for i in range(n_per_thread):
                sock_idx = thread_index + i * n_threads
                slice_index = sock_idx if sock_idx < node_index else sock_idx + 1
                dst_start = slice_index * slice_bytes
                # Build a mutable bytearray view for read_many
                ios.append((sock_idx, slice_bytes))
            results = self.network.read_many(ios)

            # Copy received data back into buffer
            for i, data in enumerate(results):
                sock_idx = thread_index + i * n_threads
                slice_index = sock_idx if sock_idx < node_index else sock_idx + 1
                dst_start = slice_index * slice_bytes
                buffer[dst_start:dst_start + slice_bytes] = data


# ===========================================================================
# Config serialisation helpers (binary protocol, little-endian)
# ===========================================================================

_FMT_INT = "<i"


def _pack_int(v: int) -> bytes:
    return struct.pack(_FMT_INT, v)


def _pack_size3d(sz: NnSize3D) -> bytes:
    return struct.pack("<iiii", sz.float_type, sz.z, sz.y, sz.x)


def _unpack_size3d(data: bytes, offset: int = 0) -> Tuple[NnSize3D, int]:
    ft, z, y, x = struct.unpack_from("<iiii", data, offset)
    return NnSize3D(float_type=ft, z=z, y=y, x=x), offset + 16


def _pack_pointer(p: NnPointerConfig) -> bytes:
    return struct.pack("<iii", p.source, p.pointer_index, p.pointer_type)


def _unpack_pointer(data: bytes, offset: int) -> Tuple[NnPointerConfig, int]:
    src, idx, typ = struct.unpack_from("<iii", data, offset)
    return NnPointerConfig(source=src, pointer_index=idx, pointer_type=typ), offset + 12


# ===========================================================================
# NnRootConfigWriter
# ===========================================================================

class NnRootConfigWriter:
    """Serialise and send net/node configs from root to workers."""

    def __init__(self, network: NnNetwork):
        self.network = network

    # -------------------------------------------------------------------

    def write_net(self, socket_index: int, config: NnNetConfig) -> None:
        net = self.network
        net.write_ack(socket_index)
        net._write_raw(socket_index, _pack_int(config.n_batches))
        net._write_raw(socket_index, _pack_int(config.n_nodes))
        net._write_raw(socket_index, _pack_int(len(config.pipes)))
        for pipe in config.pipes:
            net._write_raw(socket_index, _pack_size3d(pipe.size))
            _send_string(net._sockets[socket_index], pipe.name)
        net._write_raw(socket_index, _pack_int(len(config.pre_syncs)))
        for ps in config.pre_syncs:
            net._write_raw(socket_index, _pack_int(ps.pipe_index))
        net.read_ack(socket_index)

    def write_node(self, socket_index: int, config: NnNodeConfig) -> None:
        net = self.network
        net.write_ack(socket_index)
        net._write_raw(socket_index, _pack_int(config.node_index))
        net._write_raw(socket_index, _pack_int(len(config.buffers)))
        net._write_raw(socket_index, _pack_int(len(config.segments)))

        # Buffers
        for buf in config.buffers:
            net._write_raw(socket_index, _pack_size3d(buf.size))
            _send_string(net._sockets[socket_index], buf.name)

        # Segments
        for seg in config.segments:
            net._write_raw(socket_index, _pack_int(len(seg.syncs)))
            net._write_raw(socket_index, _pack_int(len(seg.ops)))

            # Syncs
            for sync_cfg in seg.syncs:
                net._write_raw(socket_index, _pack_int(sync_cfg.pipe_index))
                net._write_raw(socket_index, _pack_int(sync_cfg.sync_type))

            # Ops
            for op in seg.ops:
                net._write_raw(socket_index, _pack_int(op.code))
                net._write_raw(socket_index, _pack_int(op.index))
                net._write_raw(socket_index, _pack_size3d(op.weight_size))

                # config size and bytes
                config_bytes = _serialize_op_config(op.config)
                net._write_raw(socket_index, _pack_int(len(config_bytes)))

                _send_string(net._sockets[socket_index], op.name)
                net._write_raw(socket_index, _pack_pointer(op.input))
                net._write_raw(socket_index, _pack_pointer(op.output))
                if config_bytes:
                    net._write_raw(socket_index, config_bytes)

        net.read_ack(socket_index)

    def write_to_workers(self, net_config: NnNetConfig,
                         node_configs: List[NnNodeConfig]) -> None:
        """Send net config + each worker's node config."""
        for node_index in range(1, net_config.n_nodes):
            socket_index = node_index - 1
            self.write_net(socket_index, net_config)
            self.write_node(socket_index, node_configs[node_index])


# ===========================================================================
# NnWorkerConfigReader
# ===========================================================================

class NnWorkerConfigReader:
    """Receive and deserialise net/node configs on a worker."""

    def __init__(self, network: NnNetwork):
        self.network = network

    # -------------------------------------------------------------------

    def read_net(self) -> NnNetConfig:
        net = self.network
        net.read_ack(ROOT_SOCKET_INDEX)

        n_batches = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
        # track bytes manually for stats -- _recv_int32 doesn't go through
        # _read_raw so stats are lost; minor, acceptable
        n_nodes = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
        n_pipes = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])

        pipes = []
        for _ in range(n_pipes):
            raw_size = net._read_raw(ROOT_SOCKET_INDEX, 16)
            sz, _ = _unpack_size3d(raw_size)
            name = _recv_string(net._sockets[ROOT_SOCKET_INDEX])
            pipes.append(NnPipeConfig(name=name, size=sz))

        n_pre_syncs = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
        pre_syncs = []
        for _ in range(n_pre_syncs):
            pi = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
            pre_syncs.append(NnPreSyncConfig(pipe_index=pi))

        net.write_ack(ROOT_SOCKET_INDEX)
        return NnNetConfig(n_batches=n_batches, n_nodes=n_nodes,
                           pipes=pipes, pre_syncs=pre_syncs)

    def read_node(self) -> NnNodeConfig:
        net = self.network
        net.read_ack(ROOT_SOCKET_INDEX)

        node_index = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
        n_buffers = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
        n_segments = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])

        buffers = []
        for _ in range(n_buffers):
            raw_size = net._read_raw(ROOT_SOCKET_INDEX, 16)
            sz, _ = _unpack_size3d(raw_size)
            name = _recv_string(net._sockets[ROOT_SOCKET_INDEX])
            buffers.append(NnBufferConfig(name=name, size=sz))

        segments = []
        for _ in range(n_segments):
            n_syncs = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
            n_ops = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])

            syncs = []
            for _ in range(n_syncs):
                pi = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
                st = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
                syncs.append(NnSyncConfig(pipe_index=pi, sync_type=st))

            ops = []
            for _ in range(n_ops):
                code = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
                idx = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
                raw_wsize = net._read_raw(ROOT_SOCKET_INDEX, 16)
                wsize, _ = _unpack_size3d(raw_wsize)
                config_size = _recv_int32(net._sockets[ROOT_SOCKET_INDEX])
                name = _recv_string(net._sockets[ROOT_SOCKET_INDEX])

                raw_input = net._read_raw(ROOT_SOCKET_INDEX, 12)
                inp, _ = _unpack_pointer(raw_input, 0)
                raw_output = net._read_raw(ROOT_SOCKET_INDEX, 12)
                outp, _ = _unpack_pointer(raw_output, 0)

                op_config = None
                if config_size > 0:
                    config_bytes = net._read_raw(ROOT_SOCKET_INDEX, config_size)
                    op_config = _deserialize_op_config(code, config_bytes)

                ops.append(NnOpConfig(code=code, name=name, index=idx,
                                      input=inp, output=outp,
                                      weight_size=wsize, config=op_config))

            segments.append(NnSegmentConfig(ops=ops, syncs=syncs))

        net.write_ack(ROOT_SOCKET_INDEX)
        return NnNodeConfig(node_index=node_index, buffers=buffers,
                            segments=segments)


# ===========================================================================
# Op-config serialisation (mirrors the pickling of op-specific dataclasses)
# ===========================================================================

# Registry: op code -> dataclass type
_OP_CONFIG_TYPES: dict = {}

import sys as _sys

_MODULE = _sys.modules[__name__]


def _register_op_config(code: int, cls: type) -> None:
    _OP_CONFIG_TYPES[code] = cls


def _serialize_op_config(config: Any) -> bytes:
    """Pack an op-specific config dataclass into bytes (custom per type)."""
    if config is None:
        return b""

    from .graph_builder import (
        NnEmbeddingOpConfig, NnInvRmsOpConfig, NnRmsNormOpConfig,
        NnMatmulOpConfig, NnRopeOpConfig, NnMultiHeadAttOpConfig,
        NnMergeAddOpCodeConfig, NnMergeSumOpCodeConfig,
        NnSiluOpCodeConfig, NnMulOpCodeConfig, NnScaleOpCodeConfig,
        NnCastOpCodeConfig, NnRepeatZOpCodeConfig, NnShiftOpCodeConfig,
        NnSoftmaxOpCodeConfig, NnMoeGateOpCodeConfig,
    )

    cls = type(config)
    name = cls.__name__

    if isinstance(config, NnEmbeddingOpConfig):
        return _pack_int(0)  # no fields
    elif isinstance(config, NnInvRmsOpConfig):
        return struct.pack("<if", config.n_columns, config.epsilon)
    elif isinstance(config, NnRmsNormOpConfig):
        return struct.pack("<ii", config.inv_rms_buffer_index, config.n_columns)
    elif isinstance(config, NnMatmulOpConfig):
        return struct.pack("<iii",
                           config.n_experts,
                           config.n_active_experts,
                           config.active_expert_indexes_buffer_index)
    elif isinstance(config, NnRopeOpConfig):
        from .graph_builder import NnRopeSlice
        result = struct.pack("<iiiifffi",
                           config.rope_type, config.is_q,
                           config.position_pipe_index,
                           config.rope_cache_buffer_index,
                           config.rope_scaling_factor,
                           config.rope_scaling_low_freq_factor,
                           config.rope_scaling_high_freq_factor,
                           config.rope_scaling_orig_max_seq_len)
        slc = config.slice
        if slc is not None:
            slice_bytes = struct.pack("<iiiiiiiiiiifiiii",
                slc.q_dim0, slc.q_dim_start, slc.q_dim_end,
                slc.q_shift, slc.kv_dim, slc.kv_dim0,
                slc.kv_dim_start, slc.slice_dim, slc.seq_len,
                slc.head_dim, slc.n_kv_heads, slc.rope_theta,
                slc.cache_size.float_type, slc.cache_size.z,
                slc.cache_size.y, slc.cache_size.x)
            result += slice_bytes
        return result
    elif isinstance(config, NnMultiHeadAttOpConfig):
        return struct.pack("<iiiiiiiiiiii",
                           config.n_heads, config.n_heads0,
                           config.n_kv_heads, config.head_dim,
                           config.seq_len, config.q_slice_d0,
                           config.kv_dim0, config.position_pipe_index,
                           config.query_buffer_index,
                           config.key_cache_buffer_index,
                           config.value_cache_buffer_index,
                           config.att_buffer_index)
    elif isinstance(config, (NnMergeAddOpCodeConfig, NnMergeSumOpCodeConfig,
                             NnSiluOpCodeConfig, NnCastOpCodeConfig,
                             NnRepeatZOpCodeConfig, NnSoftmaxOpCodeConfig)):
        return _pack_int(0)  # no fields
    elif isinstance(config, NnMulOpCodeConfig):
        return _pack_int(config.multiplier_buffer_index)
    elif isinstance(config, NnScaleOpCodeConfig):
        return _pack_int(config.scale_buffer_index)
    elif isinstance(config, NnShiftOpCodeConfig):
        return _pack_int(config.index_pipe_index)
    elif isinstance(config, NnMoeGateOpCodeConfig):
        return struct.pack("<iii", config.k, config.norm_topk,
                           config.indexes_buffer_index)
    else:
        # Fallback: pickle
        import pickle
        return pickle.dumps(config)


def _deserialize_op_config(code: int, data: bytes) -> Any:
    """Reconstruct an op-specific config dataclass from bytes."""
    from .graph_builder import (
        NnEmbeddingOpConfig, NnInvRmsOpConfig, NnRmsNormOpConfig,
        NnMatmulOpConfig, NnRopeOpConfig, NnMultiHeadAttOpConfig,
        NnMergeAddOpCodeConfig, NnMergeSumOpCodeConfig,
        NnSiluOpCodeConfig, NnMulOpCodeConfig, NnScaleOpCodeConfig,
        NnCastOpCodeConfig, NnRepeatZOpCodeConfig, NnShiftOpCodeConfig,
        NnSoftmaxOpCodeConfig, NnMoeGateOpCodeConfig,
    )
    from .model import NnOpCode

    if code == NnOpCode.EMBEDDING:
        return NnEmbeddingOpConfig()
    elif code == NnOpCode.INV_RMS:
        n_cols, eps = struct.unpack("<if", data[:8])
        return NnInvRmsOpConfig(n_columns=n_cols, epsilon=eps)
    elif code == NnOpCode.RMS_NORM:
        inv_buf, n_cols = struct.unpack("<ii", data[:8])
        return NnRmsNormOpConfig(inv_rms_buffer_index=inv_buf,
                                 n_columns=n_cols)
    elif code == NnOpCode.MATMUL:
        n_exp, n_act, act_buf = struct.unpack("<iii", data[:12])
        return NnMatmulOpConfig(
            n_experts=n_exp,
            n_active_experts=n_act,
            active_expert_indexes_buffer_index=act_buf)
    elif code == NnOpCode.ROPE:
        rt, is_q, ppi, rcbi, rsf, rslf, rshf, rsoms = \
            struct.unpack("<iiiifffi", data[:32])
        config = NnRopeOpConfig(
            rope_type=rt, is_q=is_q,
            position_pipe_index=ppi,
            rope_cache_buffer_index=rcbi,
            rope_scaling_factor=rsf,
            rope_scaling_low_freq_factor=rslf,
            rope_scaling_high_freq_factor=rshf,
            rope_scaling_orig_max_seq_len=rsoms)
        if len(data) > 32:
            from .graph_builder import NnRopeSlice
            vals = struct.unpack("<iiiiiiiiiiifiiii", data[32:])
            slc = NnRopeSlice()
            slc.q_dim0 = vals[0]
            slc.q_dim_start = vals[1]
            slc.q_dim_end = vals[2]
            slc.q_shift = vals[3]
            slc.kv_dim = vals[4]
            slc.kv_dim0 = vals[5]
            slc.kv_dim_start = vals[6]
            slc.slice_dim = vals[7]
            slc.seq_len = vals[8]
            slc.head_dim = vals[9]
            slc.n_kv_heads = vals[10]
            slc.rope_theta = vals[11]
            slc.cache_size.float_type = vals[12]
            slc.cache_size.z = vals[13]
            slc.cache_size.y = vals[14]
            slc.cache_size.x = vals[15]
            config.slice = slc
        return config
    elif code == NnOpCode.MULTIHEAD_ATT:
        vals = struct.unpack("<iiiiiiiiiiii", data[:48])
        return NnMultiHeadAttOpConfig(
            n_heads=vals[0], n_heads0=vals[1],
            n_kv_heads=vals[2], head_dim=vals[3],
            seq_len=vals[4], q_slice_d0=vals[5],
            kv_dim0=vals[6], position_pipe_index=vals[7],
            query_buffer_index=vals[8],
            key_cache_buffer_index=vals[9],
            value_cache_buffer_index=vals[10],
            att_buffer_index=vals[11])
    elif code == NnOpCode.MERGE_ADD:
        return NnMergeAddOpCodeConfig()
    elif code == NnOpCode.MERGE_SUM:
        return NnMergeSumOpCodeConfig()
    elif code == NnOpCode.GELU:
        return NnSiluOpCodeConfig()  # GELU has no dedicated config class
    elif code == NnOpCode.SILU:
        return NnSiluOpCodeConfig()
    elif code == NnOpCode.MUL:
        mb = struct.unpack("<i", data[:4])[0]
        return NnMulOpCodeConfig(multiplier_buffer_index=mb)
    elif code == NnOpCode.SCALE:
        sb = struct.unpack("<i", data[:4])[0]
        return NnScaleOpCodeConfig(scale_buffer_index=sb)
    elif code == NnOpCode.CAST:
        return NnCastOpCodeConfig()
    elif code == NnOpCode.REPEAT_Z:
        return NnRepeatZOpCodeConfig()
    elif code == NnOpCode.SHIFT:
        ipi = struct.unpack("<i", data[:4])[0]
        return NnShiftOpCodeConfig(index_pipe_index=ipi)
    elif code == NnOpCode.SOFTMAX:
        return NnSoftmaxOpCodeConfig()
    elif code == NnOpCode.MOE_GATE:
        k, norm_topk, idx_buf = struct.unpack("<iii", data[:12])
        return NnMoeGateOpCodeConfig(
            k=k, norm_topk=norm_topk,
            indexes_buffer_index=idx_buf)
    else:
        # Fallback: unpickle
        import pickle
        return pickle.loads(data)


# ===========================================================================
# NnRootWeightLoader
# ===========================================================================

class NnRootWeightLoader:
    """Load weights on root and distribute to workers."""

    def __init__(self, executor: Any, network: NnNetwork, n_nodes: int):
        self.executor = executor      # has .load_weight(name, idx, off, n, data)
        self.network = network
        self.n_nodes = n_nodes
        self._temp = bytearray()
        self._temp_size = 0

    def _ensure_temp(self, size: int) -> None:
        if len(self._temp) < size:
            self._temp = bytearray(size)

    # -------------------------------------------------------------------

    def write_weight(self, node_index: int, op_name: str, op_index: int,
                     offset: int, n_bytes: int, weight: bytes) -> None:
        """Send one weight blob to a single worker."""
        socket_index = node_index - 1
        net = self.network
        _send_string(net._sockets[socket_index], op_name)
        net._write_raw(socket_index, _pack_int(op_index))
        net._write_raw(socket_index, _pack_int(offset))
        net._write_raw(socket_index, _pack_int(n_bytes))
        net._write_raw(socket_index, bytes(weight[:n_bytes]))

    def load_root(self, op_name: str, op_index: int,
                  n_bytes: int, weight_data: bytes) -> int:
        """Load weight for root node only."""
        self.executor.load_weight(op_name, op_index, 0, n_bytes, weight_data)
        return n_bytes

    def load_all(self, op_name: str, op_index: int,
                 n_bytes: int, weight_data: bytes) -> int:
        """Load weight on root AND broadcast to all workers."""
        self.executor.load_weight(op_name, op_index, 0, n_bytes, weight_data)
        if self.n_nodes > 1:
            for node_idx in range(1, self.n_nodes):
                self.write_weight(node_idx, op_name, op_index,
                                  0, n_bytes, weight_data)
        return n_bytes

    def load_row_matmul_slices(self, op_name: str, op_index: int,
                               expert_index: int, slice_obj: Any,
                               weight_data: bytes) -> int:
        """Distribute row-matmul weight slices (split columns) across nodes."""
        offset = expert_index * slice_obj.slice_size.n_bytes
        if self.n_nodes == 1:
            self.executor.load_weight(
                op_name, op_index, offset,
                slice_obj.slice_size.n_bytes, weight_data)
        else:
            n_bytes_per_node = slice_obj.slice_size.n_bytes
            self._ensure_temp(n_bytes_per_node)
            for node_idx in range(self.n_nodes):
                split_row_matmul_weight(
                    slice_obj, node_idx, weight_data, self._temp)
                if node_idx == 0:
                    self.executor.load_weight(
                        op_name, op_index, offset,
                        n_bytes_per_node, self._temp)
                else:
                    self.write_weight(
                        node_idx, op_name, op_index, offset,
                        n_bytes_per_node, self._temp)
        return slice_obj.size.n_bytes

    def load_col_matmul_slices(self, op_name: str, op_index: int,
                               expert_index: int, slice_obj: Any,
                               weight_data: bytes) -> int:
        """Distribute col-matmul weight slices (split rows) across nodes."""
        offset = expert_index * slice_obj.slice_size.n_bytes
        if self.n_nodes == 1:
            self.executor.load_weight(
                op_name, op_index, offset,
                slice_obj.slice_size.n_bytes, weight_data)
        else:
            n_bytes_per_node = slice_obj.slice_size.n_bytes
            self._ensure_temp(n_bytes_per_node)
            for node_idx in range(self.n_nodes):
                split_col_matmul_weight(
                    slice_obj, node_idx, weight_data, self._temp)
                if node_idx == 0:
                    self.executor.load_weight(
                        op_name, op_index, offset,
                        n_bytes_per_node, self._temp)
                else:
                    self.write_weight(
                        node_idx, op_name, op_index, offset,
                        n_bytes_per_node, self._temp)
        return slice_obj.size.n_bytes

    def finish(self) -> None:
        """Signal workers that weight loading is complete (zero-size name)."""
        for socket_idx in range(self.n_nodes - 1):
            _send_int32(self.network._sockets[socket_idx], 0)
            self.network.read_ack(socket_idx)
        self._temp = bytearray()


# ===========================================================================
# NnWorkerWeightReader
# ===========================================================================

class NnWorkerWeightReader:
    """Receive weights from root and load into the local executor."""

    def __init__(self, executor: Any, network: NnNetwork):
        self.executor = executor
        self.network = network
        self._temp = bytearray()

    def _ensure_temp(self, size: int) -> None:
        if len(self._temp) < size:
            self._temp = bytearray(size)

    def read(self) -> None:
        """Loop receiving ops/weights until finish signal (nameSize == 0)."""
        net = self.network
        sock = net._sockets[ROOT_SOCKET_INDEX]

        while True:
            name_size = _recv_int32(sock)
            if name_size == 0:
                net.write_ack(ROOT_SOCKET_INDEX)
                self._temp = bytearray()
                break

            op_name = _recv_all(sock, name_size)
            if op_name and op_name[-1:] == b"\0":
                op_name = op_name[:-1]
            op_name_str = op_name.decode("utf-8")

            op_index = _recv_int32(sock)
            offset = _recv_int32(sock)
            n_bytes = _recv_int32(sock)

            self._ensure_temp(n_bytes)
            raw = _recv_all(sock, n_bytes)
            self._temp[:n_bytes] = raw

            self.executor.load_weight(
                op_name_str, op_index, offset, n_bytes, self._temp)
            print(f"Loaded {op_name_str:>22s} {op_index:3d}, "
                  f"{n_bytes // 1024:12d} kB")

        print("Weights loaded")
