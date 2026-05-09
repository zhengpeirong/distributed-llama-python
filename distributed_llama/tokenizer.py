"""BPE tokenizer compatible with distributed-llama binary format.

Port of src/tokenizer.cpp — supports both old (0x567123) and new (0x567124)
tokenizer file formats, FNV-1a 64-bit hash lookup, BPE merge with vocab scores,
UTF-8 streaming decode with recovery, special token detection, and chat templates.
"""

import struct
from typing import List, Tuple, Optional


def _fnv1a_64(s: str) -> int:
    """FNV-1a 64-bit hash over raw bytes (matches C++ behavior).

    Token strings are stored as latin-1 (one char per byte), so encoding
    with latin-1 recovers the original byte sequence for hashing.
    """
    h = 1469598103934665603
    for byte in s.encode('latin-1'):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class Tokenizer:
    """BPE tokenizer with special token support and chat templates."""

    def __init__(self, tokenizer_path: str):
        self.bos_id: int = -1
        self.chat_template: Optional[str] = None
        self.add_bos: bool = False
        self.max_token_length: int = 0
        self.vocab_size: int = 0
        self.vocab: List[str] = []
        self.vocab_length: List[int] = []
        self.vocab_scores: List[float] = []
        self.eos_token_ids: List[int] = []

        with open(tokenizer_path, "rb") as f:
            magic = struct.unpack("<i", f.read(4))[0]

            if magic == 0x567123:
                self._read_old_format(f)
            elif magic == 0x567124:
                self._read_new_format(f)
            else:
                raise ValueError(f"Invalid tokenizer magic: {magic:#x}")

            self._read_vocab_entries(f)

        if self.max_token_length < 1:
            raise ValueError("Invalid max token length")

        # split regular vs special vocab (bos_id is the boundary)
        self.regular_vocab_size = self.bos_id if self.bos_id >= 0 else self.vocab_size
        self.special_vocab_size = self.vocab_size - self.regular_vocab_size

        # build regular vocab hash map
        self._regular_map: dict = {}
        for i in range(self.regular_vocab_size):
            h = _fnv1a_64(self.vocab[i])
            self._regular_map.setdefault(h, []).append(i)

        # build special vocab list
        self._special_vocab: List[Tuple[str, int]] = []
        for i in range(self.special_vocab_size):
            idx = i + self.regular_vocab_size
            self._special_vocab.append((self.vocab[idx], idx))

        # decode buffer
        str_buf_size = self.max_token_length * 2
        str_buf_size = max(str_buf_size, 8)
        self._str_buf_size = str_buf_size + 3
        self._str_buffer = bytearray(self._str_buf_size)
        self._str_buf_pos = 0

    def _read_old_format(self, f):
        header = struct.unpack("<IIiii", f.read(20))
        self.vocab_size = header[0]
        self.max_token_length = header[1]
        self.bos_id = header[2]
        self.eos_token_ids = [header[3]]

    def _read_new_format(self, f):
        header_size = struct.unpack("<i", f.read(4))[0]
        n_kv = (header_size - 8) // 8
        buffer = struct.unpack(f"<{n_kv * 2}i", f.read(n_kv * 8))

        version = -1
        chat_template_len = -1
        n_eos_tokens = 0
        i = 0
        while i < len(buffer):
            key = buffer[i]
            value = buffer[i + 1]
            i += 2

            if key == 0:  # VERSION
                version = value
            elif key == 1:  # VOCAB_SIZE
                self.vocab_size = value
            elif key == 2:  # MAX_TOKEN_LENGTH
                self.max_token_length = value
            elif key == 3:  # BOS_ID
                self.bos_id = value
            elif key == 4:  # EOS_ID (backward compat)
                self.eos_token_ids.append(value)
            elif key == 6:  # CHAT_EOS_ID (backward compat)
                self.eos_token_ids.append(value)
            elif key == 7:  # CHAT_TEMPLATE
                chat_template_len = value
            elif key == 8:  # CHAT_STOP (ignored)
                f.seek(value, 1)
            elif key == 5:  # PAD_ID (ignored)
                pass
            elif key == 9:  # N_EOS_TOKENS
                n_eos_tokens = value
            elif key == 10:  # ADD_BOS
                self.add_bos = value == 1
            else:
                raise ValueError(f"Invalid tokenizer header key: {key}")

        if version != 1:
            raise ValueError(
                f"Old tokenizer version ({version}), please regenerate"
            )

        if chat_template_len > 0:
            self.chat_template = f.read(chat_template_len).decode("utf-8")

        if n_eos_tokens > 0:
            for _ in range(n_eos_tokens):
                eos = struct.unpack("<i", f.read(4))[0]
                self.eos_token_ids.append(eos)

    def _read_vocab_entries(self, f):
        for _ in range(self.vocab_size):
            score, length = struct.unpack("<fI", f.read(8))
            token_bytes = f.read(length)
            token_str = token_bytes.decode("latin-1")
            self.vocab_scores.append(score)
            self.vocab_length.append(length)
            self.vocab.append(token_str)

    def print_header(self):
        if self.bos_id >= 0:
            bos = self.vocab[self.bos_id] if self.bos_id < len(self.vocab) else "?"
            print(f"  AddBos: {1 if self.add_bos else 0}")
            print(f"  BosId: {self.bos_id!r} ({bos})")
        if self.eos_token_ids:
            eos_info = " ".join(
                f"{eid} ({self.vocab[eid]})" for eid in self.eos_token_ids
            )
            print(f"  EosId: {eos_info}")
        print(f"  RegularVocabSize: {self.regular_vocab_size}")
        print(f"  SpecialVocabSize: {self.special_vocab_size}")

    def _find_special_token_startswith(self, text: str) -> int:
        """Check if any special token is a prefix of the given text."""
        for token_str, token_id in self._special_vocab:
            if text.startswith(token_str):
                return token_id
        return -1

    def _find_regular_token(self, piece: str) -> int:
        h = _fnv1a_64(piece)
        candidates = self._regular_map.get(h)
        if candidates is None:
            return -1
        for tid in candidates:
            if self.vocab[tid] == piece:
                return tid
        return -1

    def is_eos(self, token: int) -> bool:
        return token in self.eos_token_ids

    def reset_decoder(self):
        self._str_buf_pos = 0

    def decode(self, token: int) -> Optional[str]:
        """Decode a single token. Returns decoded UTF-8 string or None."""
        if token == self.bos_id:
            return None
        if self.is_eos(token):
            if self._str_buf_pos > 0:
                return self._str_buffer[: self._str_buf_pos].decode(
                    "utf-8", errors="replace"
                )
            return None

        piece = self.vocab[token]
        piece_len = self.vocab_length[token]

        self._str_buffer[self._str_buf_pos : self._str_buf_pos + piece_len] = piece.encode(
            "latin-1"
        )
        self._str_buf_pos += piece_len
        self._str_buffer[self._str_buf_pos] = 0

        return self._detok_utf8()

    def _detok_utf8(self) -> Optional[str]:
        """Streaming UTF-8 decode with recovery."""
        src = memoryview(self._str_buffer)[: self._str_buf_pos + 1]
        dst = bytearray(self._str_buf_size)
        checkpoint_src = 0
        checkpoint_dst = 0
        expect_cont = 0
        si = 0
        di = 0

        while si < self._str_buf_pos:
            c = src[si]
            need_recovery = False

            if expect_cont > 0:
                if (c & 0xC0) == 0x80:
                    dst[di] = c
                    di += 1
                    si += 1
                    expect_cont -= 1
                else:
                    need_recovery = True
            elif c <= 0x7F:
                dst[di] = c
                di += 1
                si += 1
            elif 0xC0 <= c <= 0xDF:
                dst[di] = c
                di += 1
                si += 1
                expect_cont = 1
            elif 0xE0 <= c <= 0xEF:
                dst[di] = c
                di += 1
                si += 1
                expect_cont = 2
            elif 0xF0 <= c <= 0xF7:
                dst[di] = c
                di += 1
                si += 1
                expect_cont = 3
            else:
                need_recovery = True

            if not need_recovery:
                if expect_cont == 0:
                    checkpoint_dst = di
                    checkpoint_src = si
            else:
                if expect_cont > 0:
                    expect_cont = 0
                else:
                    si += 1
                di = checkpoint_dst
                dst[di] = 0xEF
                dst[di + 1] = 0xBF
                dst[di + 2] = 0xBD
                di += 3

        # preserve incomplete sequence
        if si > checkpoint_src:
            remaining = si - checkpoint_src
            self._str_buffer[:remaining] = self._str_buffer[
                checkpoint_src : checkpoint_src + remaining
            ]
            self._str_buf_pos = remaining
        else:
            self._str_buf_pos = 0

        if checkpoint_dst > 0:
            return dst[:checkpoint_dst].decode("utf-8", errors="replace")
        return None

    def _get_special_bytes_map(self):
        """Return list of (token_bytes, token_id) for byte-level matching."""
        if not hasattr(self, '_special_bytes_cache'):
            self._special_bytes_cache = [
                (s.encode('latin-1'), tid) for s, tid in self._special_vocab
            ]
        return self._special_bytes_cache

    def encode(
        self,
        text: str,
        is_start: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs using BPE with merge loop."""
        if text is None:
            raise ValueError("Input text is null")

        # Work with UTF-8 bytes so multi-byte characters are handled
        # byte-by-byte, matching the C++ BPE tokenizer behavior.
        text_bytes = text.encode('utf-8')

        tokens = []

        if is_start and self.add_bos and self.bos_id >= 0:
            tokens.append(self.bos_id)

        str_buf = bytearray(self._str_buf_size)
        str_len = 0

        special_bytes_map = self._get_special_bytes_map()

        i = 0
        while i < len(text_bytes):
            if add_special_tokens:
                remaining = text_bytes[i:]
                special_id = -1
                for token_bytes, tid in special_bytes_map:
                    if remaining.startswith(token_bytes):
                        special_id = tid
                        break
                if special_id >= 0:
                    tokens.append(special_id)
                    i += self.vocab_length[special_id]
                    continue

            str_buf[str_len] = text_bytes[i]
            str_len += 1
            str_buf[str_len] = 0

            piece = str_buf[:str_len].decode("latin-1")
            tid = self._find_regular_token(piece)
            if tid != -1:
                tokens.append(tid)
                str_len = 0
            i += 1

        if str_len != 0:
            raise ValueError(
                f"Cannot tokenize remaining '{str_buf[:str_len].decode('latin-1')}'"
            )

        # BPE merge loop
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for j in range(len(tokens) - 1):
                t0 = tokens[j]
                t1 = tokens[j + 1]
                len0 = self.vocab_length[t0]
                len1 = self.vocab_length[t1]
                if len0 + len1 > self.max_token_length:
                    continue

                merged = self.vocab[t0] + self.vocab[t1]
                tid = self._find_regular_token(merged)
                if tid != -1 and self.vocab_scores[tid] > best_score:
                    best_score = self.vocab_scores[tid]
                    best_id = tid
                    best_idx = j

            if best_idx == -1:
                break

            tokens[best_idx] = best_id
            del tokens[best_idx + 1]

        return tokens


class Sampler:
    """Top-p (nucleus) and temperature sampling."""

    def __init__(
        self,
        vocab_size: int,
        temperature: float = 1.0,
        topp: float = 0.9,
        rng_seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.topp = topp
        self.rng_state = rng_seed

    def _random_u32(self) -> int:
        """xorshift* rng."""
        self.rng_state ^= self.rng_state >> 12
        self.rng_state ^= self.rng_state << 25
        self.rng_state ^= self.rng_state >> 27
        return ((self.rng_state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def _random_f32(self) -> float:
        """Random float in [0, 1)."""
        return (self._random_u32() >> 8) / 16777216.0

    def set_temp(self, temp: float):
        self.temperature = temp

    def set_seed(self, seed: int):
        self.rng_state = seed

    def sample(self, logits: List[float]) -> int:
        """Sample next token from logits."""
        if self.temperature == 0.0:
            return self._sample_argmax(logits)

        # apply temperature
        logits = [l / self.temperature for l in logits]

        # softmax
        max_val = max(logits)
        exps = [2.718281828459045 ** (l - max_val) for l in logits]
        sum_exps = sum(exps)
        if sum_exps == 0:
            sum_exps = 1e-6
        probs = [e / sum_exps for e in exps]

        coin = self._random_f32()

        if self.topp <= 0 or self.topp >= 1:
            return self._sample_mult(probs, coin)
        else:
            return self._sample_topp(probs, coin)

    def _sample_argmax(self, probs: List[float]) -> int:
        max_i = 0
        max_p = probs[0]
        for i, p in enumerate(probs):
            if p > max_p:
                max_p = p
                max_i = i
        return max_i

    def _sample_mult(self, probs: List[float], coin: float) -> int:
        cdf = 0.0
        for i, p in enumerate(probs):
            cdf += p
            if coin < cdf:
                return i
        return len(probs) - 1

    def _sample_topp(self, probs: List[float], coin: float) -> int:
        n = len(probs)
        cutoff = (1.0 - self.topp) / (n - 1)
        candidates = [
            (i, probs[i]) for i in range(n) if probs[i] >= cutoff
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        cumulative = 0.0
        last_idx = len(candidates) - 1
        for i, (_, p) in enumerate(candidates):
            cumulative += p
            if cumulative > self.topp:
                last_idx = i
                break

        r = coin * cumulative
        cdf = 0.0
        for i in range(last_idx + 1):
            cdf += candidates[i][1]
            if r < cdf:
                return candidates[i][0]
        return candidates[last_idx][0]


class TokenizerChatStops:
    """Extract stop strings from tokenizer EOS tokens."""

    def __init__(self, tokenizer: Tokenizer):
        self.stops: List[str] = [
            tokenizer.vocab[eid] for eid in tokenizer.eos_token_ids
        ]
        self.n_stops = len(self.stops)
        self.max_stop_length = max((len(s) for s in self.stops), default=0)


class ChatTemplateType:
    UNKNOWN = 0
    LLAMA2 = 1
    LLAMA3 = 2
    DEEPSEEK3 = 3
    CHATML = 4


class ChatItem:
    def __init__(self, role: str, message: str):
        self.role = role
        self.message = message


class ChatTemplateGenerator:
    """Generate formatted chat prompts from conversation history."""

    def __init__(
        self,
        template_type: int,
        chat_template: Optional[str],
        eos: str,
    ):
        self.eos = eos

        if template_type == ChatTemplateType.UNKNOWN:
            if chat_template is None:
                raise ValueError("Tokenizer does not include chat template")
            if "[INST]" in chat_template:
                self.type = ChatTemplateType.LLAMA2
            elif "<|start_header_id|>" in chat_template:
                self.type = ChatTemplateType.LLAMA3
            elif "〈Assistant〉" in chat_template or "<│Assistant│>" in chat_template:
                self.type = ChatTemplateType.DEEPSEEK3
            elif "<|im_start|>" in chat_template:
                self.type = ChatTemplateType.CHATML
            else:
                raise ValueError("Not supported chat template")
        else:
            self.type = template_type

        type_names = {1: "llama2", 2: "llama3", 3: "deepSeek3", 4: "chatml"}
        print(f"  Chat template: {type_names.get(self.type, 'unknown')}")

    def generate(
        self,
        items: List[ChatItem],
        append_generation_prompt: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """Generate formatted chat text. Returns (content, public_prompt)."""
        parts = []
        public_prompt = None

        if self.type == ChatTemplateType.LLAMA2:
            i = 0
            if (
                len(items) >= 2
                and items[0].role == "system"
                and items[1].role == "user"
            ):
                parts.append(
                    f"[INST] <<SYS>>\n{items[0].message}\n<</SYS>>\n\n"
                    f"{items[1].message} [/INST]{self.eos}"
                )
                i = 2
            for j in range(i, len(items)):
                item = items[j]
                if item.role == "assistant":
                    parts.append(f"{item.message}{self.eos}")
                elif item.role == "user":
                    parts.append(f"[INST] {item.message} [/INST]{self.eos}")
                elif item.role == "tool":
                    parts.append(
                        f"[INST] Tool output:\n{item.message} [/INST]{self.eos}"
                    )

        elif self.type == ChatTemplateType.LLAMA3:
            for item in items:
                parts.append(
                    f"<|start_header_id|>{item.role}<|end_header_id|>\n\n"
                    f"{item.message}{self.eos}"
                )
            if append_generation_prompt:
                parts.append(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

        elif self.type == ChatTemplateType.DEEPSEEK3:
            i = 0
            if items and items[0].role == "system":
                parts.append(items[0].message)
                i = 1
            for j in range(i, len(items)):
                item = items[j]
                if item.role == "user":
                    parts.append(f"<｜User｜>{item.message}")
                elif item.role == "assistant":
                    parts.append(f"<｜Assistant｜>{item.message}")
            if append_generation_prompt:
                parts.append("<｜Assistant｜><think>\n")
                public_prompt = "<think>\n"

        elif self.type == ChatTemplateType.CHATML:
            for item in items:
                if item.role == "system":
                    parts.append(
                        f"<|im_start|>system\n{item.message}<|im_end|>\n"
                    )
                elif item.role == "user":
                    parts.append(
                        f"<|im_start|>user\n{item.message}<|im_end|>\n"
                    )
                elif item.role == "assistant":
                    parts.append(
                        f"<|im_start|>assistant\n{item.message}<|im_end|>\n"
                    )
                elif item.role == "tool":
                    parts.append(
                        f"<|im_start|>tool\n{item.message}<|im_end|>\n"
                    )
            if append_generation_prompt:
                parts.append("<|im_start|>assistant\n")

        content = "".join(parts)
        return content, public_prompt


class EosDetector:
    """Streaming EOS token detection with padding tolerance."""

    def __init__(
        self,
        n_tokens: int,
        tokens: List[int],
        pieces: List[str],
        padding_left: int,
        padding_right: int,
    ):
        self.n_tokens = n_tokens
        self.tokens = tokens
        self.pieces = pieces
        self.piece_sizes = [len(p) for p in pieces]
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.buffer: bytearray = bytearray()
        self.buf_pos = 0
        self.eos_pos = -1

        for p in pieces:
            print(f"  Stop: {p}")

    def is_eos(self, token_id: int) -> bool:
        return token_id in self.tokens

    def append(self, token_id: int, piece: Optional[str]) -> int:
        """Append decoded piece and check for EOS.
        Returns: NOT_EOS=2, MAYBE_EOS=0, EOS=1
        """
        NOT_EOS = 2
        MAYBE_EOS = 0
        EOS = 1

        if piece is not None:
            piece_bytes = piece.encode("latin-1")
            new_size = self.buf_pos + len(piece_bytes) + 1
            if len(self.buffer) < new_size:
                self.buffer.extend(b"\x00" * (new_size - len(self.buffer)))
            self.buffer[self.buf_pos : self.buf_pos + len(piece_bytes)] = piece_bytes
            self.buf_pos += len(piece_bytes)
            self.buffer[self.buf_pos] = 0

        if self.is_eos(token_id):
            self.eos_pos = self.buf_pos
            return EOS

        self.eos_pos = -1

        for s, piece_str in enumerate(self.pieces):
            piece_size = self.piece_sizes[s]
            if self.buf_pos > piece_size + self.padding_left + self.padding_right:
                continue

            for lo in range(self.padding_left + 1):
                n = self.buf_pos - lo
                if n == 0 or n > piece_size + self.padding_right:
                    continue
                if n > piece_size:
                    n = piece_size
                piece_bytes = piece_str.encode("utf-8")
                if bytes(self.buffer[lo : lo + n]) == piece_bytes[:n]:
                    if n == piece_size:
                        self.eos_pos = lo
                        self.buffer[self.eos_pos] = 0
                        return EOS
                    return MAYBE_EOS
        return NOT_EOS

    def get_delta(self) -> Optional[str]:
        if self.buf_pos == 0:
            return None
        if self.eos_pos == -1:
            return bytes(self.buffer[: self.buf_pos]).decode(
                "utf-8", errors="replace"
            )
        if self.eos_pos == 0:
            return None
        return bytes(self.buffer[: self.eos_pos]).decode(
            "utf-8", errors="replace"
        )

    def reset(self):
        self.buf_pos = 0
