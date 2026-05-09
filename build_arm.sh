#!/bin/bash
# Build ARM .so for Raspberry Pi
#   ./build_arm.sh cross    - cross-compile (needs aarch64-linux-gnu-gcc)
#   ./build_arm.sh native   - build directly on Pi
#   ./build_arm.sh cross dotprod  - cross-compile with dotprod for Pi 5

set -e
MODE=${1:-cross}
FEAT=${2:-neon}  # neon | dotprod

CC=""
CFLAGS=""
TARGET="distributed_llama/ops/_ops.arm.so"

if [ "$MODE" = "cross" ]; then
    CC=aarch64-linux-gnu-gcc
    PREFIX=aarch64-linux-gnu-
    PY_INCLUDE=/usr/include/python3.10
    if [ "$FEAT" = "dotprod" ]; then
        CFLAGS="-O3 -fPIC -march=armv8.2-a+simd+dotprod -mtune=cortex-a76"
    else
        CFLAGS="-O3 -fPIC -march=armv8-a+simd -mtune=cortex-a72"
    fi
elif [ "$MODE" = "native" ]; then
    CC=gcc
    PY_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    CFLAGS="-O3 -fPIC -march=native"
else
    echo "Usage: $0 [cross|native] [neon|dotprod]"
    exit 1
fi

echo "Building ARM .so (mode=$MODE, features=$FEAT)"
echo "  CC=$CC"
echo "  CFLAGS=$CFLAGS"

$CC $CFLAGS -I"$PY_INCLUDE" \
    -c distributed_llama/ops/ops.c -o /tmp/ops_arm.o

$CC -shared /tmp/ops_arm.o -o "$TARGET" -lpthread -lm

echo "Done: $TARGET ($(du -h $TARGET | cut -f1))"
echo ""
echo "Deploy to Pi:"
echo "  scp $TARGET pi@<host>:~/distributed-llama-python/distributed_llama/ops/"
echo "  # Then rename on Pi to match Python version:"
echo "  # mv _ops.arm.so _ops.cpython-*-aarch64-linux-gnu.so"
