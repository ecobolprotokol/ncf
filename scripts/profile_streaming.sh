#!/usr/bin/env bash
set -euo pipefail

echo "Building benches (release)..."
cargo build --release --bench ncf_performance

BIN=$(ls target/release/deps/ncf_performance-* | head -n1)
if [ -z "$BIN" ]; then
  echo "Bench binary not found"
  exit 1
fi

echo "Bench binary: $BIN"

echo "Make sure 'perf' and 'flamegraph' are installed. Example (Ubuntu): sudo apt install linux-tools-$(uname -r) && cargo install flamegraph"

echo "Recording perf data (will require root on most systems)..."
# Use sudo if not root
PERF_CMD=(perf record -F 99 -g -- "$BIN" ncf_streaming_chunk_verify --warm-up-time 1 --measurement-time 5)
if [ "$EUID" -ne 0 ]; then
  PERF_CMD=(sudo "${PERF_CMD[@]}")
fi
"${PERF_CMD[@]}"

echo "Generating flamegraph.svg (requires 'flamegraph' script from the flamegraph crate)"
if command -v flamegraph >/dev/null 2>&1; then
  perf script | flamegraph > streaming_flame.svg
  echo "Wrote streaming_flame.svg"
else
  echo "flamegraph tool not found. Install with: cargo install flamegraph"
  exit 1
fi
