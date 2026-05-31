Profiling NCF benches with perf + flamegraph

Prerequisites (Ubuntu):

- Install system perf tools (may require root):

  sudo apt update
  sudo apt install linux-tools-$(uname -r) linux-tools-common linux-tools-generic

- Install flamegraph helper (Rust crate):

  cargo install flamegraph

Usage:

1. Build and record a perf profile for the streaming benchmark:

  ./scripts/profile_streaming.sh

2. The script builds the bench binary, runs `perf record`, and writes `streaming_flame.svg` in the repo root.

Notes:

- Many distributions require root (sudo) to record perf on the system; run the script on a machine where you can run `perf`.
- If you're running inside a container or CI without perf support, run profiling on the host machine or use a VM with perf available.
- If you want me to add an alternative instrumentation-based profiler (no perf), tell me and I can implement pprof-based sampling in the bench harness.
