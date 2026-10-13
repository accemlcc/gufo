#!/usr/bin/env bash
# Build and run the Kolibri 1 checks on the box that has the GPU.
#
# The engine is written on the smaller machine and compiled on the big one, so
# this ships the sources, builds inside the ROCm container and runs the result.
# Two details are worth knowing before editing this:
#   - /work inside the container is a bind mount of the source directory on the
#     remote host, so a synced file is immediately visible to the compiler and
#     the binary it produces also exists on the remote host.
#   - a fresh build directory needs HIP named explicitly, because the toolchain
#     lives in a bind mount CMake does not search on its own:
#       cmake -B build/gpu-test -G Ninja -DCMAKE_BUILD_TYPE=Release \
#             -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=ON \
#             -DROCM_PATH=/opt/rocm \
#             -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
#     `-DROCM_PATH` is not decoration: the container gets /opt/rocm as a bind
#     mount (`rocm-sdk`), and CMake's HIP detection aborts with "Failed to find
#     ROCm root directory" on a directory it can see, because it looks for the
#     ROCm install registry rather than the compiler. A build directory
#     configured before that mount existed carries the answer in its cache and
#     never asks again, which is why the failure appears only on a clean one.
#     BUILD_TESTING also decides whether the test targets exist at all, so a
#     directory configured without it knows no kolibri1_*_test to build.
#   - KOLIBRI_REMOTE_SRC and KOLIBRI_CONTAINER pick another source tree and
#     container, which is how parallel work on the one GPU stays apart.
#   - the model-backed checks read a 45 GB artifact that is not mounted into the
#     container. They run on the remote host instead, which has ROCm too.
#
#   tools/kolibri/m1.sh build kolibri1_attention_test
#   tools/kolibri/m1.sh test  kolibri1_chat_template
#   tools/kolibri/m1.sh model-test kolibri1_artifact      # needs the artifact
#   tools/kolibri/m1.sh gpu-test kolibri1_attention    # needs the matrix cores
#   tools/kolibri/m1.sh suite                            # every cpu check
#   tools/kolibri/m1.sh throughput                       # link the measure driver
#   tools/kolibri/m1.sh bench 4096 32 128 131072         # needs the artifact
# The bench run is the one number the quiet GPU is owed: with another model on
# the device the same command still checks the forwards but reads its timings
# against a shared memory controller. KOLIBRI_STAGE_PROFILE=1 turns on the
# executor's own stage attribution, which prints one line per forward.
set -euo pipefail

remote="${KOLIBRI_REMOTE:-m1}"
remote_src="${KOLIBRI_REMOTE_SRC:-/home/arnomatic/gufo-kolibri-src}"
container="${KOLIBRI_CONTAINER:-gufokolibri}"
# The build container has the toolchain but no render node; the checks that need
# the matrix cores run in the container that has /dev/kfd.
gpu_container="${KOLIBRI_GPU_CONTAINER:-gufokolibri-run}"
build_dir="${KOLIBRI_BUILD:-build/gpu-test}"
artifact="${KOLIBRI_ARTIFACT:-/home/arnomatic/models/Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

# The root CMakeLists.txt travels with the sources on purpose: it names the
# targets, so a tree that ships only src/ leaves the container's own copy
# behind — a `gufo` link there then misses a model library the local file
# already lists, and the failure points at the link, not at the missing file.
sync() {
  tar -cf - -C "$root" CMakeLists.txt src tests tools |
    ssh "$remote" "cat > /tmp/kolibri-sync.tar"
  ssh "$remote" "podman cp /tmp/kolibri-sync.tar $container:/tmp/ &&
                 podman exec $container tar -xf /tmp/kolibri-sync.tar -C /work"
}

# Everything inside the container needs the ROCm runtime libraries on the path.
in_container() {
  ssh "$remote" "podman exec $container bash -lc 'cd /work && $1'"
}

command="${1:-}"
[[ -n "$command" ]] || { sed -n '2,20p' "$0"; exit 2; }
shift

case "$command" in
  sync)
    sync
    echo "synced to $remote:$remote_src"
    ;;
  build)
    sync
    in_container "cmake --build $build_dir --target $* -j8 2>&1 | tail -20"
    ;;
  test)
    sync
    in_container "LD_LIBRARY_PATH=/opt/rocm/lib $build_dir/tests/models/kolibri1/$1_test"
    ;;
  model-test)
    # Runs on the remote host, which sees the artifact but not the container's
    # mount, so the binary path is spelled from the source directory.
    sync
    in_container "cmake --build $build_dir --target ${1}_test -j8 2>&1 | tail -5"
    ssh "$remote" "LD_LIBRARY_PATH=/opt/rocm/lib GUFO_KOLIBRI1_MODEL=$artifact \
                   $remote_src/$build_dir/tests/models/kolibri1/${1}_test"
    ;;
  gpu-test)
    # Build where the toolchain is, run where the GPU is: the binary goes from
    # the build container's tree to a shared path and then into the other one.
    sync
    in_container "cmake --build $build_dir --target ${1}_test -j8 2>&1 | tail -8"
    # Named after the check so two people on the one GPU do not overwrite each
    # other's binary on the way across.
    #
    # KOLIBRI_TIME reaches the check only when it is really set: the checks ask
    # getenv() != nullptr rather than testing a value, so passing
    # "-e KOLIBRI_TIME=0" switches every timing block on.
    time_env=""
    [[ -z "${KOLIBRI_TIME:-}" ]] || time_env="-e KOLIBRI_TIME=$KOLIBRI_TIME"
    ssh "$remote" "cp $remote_src/$build_dir/tests/models/kolibri1/${1}_test /tmp/${1}.bin && \
                   podman cp /tmp/${1}.bin $gpu_container:/tmp/${1} && \
                   podman exec $time_env $gpu_container bash -lc 'LD_LIBRARY_PATH=/opt/rocm/lib /tmp/${1}'"
    ;;
  throughput)
    # The driver links the engine's archives, so it is built where they are and
    # run where the artifact is.
    sync
    in_container "cmake --build $build_dir --target gufo_kolibri1 -j8 2>&1 | tail -3 && \
                  ROCM_PATH=/opt/rocm bash tools/kolibri/build_throughput.sh $build_dir 2>&1 | tail -2"
    ;;
  bench)
    sync
    in_container "cmake --build $build_dir --target gufo_kolibri1 -j8 2>&1 | tail -3 && \
                  ROCM_PATH=/opt/rocm bash tools/kolibri/build_throughput.sh $build_dir 2>&1 | tail -1"
    ssh "$remote" "LD_LIBRARY_PATH=/opt/rocm/lib GUFO_KOLIBRI1_MODEL=$artifact \
                   ${KOLIBRI_STAGE_PROFILE:+GUFO_KOLIBRI_ROCM_PREFILL_STAGE_PROFILE=1} \
                   $remote_src/artifacts/kolibri/throughput $artifact $*"
    ;;
  suite)
    sync
    in_container "LD_LIBRARY_PATH=/opt/rocm/lib ctest --test-dir $build_dir \
                  -L kolibri1 --exclude-regex slow 2>&1 | tail -8"
    ;;
  *)
    echo "unknown command $command" >&2
    exit 2
    ;;
esac
