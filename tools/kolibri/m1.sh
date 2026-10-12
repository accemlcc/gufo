#!/usr/bin/env bash
# Build and run the Kolibri 1 checks on the box that has the GPU.
#
# The engine is written on the smaller machine and compiled on the big one, so
# this ships the sources, builds inside the ROCm container and runs the result.
# Two details are worth knowing before editing this:
#   - /work inside the container is a bind mount of the source directory on the
#     remote host, so a synced file is immediately visible to the compiler and
#     the binary it produces also exists on the remote host.
#   - the model-backed checks read a 45 GB artifact that is not mounted into the
#     container. They run on the remote host instead, which has ROCm too.
#
#   tools/kolibri/m1.sh build kolibri1_attention_test
#   tools/kolibri/m1.sh test  kolibri1_chat_template
#   tools/kolibri/m1.sh model-test kolibri1_artifact      # needs the artifact
#   tools/kolibri/m1.sh suite                            # every cpu check
set -euo pipefail

remote="${KOLIBRI_REMOTE:-m1}"
remote_src="${KOLIBRI_REMOTE_SRC:-/home/arnomatic/gufo-kolibri-src}"
container="${KOLIBRI_CONTAINER:-gufokolibri}"
build_dir="${KOLIBRI_BUILD:-build/gpu-test}"
artifact="${KOLIBRI_ARTIFACT:-/home/arnomatic/models/Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

sync() {
  tar -cf - -C "$root" src tests tools |
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
