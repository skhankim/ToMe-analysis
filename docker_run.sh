#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-tome}"
CONTAINER_NAME="${CONTAINER_NAME:-ToMe}"
HOST_GPU="${HOST_GPU:-0}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/ToMe}"
SHM_SIZE="${SHM_SIZE:-16g}"

usage() {
  cat <<EOF
Usage: ./docker_run.sh [shell|up|build|stop|rm]

  shell : build if needed, create/start container if needed, then enter bash
  up    : build if needed, create/start container
  build : build image only
  stop  : stop container
  rm    : remove container

Environment overrides:
  IMAGE_NAME=${IMAGE_NAME}
  CONTAINER_NAME=${CONTAINER_NAME}
  HOST_GPU=${HOST_GPU}
  CONTAINER_WORKDIR=${CONTAINER_WORKDIR}
  SHM_SIZE=${SHM_SIZE}

Notes:
  - The repo is bind-mounted into the container.
  - Host GPU ${HOST_GPU} is exposed as container GPU 0.
  - server-53: HOST_GPU=0 ./docker_run.sh shell
  - server-55: HOST_GPU=1 ./docker_run.sh shell
EOF
}

ACTION="${1:-shell}"

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

build_image() {
  docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
}

ensure_image() {
  if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    build_image
  fi
}

ensure_container() {
  ensure_image

  if container_exists; then
    if ! container_running; then
      docker start "${CONTAINER_NAME}" >/dev/null
    fi
    return
  fi

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus "device=${HOST_GPU}" \
    --shm-size="${SHM_SIZE}" \
    -v "${SCRIPT_DIR}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    sleep infinity >/dev/null
}

case "${ACTION}" in
  shell)
    ensure_container
    exec docker exec -it "${CONTAINER_NAME}" bash
    ;;
  up)
    ensure_container
    echo "Container is ready: ${CONTAINER_NAME}"
    echo "Enter with: docker exec -it ${CONTAINER_NAME} bash"
    ;;
  build)
    build_image
    ;;
  stop)
    docker stop "${CONTAINER_NAME}"
    ;;
  rm)
    docker rm -f "${CONTAINER_NAME}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
