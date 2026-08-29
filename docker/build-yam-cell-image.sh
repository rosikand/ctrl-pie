#!/bin/sh
set -eu

checkout_path="${CTRL_PI_I2RT_CHECKOUT:-}"
dependency_commit="${CTRL_PI_I2RT_DEPENDENCY_COMMIT:-}"

if [ -z "${checkout_path}" ] || [ ! -d "${checkout_path}" ]; then
  echo "ctrl-pi: CTRL_PI_I2RT_CHECKOUT must name the local i2rt checkout" >&2
  exit 2
fi

if ! printf '%s\n' "${dependency_commit}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "ctrl-pi: CTRL_PI_I2RT_DEPENDENCY_COMMIT must be exact lowercase 40-hex" >&2
  exit 2
fi

checkout_root="$(cd "${checkout_path}" && pwd -P)"
git_root="$(git -C "${checkout_root}" rev-parse --show-toplevel)"
head_commit="$(git -C "${checkout_root}" rev-parse --verify 'HEAD^{commit}')"

if [ "${git_root}" != "${checkout_root}" ]; then
  echo "ctrl-pi: CTRL_PI_I2RT_CHECKOUT must be the checkout root" >&2
  exit 2
fi

if [ "${head_commit}" != "${dependency_commit}" ]; then
  echo "ctrl-pi: local i2rt HEAD does not match CTRL_PI_I2RT_DEPENDENCY_COMMIT" >&2
  exit 2
fi

checkout_status="$({
  git -C "${checkout_root}" status \
    --porcelain=v1 --ignored --untracked-files=all
})"
if [ -n "${checkout_status}" ]; then
  echo "ctrl-pi: local i2rt checkout must have no modified, staged, untracked, or ignored files" >&2
  exit 2
fi

docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml build app

image_commit="$(
  docker image inspect ctrl-pi:local \
    --format '{{ index .Config.Labels "org.ctrl-pi.i2rt-dependency-commit" }}'
)"
if [ "${image_commit}" != "${dependency_commit}" ]; then
  echo "ctrl-pi: built image does not retain the requested i2rt dependency identity" >&2
  exit 2
fi

echo "ctrl-pi: yam-cell image verified at i2rt dependency commit ${image_commit}"
