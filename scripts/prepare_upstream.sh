#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pin=4762ad7316dcdec20016ab5985fb46a27902204d
pristine=third_party/llama.cpp
patched=third_party/llama-round2
mkdir -p third_party
if [[ ! -e "$pristine" ]]; then
  git clone --depth 1 --branch b6000 https://github.com/ggml-org/llama.cpp.git "$pristine"
fi
[[ "$(git -C "$pristine" rev-parse HEAD)" == "$pin" ]] || {
  echo 'Expected the pinned llama.cpp b6000 commit.' >&2; exit 1;
}
[[ -z "$(git -C "$pristine" status --porcelain --untracked-files=no)" ]] || {
  echo 'The pristine upstream checkout has tracked modifications.' >&2; exit 1;
}
if [[ ! -e "$patched" ]]; then
  git clone --no-hardlinks "$pristine" "$patched"
fi
python3 scripts/prepare_full_model_source.py --source "$patched"
