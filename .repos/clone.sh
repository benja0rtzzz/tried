#!/usr/bin/env bash
# Clone reference repos for SFT pair extraction.
#
# Each repo is shallow-cloned (--depth 1). The HEAD SHA at clone time is
# recorded into COMMITS.txt so docs/data-sources.md can cite an exact
# pinned commit for each source.
#
# Usage:
#   .repos/clone.sh            # initial clone (skips repos already present)
#   .repos/clone.sh --reset    # reset every cloned repo to its recorded SHA

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITS_FILE="$SCRIPT_DIR/COMMITS.txt"

# Tier 1 (best pair quality, broadest category coverage)
# Tier 2 (filler for under-quota categories)
REPOS=(
  # name|url|tier
  "liger-kernel|https://github.com/linkedin/Liger-Kernel.git|1"
  "flash-attention|https://github.com/Dao-AILab/flash-attention.git|1"
  "torchao|https://github.com/pytorch/ao.git|1"
  "tritonbench|https://github.com/pytorch-labs/tritonbench.git|1"
  "triton|https://github.com/triton-lang/triton.git|1"
  "attorch|https://github.com/BobMcDear/attorch.git|1"
  "flaggems|https://github.com/FlagOpen/FlagGems.git|1"
  "flash-linear-attention|https://github.com/fla-org/flash-linear-attention.git|1"
  "applied-ai|https://github.com/meta-pytorch/applied-ai.git|1"
  "attention-gym|https://github.com/meta-pytorch/attention-gym.git|1"
  "fbgemm|https://github.com/pytorch/FBGEMM.git|2"
  "mamba|https://github.com/state-spaces/mamba.git|2"
  "vllm|https://github.com/vllm-project/vllm.git|2"
  "xformers|https://github.com/facebookresearch/xformers.git|2"
  "scattermoe|https://github.com/shawntan/scattermoe.git|2"
  "flashnn|https://github.com/AlibabaPAI/FLASHNN.git|2"
)

mode="${1:-clone}"

if [[ "$mode" == "--reset" ]]; then
  if [[ ! -f "$COMMITS_FILE" ]]; then
    echo "no $COMMITS_FILE — nothing to reset to" >&2
    exit 1
  fi
  while IFS=$'\t' read -r name sha _date _tier _url; do
    dir="$SCRIPT_DIR/$name"
    if [[ -d "$dir/.git" ]]; then
      echo "==> resetting $name to $sha"
      git -C "$dir" fetch origin "$sha" --depth 1 2>/dev/null || git -C "$dir" fetch origin
      git -C "$dir" checkout "$sha"
    else
      echo "!!  $name not cloned, skip"
    fi
  done < "$COMMITS_FILE"
  exit 0
fi

# Fresh clone (idempotent — skips already-present repos but always rewrites COMMITS.txt)
: > "$COMMITS_FILE"
printf "# Pinned commits for cloned reference repos\n" >> "$COMMITS_FILE"
printf "# columns: name<TAB>sha<TAB>committer_date_iso8601<TAB>tier<TAB>url\n" >> "$COMMITS_FILE"

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url tier <<< "$entry"
  dir="$SCRIPT_DIR/$name"
  if [[ -d "$dir/.git" ]]; then
    echo "==> $name already at $dir (using existing checkout)"
  else
    echo "==> cloning $name"
    git clone --depth 1 "$url" "$dir"
  fi
  sha="$(git -C "$dir" rev-parse HEAD)"
  date="$(git -C "$dir" log -1 --format=%cI)"
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$sha" "$date" "$tier" "$url" >> "$COMMITS_FILE"
done

echo
echo "Cloned ${#REPOS[@]} repos. SHAs recorded in $COMMITS_FILE."
echo "Total disk usage:"
du -sh "$SCRIPT_DIR"/*/ 2>/dev/null | sort -h | tail
