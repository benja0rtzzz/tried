#!/usr/bin/env bash
# Collect every value the `verification:` block of docs/specs.yaml expects.
#
# Run from the repo root on the Lenovo (after `uv sync`):
#   bash scripts/collect_lenovo_specs.sh
#
# The script prints a YAML-shaped block to stdout, followed by a
# diagnostic log explaining any field that came back empty. Copy the
# YAML block into docs/specs.yaml under `verification:`; use the log to
# fix anything that didn't populate.
#
# WSL2-aware: looks for nvidia-smi at /usr/lib/wsl/lib/nvidia-smi and
# falls back to the Windows-side nvidia-smi.exe via WSL interop.
#
# No side effects: read-only commands only.

set -u

# --- diagnostic log scaffolding -------------------------------------------

declare -a DIAG=()
note() { DIAG+=("$1"); }

# Detect environment context once (used in several fallbacks below)
IS_WSL=0
WSL_DISTRO=""
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  IS_WSL=1
  WSL_DISTRO="${WSL_DISTRO_NAME:-unknown}"
  note "environment detected: WSL2 (distro=${WSL_DISTRO}); some host-level values are not directly visible"
fi

# --- emit helpers ----------------------------------------------------------

emit() {
  local key="$1"; shift
  local value="$*"
  if [[ -z "${value// }" ]]; then
    printf '    %s:        ""\n' "$key"
  else
    printf '    %s:        "%s"\n' "$key" "$value"
  fi
}

# Resolve nvidia-smi, including WSL2 paths.
# Prints the resolved command (one or two tokens) on stdout; returns 1 if none.
resolve_nvidia_smi() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi'
    return 0
  fi
  if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    printf '/usr/lib/wsl/lib/nvidia-smi'
    return 0
  fi
  if (( IS_WSL )) && command -v nvidia-smi.exe >/dev/null 2>&1; then
    printf 'nvidia-smi.exe'
    return 0
  fi
  return 1
}

# --- Hardware --------------------------------------------------------------

machine=""
if [[ -r /sys/class/dmi/id/product_name ]]; then
  machine="$(cat /sys/class/dmi/id/product_name 2>/dev/null | tr -d '\n')"
fi
if [[ -z "$machine" ]] && (( IS_WSL )); then
  # Try Windows-side via PowerShell, returns vendor + model
  if command -v powershell.exe >/dev/null 2>&1; then
    machine="$(powershell.exe -NoProfile -Command \
      "(Get-CimInstance Win32_ComputerSystem | ForEach-Object { \$_.Manufacturer + ' ' + \$_.Model })" \
      2>/dev/null | tr -d '\r\n' | sed 's/  */ /g')"
  fi
  [[ -z "$machine" ]] && note "machine: WSL2 hides host DMI; install powershell.exe interop or set this manually (e.g. \"Lenovo LOQ 15IRX9\")"
elif [[ -z "$machine" ]]; then
  note "machine: /sys/class/dmi/id/product_name unreadable; set this manually"
fi

cpu="$(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)"
[[ -z "$cpu" ]] && note "cpu: /proc/cpuinfo did not contain a 'model name' line"

memory_bytes_raw="$(awk '/MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null)"
memory_bytes="$memory_bytes_raw"
memory_note=""
if (( IS_WSL )) && [[ -n "$memory_bytes_raw" ]]; then
  # /proc/meminfo inside WSL2 reports the VM allocation, not host RAM.
  if command -v powershell.exe >/dev/null 2>&1; then
    host_bytes="$(powershell.exe -NoProfile -Command \
      "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory" 2>/dev/null | tr -d '\r\n')"
    if [[ "$host_bytes" =~ ^[0-9]+$ ]]; then
      memory_bytes="$host_bytes"
      memory_note=" (host-reported via WSL interop; /proc/meminfo would have shown the WSL VM cap of ${memory_bytes_raw})"
      note "memory_bytes: replaced WSL VM allocation (${memory_bytes_raw}) with host total (${host_bytes}) via PowerShell interop"
    else
      note "memory_bytes: /proc/meminfo reports the WSL VM cap (${memory_bytes_raw}); fix .wslconfig or set this manually to the laptop's installed RAM"
    fi
  else
    note "memory_bytes: /proc/meminfo reports the WSL VM cap (${memory_bytes_raw}); install powershell.exe interop or set this manually"
  fi
fi
[[ -z "$memory_bytes" ]] && note "memory_bytes: /proc/meminfo did not contain MemTotal"

gpu=""; gpu_memory_mb=""; sm_arch=""; nvidia_driver=""; cuda_driver=""
if smi_cmd="$(resolve_nvidia_smi)"; then
  note "nvidia-smi resolved to: ${smi_cmd}"
  gpu="$($smi_cmd --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
  gpu_memory_mb="$($smi_cmd --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '\r')"
  sm_arch="$($smi_cmd --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
  nvidia_driver="$($smi_cmd --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
  cuda_driver="$($smi_cmd 2>/dev/null | tr -d '\r' | awk -F'CUDA Version: ' '/CUDA Version/ {print $2; exit}' | awk '{print $1}')"
  [[ -z "$gpu" ]]            && note "gpu: nvidia-smi returned empty name (no GPU visible to this shell)"
  [[ -z "$gpu_memory_mb" ]]  && note "gpu_memory_mb: nvidia-smi returned no memory.total"
  [[ -z "$sm_arch" ]]        && note "sm_arch: nvidia-smi returned no compute_cap (very old driver?)"
  [[ -z "$nvidia_driver" ]]  && note "nvidia_driver: nvidia-smi returned no driver_version"
  [[ -z "$cuda_driver" ]]    && note "cuda_driver: 'CUDA Version' not in nvidia-smi banner output"
else
  note "nvidia-smi: not found on PATH, /usr/lib/wsl/lib/, or via .exe interop — GPU/driver fields will be empty. On WSL2, install the NVIDIA WSL driver on the Windows host (https://docs.nvidia.com/cuda/wsl-user-guide/)"
fi

# --- OS --------------------------------------------------------------------

os_name=""; os_version=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  os_name="${NAME:-}"
  os_version="${VERSION:-${VERSION_ID:-}}"
fi
[[ -z "$os_name" ]] && note "os.name: /etc/os-release missing or empty NAME"

kernel="$(uname -r 2>/dev/null) $(uname -m 2>/dev/null)"

# --- Toolchain -------------------------------------------------------------

python_version=""
if command -v python3 >/dev/null 2>&1; then
  python_version="$(python3 --version 2>&1 | awk '{print $2}')"
else
  note "python: python3 not on PATH (run 'sudo apt install python3' or load a venv)"
fi

uv_version=""
if command -v uv >/dev/null 2>&1; then
  uv_version="$(uv --version 2>&1 | head -1)"
else
  note "uv: not on PATH (https://docs.astral.sh/uv/)"
fi

gcc_version=""
if command -v gcc >/dev/null 2>&1; then
  gcc_version="$(gcc --version 2>/dev/null | head -1)"
else
  note "gcc: not on PATH; Inductor compiles C at runtime and will fail without it (sudo apt install gcc g++ python3.12-dev)"
fi

cuda_runtime=""
if command -v nvcc >/dev/null 2>&1; then
  cuda_runtime="$(nvcc --version 2>/dev/null | awk '/release/ {sub(/,$/,"",$6); print $6}')"
else
  note "cuda_runtime: nvcc not on PATH. Optional — PyTorch wheels bundle their own CUDA libs. Install nvidia-cuda-toolkit only if you need to compile custom CUDA code."
fi

# --- Python packages (requires uv sync to have run) ------------------------

torch_v=""; triton_v=""; fastapi_v=""; uvicorn_v=""; pkg_err=""
if command -v uv >/dev/null 2>&1; then
  pkg_out="$(uv run --quiet python - <<'PY' 2>&1
import sys

def v(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", "") or ""
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"

print(v("torch"))
print(v("triton"))
print(v("fastapi"))
print(v("uvicorn"))
PY
  )"
  torch_v="$(printf '%s\n' "$pkg_out" | sed -n '1p')"
  triton_v="$(printf '%s\n' "$pkg_out" | sed -n '2p')"
  fastapi_v="$(printf '%s\n' "$pkg_out" | sed -n '3p')"
  uvicorn_v="$(printf '%s\n' "$pkg_out" | sed -n '4p')"
  for pair in "torch:$torch_v" "triton:$triton_v" "fastapi:$fastapi_v" "uvicorn:$uvicorn_v"; do
    name="${pair%%:*}"; val="${pair#*:}"
    if [[ "$val" == ERR:* ]]; then
      note "python_packages.${name}: ${val} — run 'uv sync' from the repo root, then rerun this script"
      # Blank the value so the YAML stays clean
      case "$name" in
        torch) torch_v="" ;;
        triton) triton_v="" ;;
        fastapi) fastapi_v="" ;;
        uvicorn) uvicorn_v="" ;;
      esac
    elif [[ -z "$val" ]]; then
      note "python_packages.${name}: import succeeded but __version__ was empty"
    fi
  done
else
  note "python_packages: uv not on PATH — cannot import torch/triton/fastapi/uvicorn"
fi

# --- Sanity checks (cross-field) -------------------------------------------

if [[ -n "$torch_v" && "$torch_v" =~ \+cu([0-9]+) ]]; then
  torch_cu="${BASH_REMATCH[1]}"
  if [[ -n "$cuda_driver" ]]; then
    # cuda_driver is a "12.x" / "13.x" string; the wheel suffix is "120"/"130".
    # Just compare the major.
    drv_major="${cuda_driver%%.*}"
    wheel_major="${torch_cu:0:2}"
    if [[ "$drv_major" != "$wheel_major" ]]; then
      note "version-skew warning: torch wheel is +cu${torch_cu} but driver reports CUDA ${cuda_driver}; usually fine forward but worth recording"
    fi
  fi
fi

# --- Emit YAML -------------------------------------------------------------

cat <<YAML
# --- paste under \`verification:\` in docs/specs.yaml -----------------------
verification:
  hardware:
$(emit machine        "$machine")
$(emit cpu            "$cpu")
    memory_bytes:  ${memory_bytes:-~}${memory_note}
$(emit gpu            "$gpu")
    gpu_memory_mb: ${gpu_memory_mb:-~}
$(emit sm_arch        "$sm_arch")
  os:
$(emit name           "$os_name")
$(emit version        "$os_version")
$(emit kernel         "$kernel")
  toolchain:
$(emit python         "$python_version")
$(emit uv             "$uv_version")
$(emit gcc            "$gcc_version")
$(emit cuda_runtime   "$cuda_runtime")
$(emit cuda_driver    "$cuda_driver")
$(emit nvidia_driver  "$nvidia_driver")
  python_packages:
$(emit torch          "$torch_v")
$(emit triton         "$triton_v")
$(emit fastapi        "$fastapi_v")
$(emit uvicorn        "$uvicorn_v")
  uv_lock_path:     uv.lock
# --- end paste -----------------------------------------------------------

# --- diagnostic log ------------------------------------------------------
# This section is for you, not for specs.yaml. Empty fields above are
# explained here; non-empty fields are silently fine.
YAML

if [[ ${#DIAG[@]} -eq 0 ]]; then
  echo "# (no issues — every field populated cleanly)"
else
  for line in "${DIAG[@]}"; do
    printf '# - %s\n' "$line"
  done
fi
echo "# --- end log ------------------------------------------------------"
