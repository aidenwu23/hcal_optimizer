#!/usr/bin/env bash
# Source project Spack, activate one environment, and expose build prefixes.

# ---------------- detect if sourced ----------------
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then _SOURCED=1; else _SOURCED=0; fi

# Save caller's shell option state, enable strict mode only within this script
__OLD_SET_OPTS="$(set +o)"
set -Eeuo pipefail
trap 'eval "$__OLD_SET_OPTS"; if [[ $_SOURCED -eq 1 ]]; then return 1; else exit 1; fi' ERR
if [[ $_SOURCED -eq 1 ]]; then trap 'eval "$__OLD_SET_OPTS"; trap - INT ERR; return 130' INT; fi
_die() { echo "ERROR: $*" >&2; eval "$__OLD_SET_OPTS"; if [[ $_SOURCED -eq 1 ]]; then return 1; else exit 1; fi; }
_warn() { echo "WARN: $*" >&2; }

append_path_var() {
  local var_name="$1"
  local new_value="$2"
  local current_value="${!var_name-}"
  [[ -n "$new_value" ]] || return 0
  case ":$current_value:" in
    *":$new_value:"*) ;;
    *)
      if [[ -n "$current_value" ]]; then
        export "$var_name=$new_value:$current_value"
      else
        export "$var_name=$new_value"
      fi
      ;;
  esac
}

append_package_prefix() {
  local package_name="$1"
  local package_prefix=""
  package_prefix="$(spack location -i "$package_name" 2>/dev/null || true)"
  [[ -d "$package_prefix" ]] || return 0
  append_path_var CMAKE_PREFIX_PATH "$package_prefix"
}

# ---------------- locate setup-env.sh ----------------
SPACK_SETUP_FILE="${SPACK_SETUP:-$HOME/tools/spack/share/spack/setup-env.sh}"
[[ -f "$SPACK_SETUP_FILE" ]] || _die "missing $SPACK_SETUP_FILE (set SPACK_SETUP to override)."

# Derive Spack tree (git clone root) from setup-env.sh location
SPACK_SHARE_DIR="$(cd -- "$(dirname -- "$SPACK_SETUP_FILE")" && pwd -P)"   # .../share/spack
SPACK_TREE="$(cd -- "$SPACK_SHARE_DIR/../.." && pwd -P)"                  # clone root (…/spack)

# ---------------- print safe, manual update instructions (no actions performed) ----------------
if git -C "$SPACK_TREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "================== Manual update (safe) =================="
  echo "  cd \"$SPACK_TREE\""
  echo "  git remote -v"
  echo "  git status -sb"
  echo "  git fetch origin --prune"
  echo "  (only if you're not in the develop branch) git switch develop || git checkout develop"
  echo "  git pull --ff-only"
  echo "  . share/spack/setup-env.sh"
  echo "  spack reindex"
  echo "=========================================================="
  echo
else
  _warn "Spack directory is not a git clone; update instructions omitted."
  echo "=========================================================="
  echo
fi

# ---------------- source Spack in *this* shell if script is sourced ----------------
# When executed (./...), environment won’t persist after the process exits anyway.
# Sourcing here (no subshell) ensures persistence when run via: source tools/activate_spack.sh
# shellcheck source=/dev/null
source "$SPACK_SETUP_FILE"
command -v spack >/dev/null 2>&1 || _die "Spack not found after sourcing: $SPACK_SETUP_FILE"

# ---------------- show environments ----------------
echo "Spack: $(spack --version)"
echo
echo "Available environments:"
spack env list || true
echo
echo "Active environment (if any):"
spack env status || true

# ---------------- activate one environment ----------------
SPACK_ENVIRONMENT_NAME="${1:-${SPACK_ENVIRONMENT_NAME:-det-env}}"
ACTIVATION_MODE="view"
if spack env activate "$SPACK_ENVIRONMENT_NAME"; then
  :
else
  _warn "view activation failed for $SPACK_ENVIRONMENT_NAME"
  spack env activate -V "$SPACK_ENVIRONMENT_NAME"
  ACTIVATION_MODE="without-view"
fi

# ---------------- expose package prefixes for CMake ----------------
append_package_prefix dd4hep
append_package_prefix root
append_package_prefix edm4hep
append_package_prefix podio
append_package_prefix geant4

if [[ -z "${DD4hep_DIR-}" ]]; then
  DD4HEP_PREFIX="$(spack location -i dd4hep 2>/dev/null || true)"
  if [[ -n "$DD4HEP_PREFIX" ]]; then
    if [[ -d "$DD4HEP_PREFIX/lib/cmake/DD4hep" ]]; then
      export DD4hep_DIR="$DD4HEP_PREFIX/lib/cmake/DD4hep"
    elif [[ -d "$DD4HEP_PREFIX/lib64/cmake/DD4hep" ]]; then
      export DD4hep_DIR="$DD4HEP_PREFIX/lib64/cmake/DD4hep"
    fi
  fi
fi

echo
echo "Activated environment: $SPACK_ENVIRONMENT_NAME ($ACTIVATION_MODE)"
spack env status || true
echo "DD4hep_DIR=${DD4hep_DIR-}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH-}"

# ---------------- restore caller's shell options and clear traps ----------------
eval "$__OLD_SET_OPTS"
trap - ERR INT
# Note: restoring shell *options* does not undo the environment changes from setup-env.sh
