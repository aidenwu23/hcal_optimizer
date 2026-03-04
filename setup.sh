#!/usr/bin/env bash
# Source me:  source setup.sh
# Workflow: ensure sourcing, resolve repository root, prepend bin/lib locations,
#           surface plugin directories, and opt into helpful DD4hep debug output.
# Adds this project's bin + libs (and ROOT libs if available) for THIS shell only.

# must be sourced
[[ "${BASH_SOURCE[0]}" != "$0" ]] || { echo "Please 'source setup.sh'"; exit 1; }

# Resolve the repository root so subsequent relative paths are well-defined
ROOT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

# 1) add install bin if present
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local}"
BIN_DIR="$INSTALL_PREFIX/bin"
if [[ -d "$BIN_DIR" ]]; then
  # Prepend the local install so the freshly built process binary appears first in PATH
  case ":$PATH:" in *":$BIN_DIR:"*) ;; *) export PATH="$BIN_DIR:$PATH"; echo "PATH += $BIN_DIR";; esac
fi

# 2) add build plugin dir (if a plugin exists), then install/lib
plug_dir="$(find "$ROOT_DIR/build" -type f -name 'libdetector_plugin.so*' -print -quit 2>/dev/null | xargs -r -I{} dirname "{}")"
if [[ -n "$plug_dir" ]]; then
  # The dd4hep plugin sits in the build tree - add it so Geant4 locates it at runtime
  if [[ "$OSTYPE" == "darwin"* ]]; then
	  case ":${DYLD_LIBRARY_PATH-}:" in *":$plug_dir:"*) ;; *)
		export DYLD_LIBRARY_PATH="$plug_dir${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}"
		echo "DYLD_LIBRARY_PATH += $plug_dir (plugin)"
	  esac
  else
	  case ":${LD_LIBRARY_PATH-}:" in *":$plug_dir:"*) ;; *)
		export LD_LIBRARY_PATH="$plug_dir${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"
		echo "LD_LIBRARY_PATH += $plug_dir (plugin)"
	  esac
  fi
fi
if [[ -d "$INSTALL_PREFIX/lib" ]]; then
  # Also expose the install prefix (cmake --install) for workflows that rely on it
  if [[ "$OSTYPE" == "darwin"* ]]; then
	  case ":${DYLD_LIBRARY_PATH-}:" in *":$INSTALL_PREFIX/lib:"*) ;; *)
		export DYLD_LIBRARY_PATH="$INSTALL_PREFIX/lib${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}"
		echo "DYLD_LIBRARY_PATH += $INSTALL_PREFIX/lib"
	  esac
  else
	  case ":${LD_LIBRARY_PATH-}:" in *":$INSTALL_PREFIX/lib:"*) ;; *)
		export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"
		echo "LD_LIBRARY_PATH += $INSTALL_PREFIX/lib"
	  esac
  fi
fi

# 3) add ROOT libs if root-config is already on PATH (no auto-loading)
if command -v root-config >/dev/null 2>&1; then
  ROOT_LIB="$(root-config --libdir)"
  if [[ -n "$ROOT_LIB" ]]; then
    # Keep ROOT's libs handy because many detector studies require them
    if [[ "$OSTYPE" == "darwin"* ]]; then
		case ":${DYLD_LIBRARY_PATH-}:" in *":$ROOT_LIB:"*) ;; *)
		  export DYLD_LIBRARY_PATH="$ROOT_LIB${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}"
		  echo "DYLD_LIBRARY_PATH += $ROOT_LIB (ROOT)"
		esac
    else
		case ":${LD_LIBRARY_PATH-}:" in *":$ROOT_LIB:"*) ;; *)
		  export LD_LIBRARY_PATH="$ROOT_LIB${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"
		  echo "LD_LIBRARY_PATH += $ROOT_LIB (ROOT)"
		esac
    fi
  fi
fi

# 4) handy default for DD4hep debugging
export DD4HEP_DEBUG_PLUGINS="${DD4HEP_DEBUG_PLUGINS:-1}"
