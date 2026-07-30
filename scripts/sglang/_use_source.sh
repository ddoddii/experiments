# shellcheck shell=bash
# Make every launch use the MODIFIED sglang source tree, not the copy pip installed
# into the conda env. Source this after `conda activate sglang`.
#
# Why this exists: `conda activate sglang` puts an installed sglang on sys.path. If the
# working tree is a separate directory (~/sglang-source), `import sglang` silently
# resolves to the INSTALLED copy -- so idle-KV-parking edits appear to do nothing and
# an experiment measures unmodified upstream behaviour. This guard makes that failure
# loud instead of silent.
#
# Preferred fix (do this once, then the guard just confirms it):
#     conda activate sglang
#     cd ~/sglang-source/python && pip install -e . --no-deps --no-build-isolation
#   --no-deps matters: without it pip re-resolves torch/flashinfer/etc and can break a
#   working env.
#
# This script also prepends the source tree to PYTHONPATH, so it works even without the
# editable install.
#
# Override the location with SGLANG_SRC_DIR=/path/to/sglang-source.

SGLANG_SRC_DIR=${SGLANG_SRC_DIR:-"$HOME/sglang-source"}

if [ -d "$SGLANG_SRC_DIR/python/sglang" ]; then
  export PYTHONPATH="$SGLANG_SRC_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "[env] WARNING: no sglang package at $SGLANG_SRC_DIR/python/sglang"
  echo "[env]          set SGLANG_SRC_DIR to your working tree"
fi

# NOTE: this file calls `exit` on failure. It is meant to be sourced by the start
# scripts (which run under `set -e`); sourcing it in an interactive shell will close
# that shell.
_SG_ERR=$(mktemp)
_SG_LOADED=$(python -c 'import os,sglang;print(os.path.dirname(os.path.abspath(sglang.__file__)))' 2>"$_SG_ERR" || true)
if [ -z "$_SG_LOADED" ]; then
  echo "[env] ERROR: cannot import sglang. Is the 'sglang' conda env active?"
  echo "[env]        python: $(command -v python)"
  echo "[env]        the import failed with:"
  sed 's/^/[env]        | /' "$_SG_ERR" | tail -6
  rm -f "$_SG_ERR"
  exit 1
fi
rm -f "$_SG_ERR"

echo "[env] sglang loaded from: $_SG_LOADED"
case "$_SG_LOADED" in
  "$SGLANG_SRC_DIR"/*)
    # confirm the branch too -- running the right tree on the wrong branch is the same
    # class of silent mistake
    if command -v git >/dev/null 2>&1 && [ -d "$SGLANG_SRC_DIR/.git" ]; then
      echo "[env] branch: $(git -C "$SGLANG_SRC_DIR" rev-parse --abbrev-ref HEAD)" \
           "@ $(git -C "$SGLANG_SRC_DIR" rev-parse --short HEAD)"
      if [ -n "$(git -C "$SGLANG_SRC_DIR" status --porcelain 2>/dev/null)" ]; then
        echo "[env] note: working tree has uncommitted changes"
      fi
    fi
    ;;
  *)
    echo "[env] ERROR: sglang is being imported from the INSTALLED copy, not from"
    echo "[env]        $SGLANG_SRC_DIR. Your edits would NOT take effect."
    echo "[env]        Fix: cd $SGLANG_SRC_DIR/python && \\"
    echo "[env]             pip install -e . --no-deps --no-build-isolation"
    exit 1
    ;;
esac
