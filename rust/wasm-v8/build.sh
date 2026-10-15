#!/usr/bin/env bash
# Build the embedded-V8 WASM bundle (Python mini-racer / R V8 backends) and,
# with --vendor, copy it into both packages.
#
#   rust/wasm-v8/build.sh            build into rust/pkg-v8/ and check it
#   rust/wasm-v8/build.sh --vendor   ... then vendor it into
#                                      src/praatfan/wasm/ and r-package/inst/wasm/
#
# The bundle is `wasm-pack build --target no-modules`: an embedded V8 has no
# ES modules and no fetch, so the browser's `--target web` glue cannot load
# there; the no-modules glue is a plain script defining a global `wasm_bindgen`.
#
# Rust panic messages carry the source path of the code that panicked, and a
# dependency's source lives in the builder's cargo home, so a plain build would
# ship the builder's home directory inside the .wasm.  Paths are remapped, and
# the build fails if one survives.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CRATE/.." && pwd)"
OUT="$CRATE/pkg-v8"
VENDOR=0
[[ "${1:-}" == "--vendor" ]] && VENDOR=1

CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
SEP=$'\x1f'
FLAGS="--remap-path-prefix=$CRATE=/src${SEP}--remap-path-prefix=$CARGO_HOME_DIR=/cargo${SEP}--remap-path-prefix=$HOME=/~"
if [[ -n "${RUSTFLAGS:-}" ]]; then
  # keep any flags already set (CARGO_ENCODED_RUSTFLAGS would override them)
  FLAGS="$(printf '%s' "$RUSTFLAGS" | tr ' ' "$SEP")$SEP$FLAGS"
fi

rm -rf "$OUT"
(cd "$CRATE" && CARGO_ENCODED_RUSTFLAGS="$FLAGS" \
  wasm-pack build --release --target no-modules --out-dir pkg-v8 --out-name praatfan \
    --no-pack --features wasm)

cp "$HERE/shim.js" "$OUT/shim.js"
cp "$ROOT/LICENSE" "$OUT/LICENSE"

# No builder paths may ship.
for f in "$OUT/praatfan_bg.wasm" "$OUT/praatfan.js"; do
  if LC_ALL=C grep -a -q -E '/home/|/Users/|C:\\Users' "$f"; then
    echo "error: $f contains a builder path:" >&2
    LC_ALL=C grep -a -o -E '(/home/|/Users/|C:\\Users)[^"[:cntrl:]]{0,60}' "$f" | sort -u | head >&2
    exit 1
  fi
done

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$CRATE/Cargo.toml" | head -1)"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
DIRTY=""
git -C "$ROOT" diff --quiet HEAD -- "$CRATE/src" "$CRATE/Cargo.toml" "$CRATE/Cargo.lock" || DIRTY="+dirty"
{
  echo "praatfan_rust $VERSION, wasm32 --target no-modules (embedded V8)"
  echo "commit: $COMMIT$DIRTY"
  echo "build: wasm-pack build --release --target no-modules --features wasm --out-name praatfan"
  echo "remap: <crate>=/src, <cargo home>=/cargo, \$HOME=/~"
  echo "toolchain: $(rustc --version)"
  echo "wasm-pack: $(wasm-pack --version)"
  echo "sha256:"
  (cd "$OUT" && sha256sum praatfan.js praatfan_bg.wasm shim.js | sed 's/^/  /')
} > "$OUT/PROVENANCE.txt"

# Smoke test: load it the way the packages do and run one request.
node "$HERE/smoke.mjs" "$OUT" "$ROOT/tests/fixtures/one_two_three_four_five.wav"

if [[ $VENDOR == 1 ]]; then
  [[ -z "$DIRTY" ]] || { echo "error: refusing to vendor a build of uncommitted Rust sources" >&2; exit 1; }
  for dest in "$ROOT/src/praatfan/wasm" "$ROOT/r-package/inst/wasm"; do
    rm -rf "$dest"
    mkdir -p "$dest"
    cp "$OUT"/{praatfan.js,praatfan_bg.wasm,shim.js,LICENSE,PROVENANCE.txt} "$dest/"
    echo "vendored -> $dest"
  done
fi
