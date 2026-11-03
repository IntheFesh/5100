#!/usr/bin/env bash
# Build the flat code package for Colab: drl_v5_code.zip (the zip root IS the code directory - the notebook unzips it to
# /content/code and runs from there). Excludes .pkl / .mp4 / third_party / .venv and everything else not listed.
# Usage: ./make_zip.sh [output.zip]      (default: ./drl_v5_code.zip)
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-drl_v5_code.zip}
FILES=(pqn_v5.py test_v5.py tests/test_stats.py sweep.py analyze.py render_demo.py check_protocol.py mechanism.py README.md CHANGELOG.md DRL_v5_colab.ipynb)
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
rm -f "$OUT"
zip -q -X "$OUT" "${FILES[@]}"
# sanity: flat layout (no leading folder), no forbidden artefacts
if unzip -Z1 "$OUT" | grep -E '\.(pkl|mp4)$|third_party|\.venv|^[^/]+/pqn_v5\.py$' >/dev/null; then echo "forbidden entry in zip" >&2; exit 1; fi
unzip -Z1 "$OUT" | grep -qx 'pqn_v5.py' || { echo "pqn_v5.py is not at the zip root" >&2; exit 1; }
echo "wrote $OUT"; unzip -Z "$OUT" | tail -n +2
