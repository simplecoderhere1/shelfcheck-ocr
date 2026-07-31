#!/usr/bin/env bash
# Assemble a pushable Hugging Face Space from this repo.
#
# The Space needs three things that live elsewhere in the repo: the pipeline
# port, the ONNX models, and the char dict. Rather than duplicate them in git
# (they'd drift), copy them in at deploy time. The copies are gitignored here.
#
# Usage:
#   bash hf_space/prepare_space.sh                 # stage into hf_space/
#   bash hf_space/prepare_space.sh /path/to/space  # stage into a Space clone
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEST="${1:-$HERE}"

mkdir -p "$DEST/models"
cp "$REPO/modal_server/ocr_pipeline.py" "$DEST/ocr_pipeline.py"
cp "$REPO/models/det.onnx"      "$DEST/models/det.onnx"
cp "$REPO/models/rec.onnx"      "$DEST/models/rec.onnx"
cp "$REPO/models/rec_dict.txt"  "$DEST/models/rec_dict.txt"

if [ "$DEST" != "$HERE" ]; then
  cp "$HERE/app.py" "$HERE/requirements.txt" "$HERE/README.md" "$DEST/"
fi

# Running the app in the staging dir (to test it before pushing) leaves
# __pycache__ behind, which must not reach the Space.
rm -rf "$DEST/__pycache__"
printf '__pycache__/\n*.pyc\n' > "$DEST/.gitignore"

# eol=lf matters: the Space builds on Linux, and a CRLF Dockerfile breaks it.
# The LFS rule is required — both .onnx files exceed the 10MB plain-git limit.
printf '* text=auto eol=lf\n*.onnx filter=lfs diff=lfs merge=lfs -text\n' > "$DEST/.gitattributes"

echo "staged into $DEST:"
ls -la "$DEST" "$DEST/models"
cat <<'EOF'

.gitignore and .gitattributes (LFS + eol=lf) were written for you.

Next (in the Space clone):
  git lfs install --local
  git add -A
  git commit -m "ShelfCheck cloud OCR"
  git push
EOF
