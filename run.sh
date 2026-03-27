#!/bin/zsh
# LYME Security Camera — easy launcher
# Usage: ./run.sh  (or double-click in Finder after chmod +x)

cd "$(dirname "$0")"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

source .venv/bin/activate
python main.py
