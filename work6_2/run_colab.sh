#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] install dependencies"
python -m pip install -U pip
python -m pip install -r work6_2/requirements_colab.txt

echo "[2/3] run silhouette"
python -m src.Work6.main silhouette --steps 220 --image-size 224 --num-views 18 --seed 2026

echo "[3/3] run textured"
python -m src.Work6.main textured --steps 240 --image-size 224 --num-views 20 --seed 2027

echo "Done."
