#!/bin/bash
echo "Rebuilding virtual environment..."
uv python install 3.12
uv venv --python 3.12 myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r venv_reqs.txt
echo "Virtual environment setup complete."