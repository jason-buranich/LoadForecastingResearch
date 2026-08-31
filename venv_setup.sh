#!/bin/bash
echo "Rebuilding virtual environment..."
uv python install 3.12
uv venv --python 3.12 myenv
source myenv/bin/activate
uv pip install --upgrade pip
uv pip install -r venv_reqs.txt
echo "Virtual environment setup complete."