#!/bin/bash
source ~/videogen/venv/bin/activate
exec uvicorn image_generation.app:app --host 0.0.0.0 --port 8000
