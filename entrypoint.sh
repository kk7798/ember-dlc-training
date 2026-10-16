#!/bin/sh
set -eu
printf '%s\n' '[container] EMBER training entrypoint reached'
exec python -u -X faulthandler /app/train_dlc.py
