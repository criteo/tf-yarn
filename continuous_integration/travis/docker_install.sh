#!/usr/bin/env bash
set -xe

cd ~/tf-yarn
pip install -r requirements.txt
pip install -v --no-deps -e .

pip install pytest

pip freeze
