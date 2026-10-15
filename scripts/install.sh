#!/bin/sh
set -eu
plugin_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
state_dir=${JOBHUNTER_HOME:-"$HOME/.local/share/jobhunter"}
python_bin=${JOBHUNTER_PYTHON:-python3}
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
umask 077
mkdir -p "$state_dir"
"$python_bin" -m venv "$state_dir/runtime"
"$state_dir/runtime/bin/python" -m pip install --disable-pip-version-check "$plugin_root[sheets]"
"$state_dir/runtime/bin/jobhunter" --help
