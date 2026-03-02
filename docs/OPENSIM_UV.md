# OpenSim Installation for UV Environments

## The Problem

OpenSim is **not on PyPI** for Python 3.9+. Conda is the only official distribution channel.
But `conda install` into a uv venv fails because conda resolves against its **own** base Python version, not the venv's.

## The Solution

Create a **temporary** conda env matching your Python version, install opensim there,
copy the package + shared libraries into the uv venv, then delete the temp env.

This has been tested and confirmed working:
- **OpenSim 4.5.2** in a **uv venv with Python 3.11** on Ubuntu

## Usage

```bash
source .venv/bin/activate
./install_opensim_uv.sh
```

## What the Script Does

1. Creates a temporary conda environment with matching Python version
2. Installs opensim via `conda install -c opensim-org opensim`
3. Copies the `opensim` package and its `libosim*`/`libcasadi*` shared libraries into `$VIRTUAL_ENV/lib/pythonX.Y/site-packages/opensim/`
4. Deletes the temporary conda environment

After this, opensim works in the uv venv with no `LD_LIBRARY_PATH` needed.

## Prerequisites

- An activated uv virtual environment (Python 3.10, 3.11, or 3.12)
- `conda` available on PATH (miniconda is sufficient)

## Supported Versions

Per [OpenSim docs](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python):

| OpenSim | Python        | Moco | Arm64 Mac |
|---------|---------------|------|-----------|
| 4.5.2   | 3.10-3.12     | Yes  | Yes       |
| 4.5.1   | 3.9-3.11      | Yes  | Yes       |

## Future: Native Pip Wheels

The OpenSim team is building pip-installable wheels ([PR #4255](https://github.com/opensim-org/opensim-core/pull/4255)).
Once merged and published to PyPI, a simple `uv pip install opensim` will work.

## References

- [OpenSim Python docs](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python)
- [OpenSim conda packages](https://anaconda.org/opensim-org/opensim)
- [Wheels PR (in progress)](https://github.com/opensim-org/opensim-core/pull/4255)
