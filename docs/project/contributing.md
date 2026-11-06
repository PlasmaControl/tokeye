# Contributing

[uv](https://docs.astral.sh/uv/) is the dev tool for this repo:
```bash
git clone git@github.com:PlasmaControl/TokEye.git
cd TokEye
uv sync --dev --extra all                # dev tools (pytest, ruff, etc.) + every extra (web app, ae model, HDF5)
uv sync --dev --extra all --group train  # the same + training deps (lightning, h5py, etc.)
```

Each `uv sync` makes `.venv` match its flags exactly, removing any extra or group you leave out, so name everything you need in one command. PyTorch comes as the default build (GPU/CUDA on Linux).

Additionally, run these the first time
```bash
uv run pre-commit install
uv run ruff check .
uv run pytest
```

This creates a `.venv/`; activate it with `source .venv/bin/activate`, or prefix commands with `uv run`.

Do not commit files over 2 MB. See the [repository](https://github.com/PlasmaControl/tokeye) for issue reporting and contribution guidelines.
