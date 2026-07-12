# TokEye on stellar (Princeton) — deploy kit

Shared group install for `kolemen`: one uv venv + one Tcl group modulefile,
both living under `/projects/EKOLEMEN`. Users just `module load tokeye`.

## What's here

| file | purpose |
|---|---|
| `setup.sh` | idempotent installer: uv venv (+extras), weight prefetch, modulefile render, group perms |
| `modulefiles/tokeye.tcl` | Tcl modulefile **template** (`@ROOT@` etc. substituted by `setup.sh`) |
| `tokeye-app.sh` | launch the web app and print the SSH-tunnel one-liner |

## Install / update (maintainer, once per release)

```bash
ssh <netid>@stellar-vis1.princeton.edu     # internet + GPU node
cd /projects/EKOLEMEN/tokeye               # the shared checkout (branch: princeton)
git pull
./deploy/princeton/setup.sh                # add --bashrc to also wire your own shell
```

The script is safe to re-run; it re-syncs the venv, re-renders the modulefile
under the current `tokeye --version`, and re-opens group permissions
(`/projects/EKOLEMEN` is setgid `kolemen`; the script also sets `umask 002`).

## Use (everyone in `kolemen`)

One-time, because stellar's automatic group-module discovery probes
`/projects/KOLEMEN` (the unix group name uppercased) and this group's space is
`/projects/EKOLEMEN` — add to `~/.bashrc`:

```bash
module use --append /projects/EKOLEMEN/Modules/modulefiles-shared
```

Then:

```bash
ssh -X <netid>@stellar-vis1.princeton.edu   # or stellar-vis2 (both: 2x V100S, X11)
module load tokeye
tokeye                                       # native GUI, GPU inference
tokeye app                                   # web app; tunnel 7860 from your laptop
tokeye princeton-batch --shots 190000-190010 --outdir /scratch/gpfs/$USER/tokeye/run1
```

Shots are read from `/scratch/gpfs/EKOLEMEN/foundation_model`
(`$TOKEYE_FOUNDATION_DIR`); batch jobs go to the A100 `gpu` partition with the
weights served offline from the shared `HF_HOME` cache.

Full cluster notes: `docs/princeton-cluster.md`. Tab guide:
`docs/princeton_tab_usage.md`.
