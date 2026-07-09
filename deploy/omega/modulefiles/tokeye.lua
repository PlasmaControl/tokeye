-- TokEye (diiid build) — Lmod modulefile.
--
-- Local use (no GitHub access needed):
--     module use <repo>/deploy/omega/modulefiles   # or /cscratch/share/tokeye/modulefiles
--     module load tokeye
-- Later, the css-omega-modules PR ships this same content as tokeye/default.lua.

help([[
TokEye — ML detection of fluctuating modes in spectrograms (diiid build).

No-X11 web app (run on somega, e.g. omega14):
    tokeye app                 # binds 127.0.0.1:7860
then from your laptop:
    ssh -N -L 7860:localhost:7860 <you>@<that-node>.gat.com
    open http://localhost:7860 -> "DIII-D" tab

CLI:  tokeye fetch --shot <N> --diag mag   |   tokeye run   |   tokeye app
Reads DIII-D shots from MDSplus (atlas.gat.com); cache at $TOKEYE_CACHE.
]])
whatis("Name        : tokeye (diiid)")
whatis("Description : ML mode detection on DIII-D spectrograms; Gradio web app")

-- Env root. Override with `TOKEYE_DIR` (e.g. once /fusion/projects/codes/tokeye
-- exists, point it there — a one-line change, no modulefile edit needed).
local root = os.getenv("TOKEYE_DIR") or "/cscratch/share/tokeye"

-- One x86_64 env serves login/somega CPU + V100 + (x86_64) H100. Only a
-- Grace-Hopper (aarch64) target would need a separate native env; the branch
-- ignores hopr for now, but the hook keeps `module load tokeye` arch-correct.
local arch = capture("uname -m"):gsub("%s+", "")
local env  = (arch == "aarch64") and "env-aarch64" or "env-x86_64"

prepend_path("PATH", pathJoin(root, env, "bin"))   -- MDSplus is bundled in the env

-- The default login environment loads the cluster's python/3.7 + mdsplus/d3d
-- modules, which export a PYTHONPATH pointing at the cluster's py3.7 MDSplus.
-- That shadows this env's own (conda-forge) MDSplus and crashes under numpy 2.x
-- (`cannot import name 'string_'`). This env is self-contained, so drop the
-- inherited PYTHONPATH while tokeye is loaded (Lmod restores it on unload).
unsetenv("PYTHONPATH")

setenv("TOKEYE_DIR", root)
setenv("TOKEYE_CACHE", pathJoin(root, "cache"))     -- shared shot cache on /cscratch
