-- TokEye (diiid build) — Lmod modulefile.
--
-- Local use (no GitHub access needed):
--     module use <repo>/deploy/omega/modulefiles   # or /cscratch/share/tokeye/modulefiles
--     module load tokeye
-- Later, the css-omega-modules PR ships this same content as tokeye/default.lua.

help([[
TokEye — ML detection of fluctuating modes in spectrograms (diiid build).

Native desktop GUI (over NoMachine / X11 — recommended on somega, e.g. omega14):
    tokeye                       # opens the DIII-D GUI (spectrogram + toroidal modespec)
    tokeye gui --view modespec   # open straight on a view

No-X11 web app (alternative; SSH-forward the port from your laptop):
    tokeye app                   # binds 127.0.0.1:7860
    ssh -N -L 7860:localhost:7860 <you>@<that-node>.gat.com  ->  http://localhost:7860

CLI:  tokeye fetch --shot <N> --diag mag  |  tokeye run  |  tokeye app  |  tokeye gui
Reads DIII-D shots from MDSplus (atlas.gat.com); cache at $TOKEYE_CACHE.
GUI won't open? Name the missing X plugin:  QT_DEBUG_PLUGINS=1 tokeye gui --self-test
]])
whatis("Name        : tokeye (diiid)")
whatis("Description : ML mode detection on DIII-D spectrograms; native GUI + Gradio web app")

-- Env root. Override with `TOKEYE_DIR` (e.g. once /fusion/projects/codes/tokeye
-- exists, point it there — a one-line change, no modulefile edit needed).
local root = os.getenv("TOKEYE_DIR") or "/cscratch/share/tokeye"

-- One x86_64 env serves login/somega CPU + V100 + (x86_64) H100. Only a
-- Grace-Hopper (aarch64) target would need a separate native env; the branch
-- ignores hopr for now, but the hook keeps `module load tokeye` arch-correct.
local arch = capture("uname -m"):gsub("%s+", "")
local env  = (arch == "aarch64") and "env-aarch64" or "env-x86_64"

prepend_path("PATH", pathJoin(root, env, "bin"))   -- MDSplus is bundled in the env

-- Make the conda env's own compiled libs win over the node's system /lib64. A
-- fresh somega login exports a system LD_LIBRARY_PATH (old gcc/libstdc++ without
-- GLIBCXX_3.4.29); without this prepend, importing torch first can load the system
-- libstdc++ and then break numpy 2.x (`Importing the numpy C-extensions failed`).
-- This is exactly what `conda activate` does. Lmod restores it on unload.
prepend_path("LD_LIBRARY_PATH", pathJoin(root, env, "lib"))

-- The default login environment loads the cluster's python/3.7 + mdsplus/d3d
-- modules, which export a PYTHONPATH pointing at the cluster's py3.7 MDSplus.
-- That shadows this env's own (conda-forge) MDSplus and crashes under numpy 2.x
-- (`cannot import name 'string_'`). This env is self-contained, so drop the
-- inherited PYTHONPATH while tokeye is loaded (Lmod restores it on unload).
unsetenv("PYTHONPATH")

-- The native GUI (PySide6/Qt6) renders through the xcb platform plugin over
-- NoMachine/X11. Pin it so Qt never probes wayland/eglfs on a login node
-- (use QT_QPA_PLATFORM=offscreen for a headless self-test). Lmod restores on unload.
setenv("QT_QPA_PLATFORM", "xcb")

setenv("TOKEYE_DIR", root)
setenv("TOKEYE_CACHE", pathJoin(root, "cache"))     -- shared shot cache on /cscratch
