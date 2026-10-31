-- -*- lua -*-
-- TokEye (DIII-D build) — shared CPU install on Omega.
--
--   module use /fusion/projects/codes/tokeye/modulefiles   (until CSS adds it)
--   module load tokeye
--
-- The install lives in /fusion/projects/codes/tokeye (env/, src/, models/, bin/).
-- This file only puts the `tokeye` launcher on PATH and sets TOKEYE_* defaults;
-- the launcher sets up the Python env for the tokeye process alone, so loading
-- the module never changes python/pip/LD_LIBRARY_PATH for the rest of the shell.

local root = "/fusion/projects/codes/tokeye"

help([[
TokEye — ML detection of coherent modes and transient events in spectrograms
(DIII-D build, CPU / OpenVINO).

  tokeye                         open the GUI (NoMachine or ssh -X, on somega)
  tokeye <shot> <signal> [tmin tmax] [key=value ...]
                                 analyze one signal; saves PNG + NPZ + JSON and
                                 prints "Saved to <dir>"
     tokeye 190904 MPI66M067D
     tokeye 190904 MPI66M067D 2000 3000
     tokeye 190904 MPI66M067D tmin=2000,tmax=3000 nfft=2048 hop=256
     tokeye 190904 B1 fs=native out=$HOME/tokeye_runs
     tokeye 0 mag                (shot 0 = latest shot; mag = default probe)
  tokeye shot --help             all parameters
  tokeye webui                   browser UI instead (tunnel port 7860 from your laptop)
  tokeye --help                  other subcommands

Defaults: signals faster than 500 kHz are decimated to ~500 kHz (the model's
training rate; slower signals stay native), n_fft=1024, hop=128, whole shot.
Memory: ~2.5 GB per second of 500 kHz signal. Results: $TOKEYE_RESULTS/<shot>/
(default /fusion/projects/results/tokeye). CPU threads follow your Slurm
allocation or the somega per-user CPU quota (override: TOKEYE_THREADS=N).
Your settings: ~/.tokeye/settings (written with commented defaults if missing).
Your data: /cscratch/$USER/.tokeye (signals, batch runs; /cscratch is cleaned
daily, TokEye rebuilds it on every start). Runbook: /fusion/projects/codes/tokeye/README.md
]])

whatis("Name        : tokeye")
whatis("Version     : 0.12.0-diiid")
whatis("Description : ML mode detection on DIII-D spectrograms (CPU/OpenVINO); GUI + CLI")
whatis("URL         : https://github.com/PlasmaControl/tokeye (branch diiid)")

prepend_path("PATH", pathJoin(root, "bin"))

setenv("TOKEYE_ROOT", root)
-- TOKEYE_RESULTS is deliberately not set here: the default
-- (/fusion/projects/results/tokeye) is built in, and users change it in
-- ~/.tokeye/settings, which an environment variable set here would override.
-- Per-user state is set up by the `tokeye` launcher on every start (never assumed
-- to exist): settings in ~/.tokeye, data (signal cache, batch runs, web-UI temp)
-- in /cscratch/$USER/.tokeye.
-- Batch jobs (web app "DIII-D Offline" tab / tokeye diiid-batch) re-load this
-- module on the compute node; CPU partition, no GPU request.
setenv("TOKEYE_MODULE_DIR", pathJoin(root, "modulefiles"))
setenv("TOKEYE_SLURM_PARTITION", "medium")
setenv("TOKEYE_SLURM_GRES", "none")
setenv("TOKEYE_SLURM_CPUS", "16")
