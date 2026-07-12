#%Module1.0
## TokEye (princeton build) — Tcl group modulefile for stellar.
##
## TEMPLATE: deploy/princeton/setup.sh substitutes @ROOT@, @MODULES_ROOT@,
## @FOUNDATION_DIR@ and @HF_HOME@ and installs the result as
##     <modules-root>/tokeye/<version>
## so `module load tokeye` picks the newest version. Do not load this template
## file directly.

proc ModulesHelp { } {
    puts stderr "TokEye — ML detection of fluctuating modes in spectrograms (princeton build)."
    puts stderr ""
    puts stderr "Run on stellar-vis1/stellar-vis2 (they have V100S GPUs + X11):"
    puts stderr "    tokeye                     # native desktop GUI (ssh -X)"
    puts stderr "    tokeye app                 # web app on 127.0.0.1:7860 (SSH-forward it)"
    puts stderr "    tokeye princeton-batch --shots 190000-190010 --outdir /scratch/gpfs/\$USER/tokeye/run1"
    puts stderr ""
    puts stderr "Shots are read from the local foundation_model archive"
    puts stderr "(\$TOKEYE_FOUNDATION_DIR) — no MDSplus, no network fetch."
    puts stderr "GUI won't open? QT_DEBUG_PLUGINS=1 tokeye gui --self-test"
}

module-whatis "tokeye (princeton): ML mode detection on DIII-D spectrograms, foundation_model source"

set root "@ROOT@"

prepend-path PATH "$root/.venv/bin"

setenv TOKEYE_DIR             "$root"
# The princeton build reads local HDF5 shots; TOKEYE_SOURCE=mds would need MDSplus.
setenv TOKEYE_SOURCE          "foundation"
setenv TOKEYE_FOUNDATION_DIR  "@FOUNDATION_DIR@"
# Model weights are prefetched into the shared env by setup.sh — compute nodes
# have no internet, so point every user at the shared cache.
setenv HF_HOME                "@HF_HOME@"
setenv TOKEYE_RUNS_DIR        "/scratch/gpfs/$env(USER)/tokeye/runs"
# Where `tokeye princeton-batch` job scripts find this module tree.
setenv TOKEYE_MODULE_DIR      "@MODULES_ROOT@"
# Slurm defaults for princeton-batch (stellar A100 batch partition).
setenv TOKEYE_SLURM_PARTITION "gpu"
setenv TOKEYE_SLURM_GRES      "gpu:a100:1"
setenv TOKEYE_SLURM_TIME      "0-02:00:00"
# X11 (ssh -X / -Y) is the display path on the vis nodes.
setenv QT_QPA_PLATFORM        "xcb"

# The anaconda3 modules export PYTHONPATH, which poisons the uv venv with
# foreign site-packages. Clear it while tokeye is loaded. (Caveat: `module
# unload tokeye` cannot restore a previously set PYTHONPATH — reload the
# anaconda module if you need it back.)
unsetenv PYTHONPATH
