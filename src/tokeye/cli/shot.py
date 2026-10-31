"""``tokeye <shot> [<signal>] [tmin] [tmax] [key=value ...]`` — one-shot analysis.

Fetch one DIII-D signal from MDSplus, bring it to the default analysis rate
(signals faster than 500 kHz are decimated to ~500 kHz — the model's training
rate and the current Mirnov rate modespec/pyspecview run on; slower signals stay
native), compute the spectrogram with the recommended (training-recipe) STFT,
run the TokEye model on CPU, and save a PNG + NPZ + JSON. The last thing
printed is ``Saved to <dir>``.

Parameters follow modespec's conventions — positional ``shot, tmin, tmax`` plus
``keyword=value`` pairs (comma-separated lists are accepted too), with the usual
``--keyword value`` flags as an equivalent::

    tokeye 190904 MPI66M067D
    tokeye 190904 MPI66M067D 2000 3000
    tokeye 190904 MPI66M067D tmin=2000,tmax=3000 nfft=2048 hop=256
    tokeye 190904 B1 fs=native fmax=600 out=~/tokeye_runs
    tokeye 0 mag                       # shot 0 = latest shot; 'mag' = default probe

The bare-number form is routed here by :func:`tokeye.cli.main`; ``tokeye shot
...`` is the explicit spelling. Module scope stays stdlib-only so ``tokeye
--help`` is instant (heavy imports happen inside :func:`_handle`).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import stat
import sys
import tempfile
import time
from datetime import UTC
from pathlib import Path

from tokeye.transforms import (
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

DEFAULT_RESULTS_DIR = "/fusion/projects/results/tokeye"
DEFAULT_SIGNAL = "MPI66M067D"  # the 'mag' preset's default probe
DEFAULT_THRESHOLD = 0.5

# modespec-style keyword -> argparse dest. Several modespec spellings
# (xmin/xmax, decimation, n_fft, ...) are accepted as aliases.
_KEY_ALIASES: dict[str, str] = {
    "shot": "shot",
    "signal": "signal",
    "point": "signal",
    "point1": "signal",
    "pointname": "signal",
    "probe": "signal",
    "tmin": "tmin",
    "xmin": "tmin",
    "tmax": "tmax",
    "xmax": "tmax",
    "fmin": "fmin",
    "fmax": "fmax",
    "nfft": "nfft",
    "n_fft": "nfft",
    "hop": "hop",
    "decim": "decim",
    "decimation": "decim",
    "fs": "fs",
    "rate": "fs",
    "fs_khz": "fs",
    "thresh": "threshold",
    "threshold": "threshold",
    "clip_low": "clip_low",
    "cliplow": "clip_low",
    "clip_high": "clip_high",
    "cliphigh": "clip_high",
    "keep_dc": "keep_dc",
    "keepdc": "keep_dc",
    "out": "out",
    "outdir": "out",
    "output": "out",
    "dir": "out",
    "tag": "tag",
    "model": "model",
    "backend": "backend",
    "threads": "threads",
    "noplot": "noplot",
    "nodata": "nodata",
    "cmap": "cmap",
    "dpi": "dpi",
}
_FLAG_KEYS = {"keep_dc", "noplot", "nodata"}

USAGE = """\
tokeye <shot> [<signal>] [tmin] [tmax] [key=value ...]

  shot     DIII-D shot number (0 = latest shot)
  signal   PTDATA pointname (e.g. MPI66M067D, B1, TECEF20, DENV2_UF, BESFU20)
           or a diagnostic preset (mag, mag_pol, mhr, ece, co2, bes) for its
           default probe. Default: MPI66M067D.
  tmin tmax  optional time window in ms (positional, like modespec)

Sampling rate: signals faster than 500 kHz are decimated to ~500 kHz (the
model's training rate and the current Mirnov rate modespec/pyspecview use),
e.g. B1-B8 2 MHz -> 500 kHz; slower signals stay at their native rate.

Parameters (modespec style key=value, comma lists OK; --key value also works):
  tmin, tmax    time window [ms]                        (default: whole shot)
  nfft          STFT window length [samples]            (default: 1024)
  hop           STFT hop [samples]                      (default: 128)
  fs            target sampling rate [kHz]; native = off (default: 500)
  decim         explicit decimation factor (overrides fs; 1 = native)
  fmin, fmax    frequency band shown in the PNG [kHz]   (default: full band)
  thresh        mask threshold for the overlay          (default: 0.5)
  clip_low, clip_high  percentile clip of the log spectrogram (default: 1, 99)
  keep_dc       keep the DC bin (flag)
  out           output directory   (default: $TOKEYE_RESULTS or
                /fusion/projects/results/tokeye; files go in <out>/<shot>/)
  tag           extra label appended to the file names
  model         registry name or .pt/.pt2 path           (default: big_tf_unet)
  backend       auto | openvino | torch                  (default: auto)
  threads       CPU threads              (default: Slurm allocation / CPU quota)
  noplot        skip the PNG;  nodata  skip the NPZ
  cmap, dpi     PNG colormap / resolution                (default: viridis, 150)

Examples:
  tokeye 190904 MPI66M067D
  tokeye 190904 MPI66M067D 2000 3000
  tokeye 190904 MPI66M067D tmin=2000,tmax=3000 nfft=2048 hop=256
  tokeye 190904 B1 fs=native fmax=600 out=~/tokeye_runs
"""


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def normalize_argv(argv: list[str]) -> list[str]:
    """Rewrite modespec-style tokens into argparse ``--flag`` form.

    ``tmin=2000,tmax=3000`` -> ``--tmin=2000 --tmax=3000``; a bare known flag word
    (``noplot``, ``keep_dc``) -> ``--noplot``. Unknown ``key=value`` keys raise
    ``SystemExit(2)`` with the list of known keys (typos should not be silent).
    """
    out: list[str] = []
    for token in argv:
        if token.startswith("-"):
            # argparse flag; normalise aliases in --key / --key=value too.
            body = token.lstrip("-")
            key, sep, value = body.partition("=")
            canon = _KEY_ALIASES.get(key.lower().replace("-", "_"))
            if canon is not None and token.startswith("--"):
                out.append(f"--{canon}{sep}{value}")
            else:
                out.append(token)
            continue
        parts = [p for p in token.split(",") if p] if "=" in token else [token]
        for part in parts:
            if "=" in part:
                key, _, value = part.partition("=")
                canon = _KEY_ALIASES.get(key.strip().lower())
                if canon is None:
                    _die(
                        f"unknown parameter {key!r} "
                        f"(known: {', '.join(sorted(set(_KEY_ALIASES)))})"
                    )
                if canon in _FLAG_KEYS:
                    if value.strip().lower() in {"1", "y", "yes", "true", "on"}:
                        out.append(f"--{canon}")
                    continue
                out.append(f"--{canon}={value.strip()}")
            elif part.lower() in _FLAG_KEYS:
                out.append(f"--{part.lower()}")
            else:
                out.append(part)
    return out


def _die(message: str) -> None:
    print(f"tokeye: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("positionals", nargs="*", metavar="SHOT [SIGNAL] [TMIN] [TMAX]")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--signal", default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--fmin", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--nfft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP)
    parser.add_argument("--decim", type=int, default=None)
    parser.add_argument("--fs", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--clip_low", type=float, default=DEFAULT_CLIP_LOW)
    parser.add_argument("--clip_high", type=float, default=DEFAULT_CLIP_HIGH)
    parser.add_argument("--keep_dc", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--backend", choices=["auto", "openvino", "torch"], default=None
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--noplot", action="store_true")
    parser.add_argument("--nodata", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--dpi", type=int, default=150)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokeye",
        usage=USAGE,
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    return parser


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "shot",
        help="Analyze one DIII-D shot signal and save PNG/NPZ/JSON "
        "(same as `tokeye <shot> <signal>`).",
        usage=USAGE.replace("tokeye <shot>", "tokeye shot <shot>", 1),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    parser.set_defaults(handler=_handle)


def parse(argv: list[str]) -> argparse.Namespace:
    """Parse a bare ``tokeye <shot> ...`` command line (modespec syntax allowed)."""
    return _resolve_positionals(build_parser().parse_args(normalize_argv(argv)))


def _resolve_positionals(args: argparse.Namespace) -> argparse.Namespace:
    """Fold ``SHOT [SIGNAL] [TMIN] [TMAX]`` positionals into the named fields."""
    pos = list(args.positionals or [])
    if pos and args.shot is None:
        if not _is_int(pos[0]):
            _die(f"expected a shot number, got {pos[0]!r}")
        args.shot = int(pos.pop(0))
    if pos and not _is_number(pos[0]) and args.signal is None:
        args.signal = pos.pop(0)
    nums = []
    for token in pos:
        if not _is_number(token):
            _die(f"unexpected argument {token!r} (parameters are key=value)")
        nums.append(float(token))
    if len(nums) > 2:
        _die(f"too many positional numbers {nums} (expected at most tmin tmax)")
    if nums and args.tmin is None:
        args.tmin = nums[0]
    if len(nums) == 2 and args.tmax is None:
        args.tmax = nums[1]
    if args.shot is None:
        _die("a shot number is required, e.g. `tokeye 190904 MPI66M067D`")
    args.positionals = []
    return args


# ── helpers used by the handler ─────────────────────────────────────────────────
def _results_root(out: str | None) -> Path:
    root = out or os.environ.get("TOKEYE_RESULTS") or DEFAULT_RESULTS_DIR
    return Path(os.path.expandvars(root)).expanduser()


def _fmt_num(v: float) -> str:
    return f"{v:g}".replace("-", "m") if v < 0 else f"{v:g}"


def file_stem(shot: int, signal: str, args: argparse.Namespace) -> str:
    """``<shot>_<signal>`` plus a short tag for every non-default parameter.

    Default runs get the plain name; custom runs get e.g.
    ``190904_MPI66M067D_t2000-3000_nfft2048`` so they never overwrite the
    default result. ``fmin``/``fmax``/``thresh``/``cmap`` only change the PNG,
    so they don't rename the data.
    """
    parts = [str(shot), signal.replace("/", "-")]
    if args.tmin is not None or args.tmax is not None:
        lo = _fmt_num(args.tmin) if args.tmin is not None else ""
        hi = _fmt_num(args.tmax) if args.tmax is not None else ""
        parts.append(f"t{lo}-{hi}")
    if args.nfft != DEFAULT_N_FFT:
        parts.append(f"nfft{args.nfft}")
    if args.hop != DEFAULT_HOP:
        parts.append(f"hop{args.hop}")
    if args.decim is not None:
        parts.append(f"decim{args.decim}")
    elif args.fs is not None:
        target = target_fs_hz(args.fs)
        if target != default_target_fs_hz():
            parts.append(f"fs{target / 1e3:g}k" if target else "native")
    if args.clip_low != DEFAULT_CLIP_LOW or args.clip_high != DEFAULT_CLIP_HIGH:
        parts.append(f"clip{_fmt_num(args.clip_low)}-{_fmt_num(args.clip_high)}")
    if args.keep_dc:
        parts.append("dc")
    if args.model:
        parts.append(Path(str(args.model)).stem)
    if args.tag:
        parts.append(str(args.tag).replace("/", "-").replace(" ", "_"))
    return "_".join(parts)


def default_target_fs_hz() -> float:
    """Default analysis rate: ``$TOKEYE_FS_KHZ`` (e.g. from ~/.tokeye/settings) or 500 kHz."""
    from tokeye.transforms import default_target_fs_hz as _default

    return _default()


def target_fs_hz(value: str | float | None) -> float:
    """Parse an ``fs=`` value [kHz] to Hz; ``native``/``0``/``off`` -> 0 (no decimation)."""
    if value is None:
        return default_target_fs_hz()
    text = str(value).strip().lower().removesuffix("khz")
    if text in {"native", "0", "off", "none", "full", ""}:
        return 0.0
    try:
        khz = float(text)
    except ValueError:
        _die(f"fs must be a rate in kHz or 'native' (got {value!r})")
    if khz < 0:
        _die(f"fs must be >= 0 kHz (got {value!r})")
    return khz * 1e3


def _fchmod_dir(d: Path, mode: int) -> None:
    """chmod a directory we just created, without ever following a symlink
    (another user could swap it for one in a non-sticky shared parent)."""
    try:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return
    try:
        if os.fstat(fd).st_uid == os.getuid():
            os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _make_shared_dir(path: Path) -> None:
    """mkdir -p for the shared results root (writable by everyone in the group).

    New dirs are setgid + sticky (3775): anyone can add results, nobody can
    replace or delete another user's. An existing shot dir that is a symlink
    planted by someone else is refused rather than written through.
    """
    missing = []
    p = path
    while not p.exists() and not p.is_symlink():
        missing.append(p)
        p = p.parent
    path.mkdir(parents=True, exist_ok=True)
    for d in reversed(missing):
        _fchmod_dir(d, 0o3775)
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) and st.st_uid != os.getuid():
        raise OSError(f"{path} is a symlink owned by another user; refusing to write there")


def _owned_by_someone_else(p: Path) -> bool:
    try:
        return os.lstat(p).st_uid != os.getuid()
    except FileNotFoundError:
        return False


def _atomic_write(final: Path, write) -> None:
    """Write ``final`` without ever following a symlink planted at its name.

    The data goes to a fresh ``mkstemp`` file (O_EXCL, mode set on the open fd),
    which is then renamed over ``final``; a rename replaces whatever entry is
    there instead of writing through it.
    """
    fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=f".{final.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o664)
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        Path(tmp).replace(final)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise


def _save_denied(path: Path, exc: Exception) -> int:
    print(
        f"tokeye: error: cannot write {path} ({exc}); save elsewhere with out=<dir>",
        file=sys.stderr,
    )
    return 2


def _resolve_signal(signal: str | None) -> tuple[str, str | None]:
    """``(pointname, diag_key)`` — a preset key resolves to its default probe."""
    from tokeye.sources.factory import active_diagnostics

    if not signal:
        return DEFAULT_SIGNAL, "mag"
    diags = active_diagnostics()
    if signal.lower() in diags:
        diag = diags[signal.lower()]
        return diag.default, diag.key
    pointname = signal.upper()
    for diag in diags.values():
        if pointname in {p.upper() for p in diag.pointnames}:
            return pointname, diag.key
    return pointname, None


def _handle(args: argparse.Namespace) -> int:
    args = _resolve_positionals(args)
    if args.nfft < 16 or args.hop < 1 or args.hop > args.nfft:
        _die(f"need nfft >= 16 and 1 <= hop <= nfft (got nfft={args.nfft}, hop={args.hop})")
    if args.decim is not None and args.decim < 1:
        _die(f"decim must be >= 1 (got {args.decim})")
    target_fs_hz(args.fs)  # validate early
    if args.tmin is not None and args.tmax is not None and args.tmax <= args.tmin:
        _die(f"tmax ({args.tmax}) must be greater than tmin ({args.tmin})")
    return run(args)


def run(args: argparse.Namespace) -> int:
    import getpass
    import json
    import platform
    from datetime import datetime

    import numpy as np

    from tokeye import accel
    from tokeye.sources.factory import get_source, latest_shot

    t_start = time.perf_counter()
    threads = accel.configure_threads(args.threads)

    shot = int(args.shot)
    if shot <= 0:
        shot = latest_shot() or 0
        if shot <= 0:
            _die("could not determine the latest shot from MDSplus (atlas.gat.com)")
        print(f"tokeye: latest shot is {shot}")
    pointname, diag_key = _resolve_signal(args.signal)

    tlim = None
    if args.tmin is not None or args.tmax is not None:
        tlim = (
            float(args.tmin) if args.tmin is not None else -np.inf,
            float(args.tmax) if args.tmax is not None else np.inf,
        )

    # 1. fetch ------------------------------------------------------------------
    print(f"tokeye: fetching {shot} {pointname} ...", flush=True)
    t0 = time.perf_counter()
    try:
        t_ms, x, fs = get_source().fetch(shot, pointname, tlim)
    except RuntimeError as exc:
        print(f"tokeye: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - atlas / tree / pointname failures
        print(f"tokeye: error: failed to fetch {shot} {pointname}: {exc}", file=sys.stderr)
        return 2
    t_fetch = time.perf_counter() - t0
    if x.size < args.nfft:
        print(
            f"tokeye: error: {shot} {pointname} has {x.size} samples"
            + (" in that window" if tlim else "")
            + f" (need at least nfft={args.nfft})",
            file=sys.stderr,
        )
        return 2
    if fs <= 0:
        print(f"tokeye: error: cannot determine the sampling rate of {pointname}", file=sys.stderr)
        return 2
    fs_native = float(fs)

    # 2. decimate to the analysis rate (default ~500 kHz; slower signals native) --
    from tokeye.transforms import auto_decimation

    target_hz = target_fs_hz(args.fs)
    decim = int(args.decim) if args.decim is not None else auto_decimation(fs, target_hz)
    if decim > 1:
        from scipy.signal import decimate

        if x.size // decim < args.nfft:
            _die(f"only {x.size // decim} samples left after decimation x{decim} "
                 f"(need nfft={args.nfft}); widen tmin/tmax or use fs=native")
        x = decimate(x, decim, ftype="fir")
        t_ms = t_ms[::decim][: x.size]
        fs = fs / decim

    # 3. spectrogram (identical to the GUI: tokeye.transforms.compute_stft) -------
    t0 = time.perf_counter()
    spec, t_axis, f_axis = accel.spectrogram(
        x,
        fs=fs,
        t0_ms=float(t_ms[0]),
        n_fft=args.nfft,
        hop=args.hop,
        clip_dc=not args.keep_dc,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )
    t_stft = time.perf_counter() - t0
    rate = "native" if decim == 1 else f"decimated x{decim} from {fs_native / 1e3:g} kHz"
    print(
        f"tokeye: {x.size:,} samples, {t_ms[0]:.1f}-{t_ms[-1]:.1f} ms, "
        f"fs={fs / 1e3:g} kHz ({rate}); n_fft={args.nfft} hop={args.hop} "
        f"-> spectrogram {spec.shape[0]} x {spec.shape[1]}",
        flush=True,
    )

    # 4. model -------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        runner = accel.load_runner(args.model, backend=args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"tokeye: error: could not load model {args.model or 'big_tf_unet'}: {exc}",
              file=sys.stderr)
        return 2
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    mask = runner.infer(spec)
    t_infer = time.perf_counter() - t0
    print(
        f"tokeye: inference {t_infer:.1f} s ({runner.backend}, {threads} threads; "
        f"model load {t_load:.1f} s)",
        flush=True,
    )

    # 5. save ----------------------------------------------------------------------
    out_dir = _results_root(args.out) / str(shot)
    try:
        _make_shared_dir(out_dir)
    except OSError as exc:
        print(f"tokeye: error: cannot create {out_dir}: {exc}", file=sys.stderr)
        return 2
    stem = file_stem(shot, pointname, args)
    exts = [e for e, on in (("png", not args.noplot), ("npz", not args.nodata), ("json", True)) if on]
    if any(_owned_by_someone_else(out_dir / f"{stem}.{e}") for e in exts):
        # Someone else already saved this result; the shared dir is sticky, so
        # keep theirs and save our own copy next to it.
        stem = f"{stem}_{getpass.getuser()}"
    saved: list[Path] = []

    from importlib.metadata import PackageNotFoundError, version

    try:
        tokeye_version = version("tokeye")
    except PackageNotFoundError:
        tokeye_version = "unknown"

    meta = {
        "shot": shot,
        "signal": pointname,
        "diagnostic": diag_key,
        "tmin_ms": float(t_ms[0]),
        "tmax_ms": float(t_ms[-1]),
        "requested_tlim_ms": [args.tmin, args.tmax],
        "fs_native_hz": fs_native,
        "fs_hz": float(fs),
        "target_fs_hz": target_hz or None,
        "decimation": decim,
        "n_fft": args.nfft,
        "hop": args.hop,
        "window": "hann",
        "clip_dc": not args.keep_dc,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "threshold": args.threshold,
        "spectrogram_shape": list(spec.shape),
        "model": runner.model_name,
        "backend": runner.backend,
        "threads": threads,
        "timing_s": {
            "fetch": round(t_fetch, 3),
            "stft": round(t_stft, 3),
            "model_load": round(t_load, 3),
            "inference": round(t_infer, 3),
        },
        "tokeye_version": tokeye_version,
        "git_commit": accel.git_commit(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "mask_channels": ["coherent", "transient"],
        "npz_layout": {
            "spectrogram": "float32 (n_freq, n_time): log1p|STFT|, percentile-clipped "
            "(the model input)",
            "mask": "uint8 (2, n_freq, n_time): model probability x 255 "
            "(channel 0 coherent, 1 transient); divide by 255",
            "t_ms": "frame centre times [ms]",
            "f_khz": "bin frequencies [kHz]",
            "meta": "this JSON as a string",
        },
    }

    if not args.noplot:
        from tokeye.shot_plot import save_shot_png

        png = out_dir / f"{stem}.png"
        plot_kwargs = {
            "spec": spec,
            "mask": mask,
            "t_ms": t_axis,
            "f_khz": f_axis,
            "title": f"DIII-D #{shot}  {pointname}",
            "subtitle": (
                f"fs={fs / 1e3:g} kHz ({'native' if decim == 1 else f'decim {decim}'})  "
                f"n_fft={args.nfft}  hop={args.hop}  thresh={args.threshold:g}  "
                f"{runner.model_name} [{runner.backend}]"
            ),
            "threshold": args.threshold,
            "fmin_khz": args.fmin,
            "fmax_khz": args.fmax,
            "cmap": args.cmap,
            "dpi": args.dpi,
        }
        try:
            _atomic_write(png, lambda fh: save_shot_png(fh, **plot_kwargs))
        except PermissionError as exc:
            return _save_denied(png, exc)
        saved.append(png)

    if not args.nodata:
        npz = out_dir / f"{stem}.npz"
        arrays = {
            "spectrogram": spec.astype(np.float32),
            "mask": np.round(np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8),
            "t_ms": t_axis.astype(np.float64),
            "f_khz": f_axis.astype(np.float64),
            "meta": np.array(json.dumps(meta)),
        }
        try:
            _atomic_write(npz, lambda fh: np.savez_compressed(fh, **arrays))
        except PermissionError as exc:
            return _save_denied(npz, exc)
        saved.append(npz)

    meta["timing_s"]["total"] = round(time.perf_counter() - t_start, 3)
    js = out_dir / f"{stem}.json"
    meta["files"] = [p.name for p in saved] + [js.name]
    payload = (json.dumps(meta, indent=2) + "\n").encode()
    try:
        _atomic_write(js, lambda fh: fh.write(payload))
    except PermissionError as exc:
        return _save_denied(js, exc)
    saved.append(js)

    print(f"Saved to {out_dir}/")
    for p in saved:
        print(f"  {p.name}")
    return 0
