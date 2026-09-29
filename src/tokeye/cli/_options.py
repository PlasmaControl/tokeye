"""Shared argparse options.

The spectrogram flags are generated from :class:`tokeye.SpectrogramConfig`,
so every subcommand accepts the same flags with the same defaults and help.
Only :mod:`tokeye.config` (standard library) is imported here, keeping
``tokeye --help`` fast.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import sys

from tokeye.config import DEFAULT_MODEL, SpectrogramConfig

_FIELDS = dataclasses.fields(SpectrogramConfig)
# CLI-only notes appended to a field's generated help.
_HELP_NOTES = {
    "window": "; checked before any input is read, even when every input is 2D",
}
# The floor is tokeye.inference.MIN_TILE; importing it would import torch.
TILE_HELP = (
    "tile side in pixels for long inputs: auto (untiled up to 2^21 pixels), "
    "none, or an int >= 512; for big_tf_unet, tiled output matches untiled "
    "output to float32 rounding"
)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# -v/--verbose, a parent of the top-level parser and of every subcommand.
# SUPPRESS: an absent flag sets nothing, so `tokeye -v run` and `tokeye run -v`
# both work and neither position overrides the other; read it with
# getattr(args, "verbose", False).
VERBOSE = argparse.ArgumentParser(add_help=False)
VERBOSE.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=argparse.SUPPRESS,
    help="show debug logs on stderr, including tracebacks",
)


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None


def _integer(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def positive_float(text: str) -> float:
    """argparse ``type=`` for a finite number > 0."""
    value = _number(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text}")
    return value


def unit_float(text: str) -> float:
    """argparse ``type=`` for a finite number in [0, 1]."""
    value = _number(text)
    if not 0.0 <= value <= 1.0:  # False for nan too
        raise argparse.ArgumentTypeError(f"must be a number in [0, 1], got {text}")
    return value


def finite_float(text: str) -> float:
    """argparse ``type=`` for any finite number (not nan or inf)."""
    value = _number(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text}")
    return value


def nonnegative_int(text: str) -> int:
    """argparse ``type=`` for an integer >= 0."""
    value = _integer(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer >= 0, got {text}")
    return value


def positive_int(text: str) -> int:
    """argparse ``type=`` for an integer >= 1."""
    value = _integer(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {text}")
    return value


def nonempty_str(text: str) -> str:
    """argparse ``type=`` for a string that is not empty or only whitespace."""
    if not text.strip():
        raise argparse.ArgumentTypeError(f"must be a non-empty string, got {text!r}")
    return text


def add_spectrogram_options(parser: argparse.ArgumentParser) -> None:
    """Add one flag per :class:`SpectrogramConfig` field.

    Every flag defaults to ``None`` ("not given"), so
    :func:`config_from_args` only overrides what the user typed. Boolean
    fields get ``--x/--no-x`` pairs.
    """
    group = parser.add_argument_group("spectrogram options")
    for field in _FIELDS:
        note = _HELP_NOTES.get(field.name, "")
        help_text = f"{field.metadata['help']}{note} (default: {field.default})"
        if isinstance(field.default, bool):
            group.add_argument(
                _flag(field.name),
                action=argparse.BooleanOptionalAction,
                default=None,
                help=help_text,
            )
        else:
            group.add_argument(
                _flag(field.name),
                type=type(field.default),
                default=None,
                metavar=field.name.upper(),
                help=help_text,
            )
    # Pre-1.0 spelling of --no-clip-dc; hidden from --help.
    group.add_argument("--keep-dc", action="store_true", help=argparse.SUPPRESS)


def config_from_args(args: argparse.Namespace) -> SpectrogramConfig:
    """Build the :class:`SpectrogramConfig` the parsed flags describe.

    Raises
    ------
    ValueError
        Out-of-range values (e.g. ``--clip-low 99 --clip-high 1``), an empty
        ``--window``, or ``--keep-dc`` combined with ``--clip-dc``.
    """
    changes = {
        field.name: getattr(args, field.name)
        for field in _FIELDS
        if getattr(args, field.name) is not None
    }
    if args.keep_dc:
        print("warning: --keep-dc is deprecated; use --no-clip-dc", file=sys.stderr)
        if changes.get("clip_dc") is True:
            raise ValueError("--keep-dc and --clip-dc contradict each other")
        changes["clip_dc"] = False
    return SpectrogramConfig.from_dict(changes)


def add_model_options(
    parser: argparse.ArgumentParser, default: str = DEFAULT_MODEL
) -> None:
    """``--model`` and ``--device``."""
    parser.add_argument(
        "--model",
        default=default,
        help="registry name or path to a .pt/.pt2 checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, mps or auto; auto = CUDA, then MPS, then CPU",
    )


def add_output_options(
    parser: argparse.ArgumentParser,
    default_dir: str,
    png_default: bool | None = None,
) -> None:
    """``--output-dir``, plus ``--png/--no-png`` (``args.png``) unless
    ``png_default`` is ``None``."""
    parser.add_argument(
        "--output-dir",
        default=default_dir,
        help="directory to write outputs to (default: %(default)s)",
    )
    if png_default is not None:
        parser.add_argument(
            "--png",
            action=argparse.BooleanOptionalAction,
            default=png_default,
            help="write a mask-overlay preview PNG per input",
        )


def add_fs_option(parser: argparse.ArgumentParser) -> None:
    """``--fs``: the sampling rate for every input."""
    parser.add_argument(
        "--fs",
        type=positive_float,
        default=None,
        help=(
            "sampling rate [Hz] for every input; enables physical time and "
            "frequency axes (default: read from the file when recorded)"
        ),
    )


def add_key_option(parser: argparse.ArgumentParser) -> None:
    """``--key``: the array to read from each container input."""
    parser.add_argument(
        "--key",
        type=nonempty_str,
        default=None,
        metavar="NAME",
        help=(
            "array to read from .npz/.mat/.h5 inputs: an entry name, or an "
            "HDF5 dataset path or its trailing path components, such as "
            "tree/pointname/data (default: the one named data, signal, x, "
            "spectrogram, values or y, else the only numeric one)"
        ),
    )


def tile_value(text: str) -> int | str | None:
    """argparse ``type=`` for ``--tile``: ``auto``, ``none`` or an integer.

    ``auto`` and ``none`` are case-insensitive. The range check is
    :func:`tokeye.inference.check_tile`, which runs before the model loads.
    """
    word = text.lower()
    if word == "auto":
        return "auto"
    if word == "none":
        return None
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    raise argparse.ArgumentTypeError(f"expected auto, none or an integer, got {text!r}")


def add_tile_option(parser: argparse.ArgumentParser) -> None:
    """``--tile`` (``args.tile``): ``"auto"``, ``None`` or an int."""
    parser.add_argument(
        "--tile",
        type=tile_value,
        default="auto",
        metavar="auto|none|N",
        help=TILE_HELP,
    )
