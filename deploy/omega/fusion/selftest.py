"""Install self-test: OpenVINO vs torch on the built-in synthetic signal.

Run by install.sh with the env's python. Checks that both CPU engines load from
the staged weights, agree with each other, and reports their speed. No MDSplus
or network needed. Exit 1 on a mismatch.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from tokeye import accel
from tokeye.examples import make_example_signal


def main() -> int:
    n = accel.configure_threads()
    x = make_example_signal()
    spec, t_ms, f_khz = accel.spectrogram(x, fs=500e3)
    print(f"  threads={n}  spectrogram {spec.shape}  ({t_ms[0]:.2f}..{t_ms[-1]:.2f} ms, "
          f"{f_khz[0]:.2f}..{f_khz[-1]:.1f} kHz)")
    masks = {}
    for backend in ("torch", "openvino"):
        runner = accel.load_runner(backend=backend)
        runner.infer(spec)  # first call: shape-specific kernel setup
        t0 = time.perf_counter()
        masks[backend] = runner.infer(spec)
        dt = time.perf_counter() - t0
        print(f"  {backend:9s} {dt * 1e3:7.1f} ms   coherent px>0.5: "
              f"{int((masks[backend][0] > 0.5).sum())}")
    diff = float(np.abs(masks["torch"] - masks["openvino"]).max())
    flips = int(((masks["torch"] > 0.5) != (masks["openvino"] > 0.5)).sum())
    print(f"  max |p_torch - p_openvino| = {diff:.2e}; thresholded pixels differing: {flips}")
    if diff > 1e-3:
        print("selftest: FAIL — OpenVINO and torch disagree", file=sys.stderr)
        return 1
    print("selftest: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
