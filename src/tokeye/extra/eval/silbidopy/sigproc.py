# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
# Edited by Joshua Zingale 2023

import logging

import numpy as np


def frame_signal(signal, frame_len: int, frame_step: float):
    '''Frame a signal into overlapping frames. The input must have frame_len <= len(signal)

    :param signal: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
                       The is a fuzzy number and may be increased or decreased slightly
                       between frames to allow that the beginning of the first frame be signal[0]
                       and the end of the last frame be signal[-1]
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    '''

    slen = len(signal)

    if frame_len > slen:
        raise ValueError("frame_len must be less than or equal to len(signal).")

    # Get the total number of frames, rounding down
    num_frames = 1 + round((slen - frame_len) / frame_step)

    # Get the first index of each frame
    starting_indices = np.linspace(0, slen - frame_len, num_frames).round().astype(int)

    # Get set of indices that will make up each frame's values
    frame_indices = np.arange(0, frame_len) + starting_indices.reshape(-1, 1)

    return signal[frame_indices]


def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.', numpy.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)


