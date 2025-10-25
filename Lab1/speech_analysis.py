import math
from typing import Callable, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

plt.rcParams["figure.facecolor"] = "w"

SAMPLING_FREQUENCY = 16000

FIG_WIDTH = 15
FIG_HEIGHT = 5


def load_signal(filename: str) -> np.ndarray:
    """Load a speech signal from the given file."""
    with open(filename) as f:
        data = [int(line.rstrip()) for line in f]
        data = np.array(data[16:])  # Actual data starts at line 17
    return data


def speech_signal_observation(
    filename: str, title: str = "Speech signal"
) -> np.ndarray:
    """Load a speech signal from the given file and plot amplitude and log energy."""
    data = load_signal(filename)
    fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)

    # Plot signal
    axs[0].set_title(title)
    axs[0].set_xlabel("Time in samples")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, len(data))
    axs[0].set_ylim(-30000, 30000)
    axs[0].plot(data, lw="0.5")

    # Plot energy
    frame_shift = 64
    frame_size = 256
    max_frames = math.floor((len(data) - frame_size + frame_shift) / frame_shift)
    energy = np.zeros(max_frames)

    for i in range(max_frames):
        loc = i * frame_shift
        window = data[loc : loc + frame_size]
        energy[i] = np.dot(window.T, window)

    log_energy = 10 * np.log10(energy + 10e-3)
    log_energy_norm = log_energy - np.mean(log_energy)

    axs[1].set_title("Short-time energy plot of the above utterance")
    axs[1].set_xlabel("Time in terms of frames")
    axs[1].set_ylabel("Log energy")
    axs[1].set_xlim(0, len(log_energy_norm))
    axs[1].set_ylim(0, max(log_energy) + 5)
    axs[1].plot(log_energy_norm, ".")

    return data


def select_speech(
    data: np.ndarray, begin: int, end: int, title: str = "Windowed signal"
) -> np.ndarray:
    """Return a window of the given speech signal and plot it.

    The data between frame numbers `begin` (inclusive) and `end` (exclusive) is
    returned."""
    window = data[begin:end]
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.gca()
    ax.set_xlabel("Time in samples")
    ax.minorticks_on()
    ax.set_xticks(np.arange(0, len(window), 20))
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(window))
    ax.set_title(title)
    ax.grid()
    ax.grid(which="minor", ls=":")
    ax.plot(window)
    return window


def autocorrelation(
    data: np.ndarray, max_lags: int, title: str = "Autocorrelation", plot: bool = True
) -> np.ndarray:
    """Return and plot the autocorrelation of the given signal."""
    # Compute autocorrelation
    n = len(data)
    correlation = np.correlate(data, data, mode="full")
    correlation = correlation[n - 1 - max_lags : n + max_lags]

    # Plot autocorrelation
    if plot:
        fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, x in enumerate([correlation, correlation[max_lags + 1 :]]):
            axs[i].minorticks_on()
            axs[i].grid()
            axs[i].grid(which="minor", ls=":")
            axs[i].set_xlabel("Lag number")
            axs[i].set_ylabel("Amplitude")
            axs[i].set_xlim((0, len(x)))
            axs[i].plot(x)

        axs[0].set_title(title)
        axs[1].set_title("Autocorrelation (right half)")

    return correlation


def fourier_spectrum(
    data: np.ndarray,
    order: int = 512,
    sf: int = SAMPLING_FREQUENCY,
    title: str = "Fourier spectrum",
) -> None:
    """Computes and plots the fourier spectrum of the given signal."""
    # Compute fourier spectrum
    fourier = np.abs(np.fft.fft(data, n=order))
    log_fourier = 20 * np.log10(fourier)
    frequencies = [n * sf / (order * 1000) for n in range(order)]

    # Plot fourier spectrum
    fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)
    for i, (freq, x) in enumerate(
        [
            (frequencies, log_fourier),
            (frequencies[: order // 2], log_fourier[: order // 2]),
        ]
    ):
        axs[i].set_xlabel("Frequency (kHz)")
        axs[i].set_ylabel("Log amplitude")
        axs[i].set_xlim((0, max(freq)))
        axs[i].plot(freq, x)

    axs[0].set_title(title)
    axs[1].set_title("Fourier spectrum (left half)")


def spectrogram(
    data: np.ndarray,
    order: int,
    window: Optional[Callable[[int], np.ndarray]] = np.hanning,
    sf: int = SAMPLING_FREQUENCY,
    title="Spectrogram",
) -> None:
    """Compute and plot the spectrogram of the given signal."""
    window_fun = lambda x: window(len(x)) * x if window is not None else None
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.specgram(data, NFFT=order, window=window_fun, Fs=sf, cmap="magma")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")


def preemphasize(data: np.ndarray) -> np.ndarray:
    """Pre-emphasize to remove the spectral tilt due to glottal pulse spectrum."""
    diff_data = np.empty_like(data)
    diff_data[0] = data[0]
    diff_data[1:] = np.diff(data)
    return diff_data


def lp_spectrum(
    data: np.ndarray,
    lp_order: int,
    order: int,
    window: Optional[Callable[[int], np.ndarray]] = np.hanning,
    sf: int = SAMPLING_FREQUENCY,
    plot: bool = True,
) -> np.ndarray:
    """Compute and plot the LP spectrum."""
    windowed_data = preemphasize(data)
    if window is not None:
        windowed_data = window(len(windowed_data)) * windowed_data

    frequencies = [n * sf / (order * 1000) for n in range(order // 2)]

    # Computation of LP spectrum
    coefficients = librosa.lpc(windowed_data, order=lp_order)
    lp_spec = -20 * np.log10(np.abs(np.fft.fft(coefficients, order)))

    # Computation of Fourier spectrum
    fourier_spec = 20 * np.log10(np.abs(np.fft.fft(data, order)))
    # print(fourier_spec)

    if plot:
        fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, (spectrum, title) in enumerate(
            [(fourier_spec, "Fourier spectrum"), (lp_spec, "LP spectrum")]
        ):
            axs[i].minorticks_on()
            axs[i].grid()
            axs[i].grid(which="minor", ls=":")
            axs[i].set_title(title)
            axs[i].set_xlabel("Frequency (kHz)")
            axs[i].set_ylabel("Log amplitude")
            axs[i].set_xlim((0, max(frequencies)))
            axs[i].plot(frequencies, spectrum[: order // 2])

    return lp_spec


def lp_residual(
    data: np.ndarray,
    lp_order: int,
    window: Optional[Callable[[int], np.ndarray]] = np.hanning,
    sf: int = SAMPLING_FREQUENCY,
    plot: bool = True,
) -> np.ndarray:
    """Compute and plot the LP residual."""
    # Pre-emphasize, window the signal and compute LPC
    windowed_data = preemphasize(data)
    if window is not None:
        windowed_data = window(len(windowed_data)) * windowed_data
    coefficients = librosa.lpc(windowed_data, order=lp_order)

    residual = np.empty_like(data)
    padded_data = np.pad(data, (lp_order, 0))

    for i in range(len(data)):
        predict = 0
        for j in range(1, lp_order + 1):
            predict = predict + coefficients[j] * padded_data[i + lp_order - j]
        residual[i] = padded_data[i + lp_order] + predict

    # Plot signal and residual
    if plot:
        fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
        plt.subplots_adjust(hspace=0.3)
        for i, (signal, title) in enumerate(
            [(data, "Original signal"), (residual, "LP residual signal")]
        ):
            axs[i].minorticks_on()
            axs[i].grid()
            axs[i].grid(which="minor", ls=":")
            axs[i].set_title(title)
            axs[i].set_xlabel("Time in samples")
            axs[i].set_ylabel("Amplitude")
            axs[i].set_xlim(0, len(signal))
            axs[i].plot(signal)

    return residual


def speaker_variation(
    utterances: List[Tuple[str, int]],
    lp_order: int = 16,
    sample_length: int = 480,
    fft_order: int = 512,
    sf: int = SAMPLING_FREQUENCY,
) -> None:
    """Plots the LP spectra for a list of utterances."""
    frequencies = [n * sf / (fft_order * 1000) for n in range(fft_order // 2)]
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax = fig.gca()
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Log amplitude")
    ax.set_title("Comparison of LP spectra")
    ax.set_xlim(0, sf // 2000)

    for filename, start in utterances:
        st_data = load_signal(filename)[start : start + sample_length]
        spectrum = lp_spectrum(st_data, lp_order, fft_order, plot=False)
        ax.plot(frequencies, spectrum[: fft_order // 2], label=filename)
    ax.legend()


def sift(
    filename: str,
    lp_order: int = 10,
    frame_size: int = 480,
    frame_shift: int = 160,
    sf: int = SAMPLING_FREQUENCY,
):
    """Compute the pitch contour using the SIFT algorithm."""
    max_lags = 256

    b = [0.0357081667, -0.0069956244, -0.0069956244, 0.0357081667]
    a = [1.0, -2.34036589, 2.01190019, -0.61419218]

    data = load_signal(filename)
    max_frames = int(np.floor((len(data) - frame_size + frame_shift) / frame_shift))
    divisor = 1.8

    pitch = []
    for i in range(max_frames):
        idx = i * frame_shift
        frame = data[idx : idx + frame_size]
        frame = lfilter(b, a, frame)
        residual = lp_residual(frame, lp_order, plot=False)
        residual = autocorrelation(residual, max_lags, plot=False)
        # print(residual[296:300])

        max_residual = residual[max_lags] / divisor
        max_idx = max_lags
        for j in range(max_lags + (sf // 400), max_lags + (sf // 80)):
            if residual[j] > max_residual:
                max_residual = residual[j]
                max_idx = j

        max_idx = max_idx - max_lags
        if max_idx > 0:
            pitch.append(sf / max_idx)
            # print(pitch[-1], max_idx)
        else:
            pitch.append(0.0)

    fig, axs = plt.subplots(2, figsize=(FIG_WIDTH, FIG_HEIGHT * 2))
    plt.subplots_adjust(hspace=0.3)

    # Plot signal
    axs[0].set_title("Speech signal")
    axs[0].set_xlabel("Time in samples")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, len(data))
    axs[0].set_ylim(-30000, 30000)
    axs[0].plot(data, lw="0.5")

    # Plot pitch
    axs[1].set_title("Pitch contour")
    axs[1].set_xlabel("Time in samples")
    axs[1].set_ylabel("Pitch frequency (Hz)")
    axs[1].set_xlim(0, len(pitch))
    axs[1].plot(pitch, ".")

    return pitch
