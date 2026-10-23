"""
Generates a realistic SYNTHETIC ground-vibration dataset -> dataset.csv
Pipeline: raw window -> FIR band-pass (5-400 Hz) -> 6 features.
Classes overlap on purpose (varying distance/SNR, wind, mains hum, spikes,
per-session sensor gain and noise floor).
"""
import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

FS = 2000
N = FS                      # 1-second windows
T = np.arange(N) / FS
SEED = 42
rng = np.random.default_rng(SEED)

FIR = firwin(101, [5, 400], pass_zero=False, fs=FS, window="hamming")
LABELS = ["none", "footsteps", "vehicle", "other"]
import os
SNR_LO = float(os.environ.get("SNR_LO", 8))
SNR_HI = float(os.environ.get("SNR_HI", 28))
WIND_MAX = float(os.environ.get("WIND_MAX", 0.01))
SESSIONS_PER_CLASS = 12
WINDOWS_PER_SESSION = 20     # -> 240 windows per class, 960 total


def burst(t0, f, decay):
    """Decaying sine burst starting at t0."""
    x = np.zeros(N)
    m = T >= t0
    tt = T[m] - t0
    x[m] = np.exp(-tt / decay) * np.sin(2 * np.pi * f * tt + rng.uniform(0, 2 * np.pi))
    return x


def footsteps(amp):
    rate = rng.uniform(1.2, 2.8)
    f = rng.uniform(10, 60)
    x = np.zeros(N)
    t0 = rng.uniform(0, 1 / rate)
    while t0 < 1:
        x += amp * rng.uniform(0.5, 1.5) * burst(t0, f * rng.uniform(0.85, 1.15),
                                                   rng.uniform(0.03, 0.09))
        t0 += (1 / rate) * rng.uniform(0.8, 1.2)
    return x


def vehicle(amp):
    f0 = rng.uniform(15, 80)
    x = np.zeros(N)
    for k, a in zip([1, 2, 3], [1.0, rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.5)]):
        x += a * np.sin(2 * np.pi * k * f0 * T + rng.uniform(0, 2 * np.pi))
    return amp * x * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * T))


def other(amp):
    x = np.zeros(N)
    for _ in range(rng.integers(1, 4)):
        x += amp * rng.uniform(0.5, 1.5) * burst(rng.uniform(0, 0.9),
                                                  rng.uniform(60, 250),
                                                  rng.uniform(0.01, 0.04))
    return x


def environment(noise_floor, wind, hum):
    """Noise present in every class."""
    x = rng.normal(0, noise_floor, N)
    # wind: low-frequency coloured noise
    w = np.cumsum(rng.normal(0, 1, N))
    w = w - np.convolve(w, np.ones(200) / 200, mode="same")
    x += wind * w / (np.std(w) + 1e-9)
    # mains hum
    x += hum * np.sin(2 * np.pi * 50 * T + rng.uniform(0, 2 * np.pi))
    # occasional spike
    if rng.random() < 0.1:
        x[rng.integers(0, N)] += rng.uniform(0.5, 2) * noise_floor * 20
    return x


def features(x):
    y = lfilter(FIR, 1.0, x)
    rms = np.sqrt(np.mean(y ** 2))
    peak = np.max(np.abs(y))
    spec = np.abs(np.fft.rfft(y * np.hanning(N))) ** 2
    freqs = np.fft.rfftfreq(N, 1 / FS)
    dom = freqs[1:][np.argmax(spec[1:])]
    tot = spec[(freqs >= 5) & (freqs <= 400)].sum() + 1e-12
    e = lambda a, b: spec[(freqs >= a) & (freqs < b)].sum() / tot
    return rms, dom, peak, e(5, 50), e(50, 150), e(150, 400)


rows = []
for label in LABELS:
    for s in range(SESSIONS_PER_CLASS):
        # per-session conditions (distance, surface, sensor gain, weather)
        snr_db = rng.uniform(SNR_LO, SNR_HI)
        noise_floor = rng.uniform(0.01, 0.05)
        gain = rng.uniform(0.7, 1.4)
        wind = rng.uniform(0, WIND_MAX)
        hum = rng.uniform(0, 0.02)
        amp = noise_floor * 10 ** (snr_db / 20)
        for _ in range(WINDOWS_PER_SESSION):
            a = amp * rng.uniform(0.6, 1.4)
            sig = {"none": np.zeros(N), "footsteps": footsteps(a),
                   "vehicle": vehicle(a), "other": other(a)}[label]
            x = gain * (sig + environment(noise_floor, wind, hum))
            rows.append((*features(x), label, f"{label}_s{s}"))

df = pd.DataFrame(rows, columns=["rms", "dominant_freq", "peak",
                                 "e_5_50", "e_50_150", "e_150_400",
                                 "label", "session"])
df.to_csv("dataset.csv", index=False)
print(df.groupby("label")[["rms", "dominant_freq", "peak",
                           "e_5_50", "e_50_150", "e_150_400"]].mean().round(3))
print("\nSaved dataset.csv:", df.shape)
