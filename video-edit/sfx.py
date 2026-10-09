"""Synthesize SFX (whoosh, pop, ding, impact, click) + a light music bed,
save each as a wav in OUTDIR, and write the timed SFX+music mix track."""
import os, sys, numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
DUR = float(sys.argv[2])
rng = np.random.default_rng(7)
t_ = lambda d: np.arange(int(d * SR)) / SR

def env(n, a, r):
    e = np.ones(n); A = int(a * SR); R = int(r * SR)
    if A: e[:A] = np.linspace(0, 1, A)
    if R: e[-R:] *= np.linspace(1, 0, R) ** 2
    return e

def band(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), x)

def whoosh(d=0.45, up=True):
    n = int(d * SR); x = rng.standard_normal(n)
    # sweep a bandpass across 12 overlapping blocks
    out = np.zeros(n); blocks = 24
    for k in range(blocks):
        f = k / (blocks - 1); f = f if up else 1 - f
        c = 300 * (12 ** f)
        y = band(x, c * 0.6, min(c * 1.6, SR / 2 - 100))
        w = np.exp(-0.5 * ((np.arange(n) / n - k / (blocks - 1)) / 0.06) ** 2)
        out += y * w
    e = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return out * e / np.abs(out).max() * 0.6

def pop(f0=900):
    t = t_(0.09)
    s = np.sin(2 * np.pi * (f0 * t + 1800 * t ** 2 / 0.09 * 0.5)) * np.exp(-t * 55)
    return s * 0.55

def ding(f=1320):
    t = t_(0.9)
    s = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t * dcy)
            for a, m, dcy in [(1, 1, 5), (.5, 2.01, 8), (.25, 3.0, 12), (.15, 4.2, 16)])
    return s * env(len(t), 0.002, 0.05) * 0.35

def impact():
    t = t_(0.9)
    sub = np.sin(2 * np.pi * (120 * np.exp(-t * 6) + 38) * t) * np.exp(-t * 4.5)
    nz = band(rng.standard_normal(len(t)), 80, 2500) * np.exp(-t * 18) * 0.5
    s = sub + nz
    return s / np.abs(s).max() * 0.85

def click():
    t = t_(0.05)
    return (np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 160) + band(rng.standard_normal(len(t)), 2000, 8000) * np.exp(-t * 220) * 0.4) * 0.5

def music(d, bpm=100):
    """soft lo-fi bed: kick, shaker, warm chord pad (Am F C G)."""
    n = int(d * SR); out = np.zeros(n); beat = 60 / bpm
    kick_t = t_(0.35); kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 9)
    hat_t = t_(0.06); hat = band(rng.standard_normal(len(hat_t)), 6000, 14000) * np.exp(-hat_t * 70) * 0.25
    for i in range(int(d / beat * 2) + 1):
        p = int(i * beat / 2 * SR)
        if i % 2 == 0 and (i // 2) % 4 in (0, 2):
            seg = kick[: max(0, min(len(kick), n - p))]; out[p:p + len(seg)] += seg * 0.8
        seg = hat[: max(0, min(len(hat), n - p))]; out[p:p + len(seg)] += seg * (1 if i % 2 else 0.5)
    chords = [[220, 261.6, 329.6], [174.6, 220, 261.6], [130.8, 196, 261.6], [196, 246.9, 293.7]]
    bar = beat * 4
    for b in range(int(d / bar) + 1):
        p = int(b * bar * SR); L = min(int(bar * SR), n - p)
        if L <= 0: break
        tt = np.arange(L) / SR
        pad = sum(np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.005 * tt) for f in chords[b % 4])
        pad += 0.6 * np.sin(2 * np.pi * chords[b % 4][0] / 2 * tt)  # bass
        out[p:p + L] += pad * env(L, 0.25, 0.3) * 0.09
    return out

def save(name, x):
    wavfile.write(os.path.join(OUT, name), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return x

S = {
    "whoosh_up.wav": save("whoosh_up.wav", whoosh(0.45, True)),
    "whoosh_down.wav": save("whoosh_down.wav", whoosh(0.35, False)),
    "pop.wav": save("pop.wav", pop()),
    "ding.wav": save("ding.wav", ding()),
    "impact.wav": save("impact.wav", impact()),
    "click.wav": save("click.wav", click()),
}
M = save("music_bed.wav", music(DUR + 1) * 0.9)

# timeline (seconds) — matches render.py shots / pops
EV = [("impact.wav", 0.00, 1.0), ("whoosh_up.wav", 0.30, 0.8),                # hook
      ("pop.wav", 0.12, 0.8)]
for t in (2.70, 4.12, 6.55, 8.15, 9.65, 11.55, 12.70, 14.35):                  # shot changes
    EV.append(("whoosh_down.wav" if int(t) % 2 else "whoosh_up.wav", t - 0.18, 0.55))
for t in (0.7, 3.0, 5.6, 7.0, 12.2, 13.4):                                     # emoji pops
    EV.append(("pop.wav", t, 0.6))
EV += [("ding.wav", 8.36, 0.7), ("ding.wav", 9.74, 0.6),                       # "3" badge
       ("impact.wav", 10.81, 0.55), ("ding.wav", 10.85, 0.8),                  # ТЕГІН + 🎁
       ("whoosh_up.wav", 14.35, 0.7), ("click.wav", 14.85, 0.9)]               # card + button

n = int(DUR * SR); fx = np.zeros(n)
for name, t, g in EV:
    x = S[name] * g; p = int(t * SR); L = min(len(x), n - p)
    if L > 0: fx[p:p + L] += x[:L]
mus = M[:n] * 0.18
mus *= np.minimum(1, np.arange(n) / (0.3 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.6 * SR))
mix = fx + mus
wavfile.write(os.path.join(OUT, "_mix_track.wav"), SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
print("ok", len(EV), "events")
