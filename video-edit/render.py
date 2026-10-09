"""Edit Sau_Mal video in the style of the reference: punch-in zooms,
word-by-word mixed-font captions, emoji pops, "3" badge, outro card."""
import json, subprocess, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, FACE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS = 1080, 1920, 30
F = "/usr/share/fonts/"
FONTS = {
    "bold":   (F + "truetype/freefont/FreeSerifBold.ttf", 104),   # БОЛД СЕРИФ, ақ
    "gold":   (F + "truetype/freefont/FreeSerifItalic.ttf", 118),  # сары курсив
    "small":  (F + "opentype/inter/Inter-SemiBold.otf", 56),      # кіші ақ sans
    "thin":   (F + "opentype/inter/Inter-Regular.otf", 84),         # outro жіңішке
}
COL = {"bold": (255, 255, 255), "gold": (246, 205, 72), "small": (255, 255, 255), "thin": (255, 255, 255)}
UPPER = {"bold"}
EMOJI_FONT = ImageFont.truetype(F + "truetype/noto/NotoColorEmoji.ttf", 109)
_fc = {}
def font(style, scale=1.0):
    p, s = FONTS[style]; k = (style, round(scale, 2))
    if k not in _fc: _fc[k] = ImageFont.truetype(p, int(s * scale))
    return _fc[k]

# ---------- face track (smoothed) ----------
raw = json.load(open(FACE))
N = 496
xs = np.full(N, np.nan); ys = np.full(N, np.nan); ws = np.full(N, np.nan)
for i, x, y, _ in raw:
    if x is not None: xs[i], ys[i], ws[i] = x, y, _
idx = np.arange(N); ok = ~np.isnan(xs)
xs = np.interp(idx, idx[ok], xs[ok]); ys = np.interp(idx, idx[ok], ys[ok]); ws = np.interp(idx, idx[ok], ws[ok])
def smooth(a, k=21):
    pad = np.pad(a, k, mode="edge"); ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(pad, ker, "same")[k:-k]
xs, ys, ws = smooth(xs), smooth(ys), smooth(ws)

# ---------- shot plan: (start, end, zoom, push) ----------
SHOTS = [
    (0.00, 0.49, 1.18, -0.18),
    (0.49, 2.70, 1.42, 0.06),
    (2.70, 4.12, 1.00, 0.03),
    (4.12, 6.55, 1.32, 0.05),
    (6.55, 8.15, 1.00, 0.03),
    (8.15, 9.65, 1.38, 0.05),
    (9.65, 11.55, 1.08, 0.03),
    (11.55, 12.70, 1.40, 0.05),
    (12.70, 14.35, 1.00, 0.03),
    (14.35, 99.0, 1.12, 0.04),
]
def zoom_at(t):
    for s, e, z, p in SHOTS:
        if s <= t < e:
            return z + p * min(1, (t - s) / max(0.01, min(e, 16.53) - s))
    return 1.0

def frame_crop(img, i, t):
    z = zoom_at(t)
    if z <= 1.001: return img, (xs[i], ys[i], ws[i])
    cw, ch = W / z, H / z
    cx = xs[i]; cy = ys[i] + ch * 0.12  # face sits in upper part of the crop
    x0 = min(max(cx - cw / 2, 0), W - cw); y0 = min(max(cy - ch / 2, 0), H - ch)
    return (img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)),
            ((xs[i] - x0) * z, (ys[i] - y0) * z, ws[i] * z))

# ---------- captions ----------
# groups: (start, end, y, lines) ; line = list of (word, style, t_on)
def G(start, end, y, *lines): return (start, end, y, lines)
CAPS = [
    G(0.49, 2.70, 1250, [("Ағзаңызға", "gold", 0.49)], [("табиғи", "small", 1.10), ("қолдау", "bold", 1.45)],
      [("көрсетіңіз", "gold", 2.05)]),
    G(2.85, 4.12, 1180, [("Дала", "bold", 2.91)], [("саумалы", "gold", 3.35)]),
    G(4.15, 5.35, 1200, [("табиғи", "small", 4.15), ("жолымен", "gold", 4.55)], [("сіздің", "small", 4.95)]),
    G(5.35, 6.60, 1200, [("денсаулығыңызды", "bold", 5.35)], [("нығайтады", "gold", 5.95)]),
    G(6.68, 8.15, 1180, [("Дәл", "small", 6.68), ("қазір", "bold", 6.90)], [("сатып алатын", "gold", 7.25)],
      [("болсаңыз", "small", 7.70)]),
    G(8.36, 9.65, 1200, [("3", "badge", 8.36), ("қорапша", "bold", 8.55)], [("алсаңыз", "gold", 9.00)]),
    G(9.74, 11.50, 1180, [("3", "badge", 9.74), ("қорапшасы", "bold", 9.95)], [("ТЕГІН", "gold", 10.81)], [("беріледі", "small", 11.10)]),
    G(11.75, 13.25, 1180, [("Және де", "small", 11.75)], [("Түркияға", "gold", 12.20)]),
    G(13.30, 14.40, 1180, [("ваучер", "bold", 13.30)], [("беріледі", "gold", 13.70)]),
]
# outro: thin single words, top-right (like the reference ending)
OUTRO = [("толық", 14.51), ("ақпарат", 14.85), ("алу үшін", 15.25), ("төмендегі", 15.60), ("батырманы", 16.00)]

# emoji / badges: (char, t_on, t_off, x, y, size)
EMOJI = [
    ("🌿", 0.7, 2.70, 820, 470, 120),
    ("🥛", 3.0, 4.12, 830, 520, 130),
    ("💪", 5.6, 6.60, 840, 560, 130),
    ("🛒", 7.0, 8.15, 830, 520, 120),
    ("🎁", 10.85, 11.55, 820, 520, 140),
    ("🇹🇷", 12.2, 13.25, 820, 500, 130),
    ("✈️", 13.4, 14.40, 840, 520, 120),
]
BADGE3 = []

def ease_pop(dt, dur=0.18):
    """0..1 -> scale with slight overshoot."""
    if dt <= 0: return 0.0
    if dt >= dur: return 1.0
    x = dt / dur
    return 1 + 2.2 * (x - 1) ** 3 + 1.2 * (x - 1) ** 2

def text_img(word, style, scale=1.0):
    if style == "badge":
        b = BADGE_IM.resize((int(150 * scale), int(150 * scale)), Image.LANCZOS)
        im = Image.new("RGBA", (b.width + 70, b.height + 40), (0, 0, 0, 0)); im.alpha_composite(b, (35, 20)); return im
    txt = word.upper() if style in UPPER else word
    f = font(style, scale)
    b = f.getbbox(txt)
    w, h = b[2] - b[0] + 40, b[3] - b[1] + 40
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((20 - b[0], 20 - b[1]), txt, font=f, fill=(0, 0, 0, 230))
    sh = sh.filter(ImageFilter.GaussianBlur(9))
    im.alpha_composite(sh, (0, 3))
    im.alpha_composite(sh, (0, 3))
    ImageDraw.Draw(im).text((20 - b[0], 20 - b[1]), txt, font=f, fill=COL[style] + (255,))
    return im

def fit_scale(words):
    tot = sum(text_img(w, s).width for w, s, _ in words) + 10 * (len(words) - 1)
    return min(1.0, 980 / tot)

def emoji_img(ch, size):
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=EMOJI_FONT, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)

def paste_scaled(base, im, cx, cy, s, alpha=1.0):
    if s <= 0.02: return
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha)); im.putalpha(a)
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def badge(n):
    im = Image.new("RGBA", (190, 190), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse((5, 5, 185, 185), fill=(250, 208, 40, 255))
    f = ImageFont.truetype(F + "opentype/inter/Inter-Black.otf", 120)
    d.text((95, 98), n, font=f, fill=(255, 255, 255, 255), anchor="mm")
    return im

# outro card (like the profile card in the reference)
def make_card():
    cw, ch = 860, 430
    im = Image.new("RGBA", (cw + 60, ch + 60), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((30, 36, 30 + cw, 36 + ch), 36, fill=(0, 0, 0, 120))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, 30 + cw, 30 + ch), 36, fill=(255, 255, 255, 250))
    fb = ImageFont.truetype(F + "opentype/inter/Inter-Bold.otf", 52)
    fm = ImageFont.truetype(F + "opentype/inter/Inter-Medium.otf", 38)
    d.text((80, 70), "DALA SAUMAL", font=fb, fill=(20, 60, 45))
    d.text((80, 150), "3 қорапша алсаңыз —", font=fm, fill=(40, 40, 40))
    d.text((80, 200), "3 қорапша ТЕГІН", font=fm, fill=(200, 140, 0))
    d.text((80, 250), "+ Түркияға ваучер", font=fm, fill=(40, 40, 40))
    for ch_, y, txt in (("🎁", 196, "3 қорапша ТЕГІН"), ("✈️", 246, "+ Түркияға ваучер")):
        e = emoji_img(ch_, 44); im.alpha_composite(e, (int(80 + d.textlength(txt, font=fm) + 14), y))
    d.rounded_rectangle((80, 330, 30 + cw - 50, 420), 45, fill=(0, 196, 180))
    d.text(((80 + cw - 20) / 2 + 10, 375), "Толық ақпарат", font=fb, fill=(255, 255, 255), anchor="mm")
    e = emoji_img("👇", 56); im.alpha_composite(e, (cw - 90, 347))
    return im
CARD = make_card()

BADGE_IM = badge("3")
# precompute caption sprites
cap_sprites = []
for s, e, y, lines in CAPS:
    sp = []
    for line in lines:
        sc = fit_scale(line)
        ims = [text_img(w, st, sc) for w, st, _ in line]
        tot = sum(i.width for i in ims) - 30 * (len(ims) - 1)
        x = W / 2 - tot / 2; row = []
        for (w, st, ton), im in zip(line, ims):
            # small words sit a bit lower, like "тағы" in the reference
            row.append((im, x + im.width / 2, ton, 18 if st == "small" else 0)); x += im.width - 30
        sp.append((row, max(i.height for i in ims)))
    cap_sprites.append((s, e, y, sp))
outro_sprites = [(text_img(w, "thin"), t) for w, t in OUTRO]
emoji_sprites = [(emoji_img(c, sz), a, b, x, y) for c, a, b, x, y, sz in EMOJI]

# bottom dark gradient for the hook (like reference intro)
grad = np.zeros((H, W, 4), np.uint8)
g = np.clip((np.arange(H) - H * 0.45) / (H * 0.55), 0, 1) ** 1.3 * 200
grad[..., 3] = g[:, None].astype(np.uint8)
GRAD = Image.fromarray(grad, "RGBA")

def hook_line(parts):
    """parts: list of (text_or_emoji, kind) -> one RGBA strip"""
    ims = []
    for txt, kind in parts:
        if kind == "emoji": ims.append(emoji_img(txt, 110))
        else:
            f = ImageFont.truetype(F + ("opentype/inter/Inter-Black.otf" if kind == "big" else "opentype/inter/Inter-SemiBold.otf"),
                                   112 if kind == "big" else 50)
            col = (246, 205, 72) if kind == "big" else (255, 255, 255)
            b = f.getbbox(txt); w, h = b[2] - b[0] + 40, b[3] - b[1] + 40
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); sh = im.copy()
            ImageDraw.Draw(sh).text((20 - b[0], 20 - b[1]), txt, font=f, fill=(0, 0, 0, 255))
            im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)), (0, 4))
            ImageDraw.Draw(im).text((20 - b[0], 20 - b[1]), txt, font=f, fill=col + (255,),
                                    stroke_width=3 if kind == "big" else 0, stroke_fill=(70, 45, 0))
            ims.append(im)
    w = sum(i.width for i in ims) + 10 * (len(ims) - 1); h = max(i.height for i in ims)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0)); x = 0
    for i in ims: out.alpha_composite(i, (x, (h - i.height) // 2)); x += i.width + 10
    if out.width > 1000: out = out.resize((1000, int(out.height * 1000 / out.width)), Image.LANCZOS)
    return out
HOOK1 = hook_line([("DALA SAUMAL · АКЦИЯ", "small")])
HOOK2 = hook_line([("3 алсаңыз — 3 ТЕГІН", "big"), ("🎁", "emoji")])
tg = np.zeros((H, W, 4), np.uint8)
tg[..., 3] = (np.clip(1 - np.arange(H) / (H * 0.30), 0, 1) ** 1.4 * 190)[:, None].astype(np.uint8)
TOPGRAD = Image.fromarray(tg, "RGBA")
HOOK_END = 2.70

def overlay(t, face):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if 0.49 <= t < 2.70:
        ov.alpha_composite(GRAD)
    if t < HOOK_END:
        out_a = min(1, (HOOK_END - t) / 0.15)
        ov.alpha_composite(TOPGRAD)
        pulse = 1 + 0.035 * math.sin(max(0, t - 0.3) * 2 * math.pi * 1.6)  # beat pulse
        paste_scaled(ov, HOOK1, W / 2, 120, ease_pop(t - 0.0, 0.2), out_a)
        paste_scaled(ov, HOOK2, W / 2, 225, ease_pop(t - 0.12, 0.25) * pulse, out_a)
        if t < 0.10:  # white flash on the very first frames
            ov.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - t / 0.10)))))
    for s, e, y0, sp in cap_sprites:
        if not (s <= t < e): continue
        out_a = min(1, (e - t) / 0.08)
        y = y0
        for row, lh in sp:
            for im, cx, ton, dy in row:
                k = ease_pop(t - ton)
                if k > 0: paste_scaled(ov, im, cx, y + lh / 2 + dy, k, out_a)
            y += lh - 12
    for im, a, b, x, y in emoji_sprites:
        if a <= t < b:
            k = ease_pop(t - a, 0.22)
            bob = 8 * math.sin((t - a) * 6)
            fx, fy, fw = face
            ex = fx + 0.62 * fw
            if ex > W - 120: ex = fx - 0.62 * fw
            ex = min(max(ex, 120), W - 120)
            paste_scaled(ov, im, ex, fy - 0.35 * fw + bob, k)
    for a, b, x, y in BADGE3:
        if a <= t < b: paste_scaled(ov, BADGE_IM, x, y, ease_pop(t - a, 0.22))
    if t >= 14.40:
        cur = [im for im, ton in outro_sprites if ton <= t]
        if cur:
            im = cur[-1]; ton = [tt for _, tt in outro_sprites if tt <= t][-1]
            a = min(1, (t - ton) / 0.12)
            paste_scaled(ov, im, W - 90 - im.width / 2, 260, 1.0, a)
        k = ease_pop(t - 14.55, 0.3)
        if k > 0:
            slide = (1 - min(1, (t - 14.55) / 0.3)) * 120
            paste_scaled(ov, CARD, W / 2, 1480 + slide, 0.85 + 0.15 * k)
    return ov

# ---------- pipeline ----------
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", str(FPS), "-i", "-", "-i", SRC, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", "-shortest", OUT], stdin=subprocess.PIPE)
i = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    t = i / FPS
    img, face = frame_crop(Image.frombuffer("RGB", (W, H), buf), min(i, N - 1), t)
    img = img.convert("RGBA")
    img.alpha_composite(overlay(t, face))
    enc.stdin.write(img.convert("RGB").tobytes())
    i += 1
enc.stdin.close(); enc.wait(); print("frames", i)
