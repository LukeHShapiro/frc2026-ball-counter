"""
generate_icon.py — Generate app icon for FRC 2026 Ball Counter desktop app.
Produces assets/icon.ico (multi-size) and assets/icon.png (256px).

Run once:  py generate_icon.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── palette (matches theme.py) ────────────────────────────────────────────────
BG        = (10,  10,  12)          # #0a0a0c
BG_PANEL  = (18,  18,  22)          # #121216
ACCENT    = (99,  102, 241)         # #6366f1  indigo
ACCENT2   = (139, 92,  246)         # #8b5cf6  violet
WHITE     = (255, 255, 255)
DIM       = (120, 120, 140)


def _hex(r, g, b, a=255):
    return (r, g, b, a)


def _circle(draw: ImageDraw.ImageDraw, cx, cy, r, fill, outline=None, width=1):
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=fill,
        outline=outline,
        width=width,
    )


def _rounded_rect(draw, x0, y0, x1, y1, radius, fill, outline=None, width=1):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                            outline=outline, width=width)


def render_icon(size: int) -> Image.Image:
    """Draw the icon at a given pixel size. Returns an RGBA Image."""
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size  # shorthand
    pad = max(1, int(s * 0.04))

    # ── background rounded square ────────────────────────────────────────────
    radius = max(2, int(s * 0.22))
    _rounded_rect(draw, pad, pad, s - pad, s - pad, radius,
                  fill=(*BG_PANEL, 255))

    # ── subtle gradient-ish inner glow (fake via alpha overlay) ─────────────
    glow_r = int(s * 0.38)
    for i in range(glow_r, 0, -1):
        alpha = int(18 * (i / glow_r) ** 2)
        draw.ellipse(
            [s // 2 - i, s // 2 - i, s // 2 + i, s // 2 + i],
            fill=(*ACCENT, alpha),
        )

    # ── outer ring ──────────────────────────────────────────────────────────
    ring_r   = int(s * 0.36)
    ring_w   = max(1, int(s * 0.045))
    draw.ellipse(
        [s // 2 - ring_r, s // 2 - ring_r,
         s // 2 + ring_r, s // 2 + ring_r],
        outline=(*ACCENT, 200), width=ring_w,
    )

    # ── ball (filled circle, center) ────────────────────────────────────────
    ball_r = int(s * 0.20)
    cx, cy = s // 2, s // 2

    # Shadow
    _circle(draw, cx + max(1, int(s * 0.025)),
            cy + max(1, int(s * 0.025)),
            ball_r, fill=(0, 0, 0, 100))

    # Ball fill — indigo → violet gradient simulation via layered circles
    for i in range(ball_r, 0, -1):
        t = i / ball_r                       # 1 at edge, 0 at center
        r = int(ACCENT[0] * t + ACCENT2[0] * (1 - t))
        g = int(ACCENT[1] * t + ACCENT2[1] * (1 - t))
        b = int(ACCENT[2] * t + ACCENT2[2] * (1 - t))
        _circle(draw, cx, cy, i, fill=(r, g, b, 255))

    # Highlight spec
    if size >= 32:
        spec_r = max(1, int(ball_r * 0.28))
        spec_x = cx - int(ball_r * 0.32)
        spec_y = cy - int(ball_r * 0.32)
        _circle(draw, spec_x, spec_y, spec_r,
                fill=(255, 255, 255, 160))

    # ── three orbit dots (trajectory arc) ───────────────────────────────────
    if size >= 24:
        dot_r       = max(1, int(s * 0.040))
        orbit_r     = int(s * 0.43)
        arc_angles  = [-55, -15, 25]          # degrees, upper-right arc
        dot_colors  = [
            (*ACCENT,  255),
            (*ACCENT,  180),
            (*ACCENT,  100),
        ]
        for angle_deg, color in zip(arc_angles, dot_colors):
            rad = math.radians(angle_deg)
            dx  = int(orbit_r * math.cos(rad))
            dy  = int(orbit_r * math.sin(rad))
            _circle(draw, cx + dx, cy + dy, dot_r, fill=color)

    # ── thin accent top-bar inside rounded rect ──────────────────────────────
    if size >= 48:
        bar_h = max(2, int(s * 0.025))
        bar_x0 = pad + radius // 2
        bar_x1 = s - pad - radius // 2
        bar_y  = pad + bar_h + 1
        draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h],
                        fill=(*ACCENT, 180))

    return img


def main():
    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)

    # Render sizes needed for .ico
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [render_icon(s) for s in sizes]

    # Save PNG (largest)
    png_path = out_dir / "icon.png"
    images[-1].save(png_path, format="PNG")
    print(f"[OK] Saved PNG -> {png_path}")

    # Save ICO (all sizes)
    ico_path = out_dir / "icon.ico"
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"[OK] Saved ICO -> {ico_path}")
    print(f"     Sizes embedded: {sizes}")


if __name__ == "__main__":
    main()
