"""The SOAP overview figure, version 2 (manuscript Figure: method).

The failed trajectory and its frozen proxy sit on the left; three lanes read
left to right:

  (1) a thin FIT strip on top, self-contained: the unlabeled reference failures
      pass through the proxy into the stacked matrix R, its SVD yields the
      spectrum, and a contiguous band C is kept;
  (2) the SPECTRAL BASE SCORE row: the failed trajectory's step vectors v_t
      (a grey column per step, t* outlined) are projected onto the band
      (drawn as a plane, so "errors project smaller" is visible as a shorter
      shadow), inverted into S, and the argmax of S drifts past t*;
  (3) the ATTENTION-GUIDED RESCORING row: dependency weights w_{i,t} route the base
      scores of later steps back to the predecessors they depend on; the rescored
      bars are drawn as base (solid) + gamma*B_i (stacked), so the correction is
      visible, and the argmax moves onto t*.

The figure is described once, as a list of primitives on a 46 x 21.6 grid (one
unit = 0.1 in), and rendered twice: with matplotlib to PDF/PNG (the manuscript
copy, STIX fonts to match the ICLR Times style) and with python-pptx to an
editable PowerPoint file whose shapes and text boxes mirror the PDF one to one.
Math is mathtext in the PDF and Unicode with real sub/superscript runs in the
pptx. No data: bars, weights and singular values are illustrative.

    python -m src.analysis.method_figure_v2

Writes manuscript/assets/soap_framework_v2.pdf and, under
artifacts/method_figure/, soap_framework_v2.{pdf,png,pptx}. Leaves
soap_framework.* untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "method_figure"
OUT = {"pdf": REPO / "manuscript" / "assets" / "soap_framework_v2.pdf",
       "pdf2": OUT_DIR / "soap_framework_v2.pdf",
       "png": OUT_DIR / "soap_framework_v2.png",
       "pptx": OUT_DIR / "soap_framework_v2.pptx"}

W, H = 46.0, 21.6                       # canvas, in 0.1 in units

# manuscript palette
PURPLE, PURPLE_DARK = "#807EAF", "#4F4D8A"
ORANGE, ORANGE_DARK, ORANGE_LIGHT = "#F1A484", "#D9722F", "#FBE0D2"
ORANGE_OUTLINE = "#a84d12"
GREY = "#7a7973"                        # stacked / per-step representations
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a887f"
PANEL, PANEL_EDGE, AXIS = "#f7f6f2", "#d6d4ca", "#c3c2b7"
STRIP = "#fcfcfa"
PROXY, PROXY_EDGE, PROXY_INK = "#dfe8f3", "#7f9fc2", "#2d4b6e"
AGENT = {"A": "#bccbe3", "B": "#cfdcc3", "C": "#eadbb4"}

# illustrative trajectory: six steps, decisive error at s3, base argmax drifts to s5
STEPS = [("A", False), ("B", False), ("A", True), ("C", False), ("B", False), ("C", False)]
T, T_STAR, T_BASE = 6, 3, 5
BASE = np.array([0.30, 0.42, 0.62, 0.55, 0.80, 0.70])          # S(s_t)
ADDED = np.array([0.06, 0.10, 0.42, 0.14, 0.00, 0.00])         # gamma * B_i
SCALE = 1.25                                                    # bar value that fills a chart
W_TO_STAR = {4: 0.55, 5: 0.75, 6: 0.95}                        # w_{3,t}, drawn as arcs


def blend(hex_color: str, alpha: float, bg: str = "#ffffff") -> str:
    """Pre-blend a colour over the background: the pptx has no alpha, so both
    renderers use the blended colour and look the same."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(bg[i:i + 2], 16) for i in (1, 3, 5)], float)
    r = alpha * c + (1 - alpha) * b
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in r)


# ----------------------------------------------------------------------------- scene
@dataclass
class Scene:
    ops: list = field(default_factory=list)

    def rect(self, x, y, w, h, fc=None, ec=None, lw=0.5, ls="-", r=0.0, z=3):
        self.ops.append(("rect", dict(x=x, y=y, w=w, h=h, fc=fc, ec=ec, lw=lw, ls=ls, r=r), z))

    def poly(self, pts, fc=None, ec=None, lw=0.5, ls="-", z=3):
        self.ops.append(("poly", dict(pts=[tuple(p) for p in pts], fc=fc, ec=ec, lw=lw, ls=ls), z))

    def line(self, pts, color=INK_2, lw=0.6, ls="-", z=4):
        self.ops.append(("line", dict(pts=[tuple(p) for p in pts], color=color, lw=lw, ls=ls), z))

    def arrow(self, p, q, color=INK_2, lw=0.8, ls="-", rad=0.0, head=True, shrink=0.15, z=5):
        p, q = np.array(p, float), np.array(q, float)
        d = q - p; n = np.linalg.norm(d)
        if n > 0 and shrink:
            p, q = p + d / n * shrink, q - d / n * shrink
        self.ops.append(("arrow", dict(p=tuple(p), q=tuple(q), color=color, lw=lw, ls=ls, rad=rad, head=head), z))

    def text(self, x, y, mpl, plain=None, size=5.8, color=INK_2, ha="center", va="center",
             bold=False, rot=0, psize=None, z=8):
        """`mpl` is mathtext for the PDF; `plain` the pptx markup (defaults to `mpl`);
        `psize` overrides the font size in the pptx only, where Unicode math runs wider."""
        self.ops.append(("text", dict(x=x, y=y, mpl=mpl, plain=plain if plain is not None else mpl,
                                      size=size, psize=psize or size, color=color, ha=ha, va=va,
                                      bold=bold, rot=rot), z))

    def dot(self, x, y, r, fc, ec=None, lw=0.5, z=6):
        self.ops.append(("dot", dict(x=x, y=y, r=r, fc=fc, ec=ec, lw=lw), z))

    def heat(self, x, y, rows, cols, cw, ch, color, alphas, gap=0.14, z=3):
        for r in range(rows):
            for c in range(cols):
                a = alphas(r, c)
                if a is None:
                    continue
                self.rect(x + c * cw, y + (rows - 1 - r) * ch, cw * (1 - gap), ch * (1 - gap),
                          fc=blend(color, a), z=z)

    def chart(self, x0, y0, w, h, base, added=None):
        """Bar chart of per-step scores; `added` stacks gamma*B_i on top of the base."""
        n = len(base); bw = w / n
        self.rect(x0, y0, w, h, fc="#ffffff", ec=AXIS, lw=0.5, z=2)
        tops = []
        for i in range(n):
            bx = x0 + i * bw + bw * 0.22
            hb = h * base[i] / SCALE
            self.rect(bx, y0, bw * 0.56, hb, fc=PURPLE, z=3)
            top = y0 + hb
            if added is not None and added[i] > 0:
                ha = h * added[i] / SCALE
                self.rect(bx, top, bw * 0.56, ha, fc=ORANGE, ec="#ffffff", lw=0.4, z=3)
                top += ha
            tops.append(top)
            self.text(x0 + (i + 0.5) * bw, y0 - 0.7, rf"$s_{i+1}$", f"*s*_{{{i+1}}}", size=5.6, color=MUTED)
        gx = x0 + (T_STAR - 0.5) * bw
        self.line([(gx, y0), (gx, y0 + h)], color=ORANGE_DARK, lw=0.6, ls="--", z=4)
        return [x0 + (i + 0.5) * bw for i in range(n)], tops, gx


def build() -> Scene:
    S = Scene()
    X0, X1 = 12.8, W - 0.4                   # the three panels' horizontal extent

    # ------------------------------------------------------------- left column
    cx, cy, cw_, ch_ = 0.9, 17.5, 4.0, 2.4
    for k in range(3):
        S.rect(cx + (2 - k) * 0.35, cy + k * 0.35, cw_, ch_, fc="#ffffff", ec=AXIS, lw=0.5, r=0.3, z=3 + 0.1 * k)
    agents, pitch, bh0, m = "ABCAB", 0.40, 0.26, 0.32              # bars fill the front card
    y0 = cy + (ch_ - (len(agents) * pitch - (pitch - bh0))) / 2
    for j, a in enumerate(agents):
        S.rect(cx + 0.7 + m, y0 + j * pitch, cw_ - 2 * m, bh0, fc=AGENT[a], z=4)
    S.text(3.6, 21.05, r"reference failures $\mathcal{D}_{\mathrm{fail}}$", "reference failures 𝒟_{fail}", size=5.4, color=INK)

    sx, sw, bh, gap = 0.7, 5.0, 1.55, 0.4
    S.text(sx + sw / 2, 16.5, r"failed trajectory $\tau$", "failed trajectory *τ*", size=5.4, color=INK)
    ytop, ys = 15.9, []
    for t, (a, gold) in enumerate(STEPS):
        y = ytop - bh - t * (bh + gap); ys.append(y + bh / 2)
        S.rect(sx, y, sw, bh, fc=AGENT[a], ec=ORANGE_DARK if gold else AXIS, lw=1.0 if gold else 0.5, r=0.3)
        S.text(sx + 0.4, y + bh / 2, rf"$s_{t+1}$", f"*s*_{{{t+1}}}", size=5.4, color=INK, ha="left")
        S.text(sx + sw - 0.4, y + bh / 2, f"Agent {a}", size=4.9, color=INK_2, ha="right")
    y_star = ys[T_STAR - 1]
    S.text(sx + sw + 0.3, y_star, r"$t^\star$", "*t*^{*}", size=5.8, color=ORANGE_DARK, ha="left")
    S.text(sx + sw / 2, ytop - T * (bh + gap) - 0.6, "✗ task fails", size=5.0, color="#b23a1f")

    px, py0, py1, pw = 7.6, 8.0, 14.6, 2.5
    S.poly([(px, py0), (px, py1), (px + pw, py1 - 1.8), (px + pw, py0 + 1.8)], fc=PROXY, ec=PROXY_EDGE, lw=0.8)
    S.text(px + pw / 2, 11.95, r"$\mathcal{M}$", "ℳ", size=9.5, color=PROXY_INK)
    S.text(px + pw / 2, 10.45, "frozen", size=5.0, color=PROXY_INK)
    S.text(px + pw / 2, 9.5, "proxy", size=5.0, color=PROXY_INK)
    S.text(px + 1.2, 13.35, "❄", size=6.5, color="#5b8ac0")
    S.arrow((cx + 4.5, 18.8), (px, 13.9))
    S.arrow((sx + sw + 0.15, ys[1]), (px, ys[1]))
    S.arrow((px + pw, 13.2), (X0 - 0.1, 19.2))
    S.text(11.1, 17.1, r"$R$", "*R*", size=5.8, color=INK)
    S.arrow((px + pw, 11.6), (X0 - 0.1, 12.6))
    S.text(11.55, 12.85, r"$v_t$", "*v*_{t}", size=5.8, color=INK)
    S.arrow((px + pw, 9.9), (X0 - 0.1, 4.6))
    S.text(12.15, 5.0, "attention", size=4.9, color=INK, ha="right")

    # ------------------------------------------------------------- (1) fit strip
    # three blocks (R + SVD arrow, spectrum, projection histograms) with equal gaps
    S.rect(X0, 17.4, X1 - X0, 3.9, fc=STRIP, ec=PANEL_EDGE, lw=0.6, ls="--", r=0.6, z=0)
    S.text(X0 + 0.6, 20.65, "(1) Fit the spectral band", size=6.4, color=INK, ha="left", bold=True)
    rx, ry, rc = 13.6, 17.85, 0.36
    Ra = np.random.default_rng(3).uniform(0.25, 0.95, (6, 5))
    S.heat(rx, ry, 6, 5, rc, rc, GREY, lambda r, c: Ra[r, c])
    ymid = ry + 3 * rc
    ax0 = rx + 5 * rc + 0.4
    S.arrow((ax0, ymid), (ax0 + 2.4, ymid))
    S.text(ax0 + 1.2, ymid + 0.6, "SVD", size=5.2, color=INK)
    vw, vh, gw, gh = 9.5, 2.5, 8.5, 2.0
    blk1_w = ax0 + 2.4 - rx
    gap1 = (X1 - 0.6 - rx - blk1_w - vw - gw) / 2
    vx, vy = rx + blk1_w + gap1, 18.0
    sv = np.array([1.0, 0.56, 0.42, 0.33, 0.27, 0.22, 0.19, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06])
    band = range(1, 5); bw = vw / len(sv)
    S.line([(vx, vy), (vx + vw, vy)], color=AXIS, lw=0.5, z=2)
    for i, v in enumerate(sv):
        inb = i in band
        S.rect(vx + i * bw + bw * 0.2, vy, bw * 0.6, vh * v, fc=ORANGE_DARK if inb else blend(PURPLE, 0.45))
    S.rect(vx + bw * 1.05, vy - 0.22, 4 * bw - bw * 0.1, vh * 0.62 + 0.22, ec=ORANGE_DARK, lw=0.7, ls="--", z=4)
    band_x = vx + 3 * bw
    S.text(vx + bw * 5.3, vy + vh * 0.62 + 0.1, r"keep band $\mathcal{C}$", "keep band 𝒞", size=5.4,
           color=ORANGE_DARK, ha="left")
    gx0, gy0 = vx + vw + gap1, 18.0
    xs = np.linspace(0, 1, 120)
    for mu, sd, col, fill, lab in ((0.30, 0.11, ORANGE_DARK, blend(ORANGE, 0.35), "errors"),
                                   (0.66, 0.15, PURPLE_DARK, blend(PURPLE, 0.3), "ordinary steps")):
        g = np.exp(-((xs - mu) / sd) ** 2) * (0.95 if mu < 0.5 else 1.0)
        pts = [(gx0 + gw * u, gy0 + gh * v) for u, v in zip(xs, g)]
        S.poly([(gx0, gy0)] + pts + [(gx0 + gw, gy0)], fc=fill, z=2)
        S.line(pts, color=col, lw=0.7, z=3)
        S.text(gx0 + gw * mu, gy0 + gh + 0.4, lab, size=4.8, color=col)
    S.line([(gx0, gy0), (gx0 + gw, gy0)], color=AXIS, lw=0.5, z=2)
    S.text(gx0 + gw / 2, gy0 - 0.45, r"$\pi_{\mathcal{C}}(v_t)$", "*π*_{𝒞}(*v*_{t})", size=4.9, color=MUTED)

    # ------------------------------------------------------------- (2) base score
    A0, A1 = 8.9, 16.9
    S.rect(X0, A0, X1 - X0, A1 - A0, fc=PANEL, ec=PANEL_EDGE, lw=0.7, r=0.6, z=0)
    S.text(X0 + 0.6, A1 - 0.7, "(2) Spectral base score", size=6.4, color=INK, ha="left", bold=True)

    # the failed trajectory's step vectors, one column per step, t* outlined
    hx, hy, hc = 13.7, 10.5, 0.7
    Va = np.random.default_rng(7).uniform(0.25, 0.95, (6, T))
    S.heat(hx, hy, 6, T, hc, hc, PURPLE, lambda r, c: Va[r, c])
    S.rect(hx + (T_STAR - 1) * hc, hy, hc * 0.86, 6 * hc - hc * 0.14, ec=ORANGE_DARK, lw=0.8, z=4)
    S.text(hx + (T_STAR - 0.57) * hc, hy + 6 * hc + 0.38, r"$t^\star$", "*t*^{*}", size=5.0, color=ORANGE_DARK)
    S.arrow((hx + T * hc + 0.3, hy + 3 * hc), (hx + T * hc + 1.5, hy + 3 * hc - 0.5))

    e1 = np.array([1.0, 0.0]); e2 = np.array([0.52, 0.40]); up = np.array([0.0, 1.0])
    O0 = np.array([19.9, 10.0])
    S.poly([O0, O0 + 5.6 * e1, O0 + 5.6 * e1 + 2.4 * e2, O0 + 2.4 * e2], fc=ORANGE_LIGHT, ec=ORANGE_DARK, lw=0.6, z=2)
    corner = O0 + 5.6 * e1 + 2.4 * e2
    S.arrow((band_x, vy - 0.32), (corner[0] - 0.2, corner[1] + 0.5), color=ORANGE_DARK, lw=0.7, ls="--", shrink=0)
    S.text(band_x + 0.75, 15.7, r"$\mathcal{C}$", "𝒞", size=5.8, color=ORANGE_DARK, ha="left")
    Og = O0 + 3.1 * e1 + 1.0 * e2
    sh1 = Og - 2.3 * e1 + 1.2 * e2; v1 = sh1 + 1.2 * up
    S.line([sh1, v1], color=PURPLE_DARK, lw=0.5, ls=":", z=5)
    S.line([Og, sh1], color=PURPLE_DARK, lw=1.6, z=5)
    S.arrow(Og, v1, color=PURPLE_DARK, lw=0.9, shrink=0, z=6)
    S.text(v1[0] - 0.2, v1[1] + 0.5, r"ordinary $v_t$", "ordinary *v*_{t}", size=5.0, color=PURPLE_DARK)
    S.text(O0[0] + 1.5, O0[1] + 0.3, r"$\pi_{\mathcal{C}}$ large", "*π*_{𝒞} large", size=4.5, color=PURPLE_DARK)
    sh2 = Og + 1.2 * e1 + 0.15 * e2; v2 = sh2 + 3.4 * up
    S.line([sh2, v2], color=ORANGE_DARK, lw=0.5, ls=":", z=5)
    S.line([Og, sh2], color=ORANGE_DARK, lw=1.6, z=5)
    S.arrow(Og, v2, color=ORANGE_DARK, lw=0.9, shrink=0, z=6)
    S.text(v2[0] + 0.3, v2[1] + 0.5, r"decisive error $v_{t^\star}$", "decisive error *v*_{t*}", size=5.0,
           color=ORANGE_DARK, ha="right")
    S.text(sh2[0] + 0.3, sh2[1] - 0.05, r"$\pi_{\mathcal{C}}$ small", "*π*_{𝒞} small", size=4.7, color=ORANGE_DARK, ha="left")
    S.dot(Og[0], Og[1], 0.1, INK, z=7)
    S.text(O0[0] + 2.8, O0[1] - 0.42, r"band $\mathcal{C}$", "band 𝒞", size=4.9, color=ORANGE_DARK)

    fx = 27.9
    S.text(fx, 14.0, r"$\pi_{\mathcal{C}}(v_t)=\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\langle v_t,\,V_{:,c}\rangle^{2}$",
           "*π*_{𝒞}(*v*_{t}) = (1/|𝒞|) Σ_{c∈𝒞} ⟨*v*_{t}, *V*_{:,c}⟩^{2}", size=5.6, psize=5.0, color=INK, ha="left")
    S.text(fx, 11.6, r"$S(s_t)=1\,/\,(\pi_{\mathcal{C}}(v_t)+\epsilon)$",
           "*S*(*s*_{t}) = 1 / (*π*_{𝒞}(*v*_{t}) + *ε*)", size=6.0, color=INK, ha="left")

    bx0, by0, bw_, bh_ = 37.2, 10.0, X1 - 0.6 - 37.2, 4.4
    S.arrow((bx0 - 1.6, 12.7), (bx0 - 0.4, 12.7), lw=0.9)
    xs_, tops, gx = S.chart(bx0, by0, bw_, bh_, BASE)
    S.text(gx + 0.25, by0 + bh_ - 0.3, r"$t^\star$", "*t*^{*}", size=5.2, color=ORANGE_DARK, ha="left", va="top")
    S.dot(xs_[T_BASE - 1], tops[T_BASE - 1] + 0.45, 0.26, PURPLE_DARK, ec="#ffffff", lw=0.6)
    S.text(bx0, by0 + bh_ + 0.55, r"$S(s_t)$", "*S*(*s*_{t})", size=5.2, color=INK, ha="left")
    S.text(bx0 + bw_, by0 + bh_ + 0.55, r"$\arg\max S\neq t^\star$", "argmax *S* ≠ *t*^{*}",
           size=4.8, color=PURPLE_DARK, ha="right")

    # ------------------------------------------------------------- (3) rescoring
    B0, B1 = 0.3, 8.0
    S.rect(X0, B0, X1 - X0, B1 - B0, fc=PANEL, ec=PANEL_EDGE, lw=0.7, r=0.6, z=0)
    S.text(X0 + 0.6, B1 - 0.7, "(3) Attention-guided rescoring", size=6.4, color=INK, ha="left", bold=True)

    wx, wy, cs = 13.7, 2.3, 0.7
    strong = {(4, 3): 0.95, (5, 3): 0.8, (6, 3): 0.98, (2, 1): 0.6, (3, 2): 0.5, (5, 4): 0.45, (4, 1): 0.35, (6, 1): 0.4}
    S.heat(wx, wy, T, T, cs, cs, ORANGE_DARK,
           lambda r, c: None if c >= r else 0.85 * strong.get((r + 1, c + 1), 0.28))
    S.rect(wx + 2 * cs, wy, cs * 0.86, 3 * cs, ec=ORANGE_OUTLINE, lw=0.8, z=4)
    S.text(wx + 2.43 * cs, wy - 0.42, r"$i=3$", "*i* = 3", size=4.6, color=ORANGE_DARK)
    S.text(wx + T * cs / 2, wy - 1.15, r"attention weights $w_{i,t}$", "attention weights *w*_{i,t}", size=5.2, color=INK)

    # the trajectory as a chain of steps; arcs carry blame back along w_{i,t}
    dx0, dy0, dsp = 20.4, 5.2, 2.1
    dxs = [dx0 + i * dsp for i in range(T)]
    S.arrow((wx + T * cs + 0.3, dy0), (dx0 - 0.6, dy0))
    S.line([(dxs[0], dy0), (dxs[-1], dy0)], color=AXIS, lw=0.9, z=3)
    for i, (a, gold) in enumerate(STEPS):
        S.dot(dxs[i], dy0, 0.33, AGENT[a], ec=ORANGE_DARK if gold else INK_2, lw=1.0 if gold else 0.5, z=6)
        S.text(dxs[i], dy0 - 0.78, rf"$s_{i+1}$", f"*s*_{{{i+1}}}", size=4.9, color=MUTED)
    for (t, i), w in sorted(strong.items(), key=lambda kv: kv[1]):
        S.arrow((dxs[t - 1], dy0 + 0.33), (dxs[i - 1] + 0.12, dy0 + 0.33), color=blend(ORANGE_DARK, 0.3 + 0.65 * w),
                lw=0.4 + 1.2 * w, rad=0.42, shrink=0, z=5)
    S.text(dxs[T_STAR - 1] + 0.3, dy0 - 0.78, r"$=t^\star$", "= *t*^{*}", size=4.9, color=ORANGE_DARK, ha="left")

    fx = 20.2
    S.text(fx, 3.0, r"$\widetilde{S}(s_i)=S(s_i)+\gamma\,B_i$", "*S̃*(*s*_{i}) = *S*(*s*_{i}) + *γB*_{i}", size=6.4, color=INK, ha="left")
    S.text(fx, 1.45, r"$B_i=\sum_{t>i} w_{i,t}\,S(s_t)\,/\,\sum_{t>i} w_{i,t}$",
           "*B*_{i} = Σ_{t>i} *w*_{i,t} *S*(*s*_{t}) / Σ_{t>i} *w*_{i,t}", size=5.4, psize=5.2, color=INK, ha="left")

    rx0, ry0 = bx0, 1.2
    S.arrow((dxs[-1] + 0.7, dy0), (rx0 - 0.4, dy0), lw=0.9)
    xs_, tops, gx = S.chart(rx0, ry0, bw_, bh_, BASE, ADDED)
    for t, w in W_TO_STAR.items():
        S.arrow((xs_[t - 1], tops[t - 1] + 0.15), (xs_[T_STAR - 1] + 0.28, tops[T_STAR - 1] + 0.1),
                color=blend(ORANGE_DARK, 0.35 + 0.6 * w), lw=0.5 + 1.4 * w, rad=0.28 + 0.06 * (t - T_STAR), shrink=0)
    S.text(rx0 + bw_, ry0 + bh_ + 1.35, r"$w_{3,t}\,S(s_t)$", "*w*_{3,t} *S*(*s*_{t})", size=4.9, color=ORANGE_DARK, ha="right")
    tri = (xs_[T_STAR - 1], tops[T_STAR - 1] + 0.5)
    S.poly([(tri[0] - 0.28, tri[1] - 0.22), (tri[0] + 0.28, tri[1] - 0.22), (tri[0], tri[1] + 0.28)],
           fc=ORANGE_DARK, ec="#ffffff", lw=0.4, z=6)
    S.text(tri[0], ry0 + bh_ + 0.55, r"$\hat t=t^\star$", "*t̂* = *t*^{*}", size=5.2, color=ORANGE_DARK)
    lx, ly = rx0 + 0.45, ry0 + bh_ - 0.6
    S.rect(lx, ly - 0.25, 0.5, 0.5, fc=PURPLE, z=4)
    S.text(lx + 0.7, ly, r"$S(s_i)$", "*S*(*s*_{i})", size=4.7, ha="left")
    S.rect(lx, ly - 0.9, 0.5, 0.5, fc=ORANGE, z=4)
    S.text(lx + 0.7, ly - 0.65, r"$\gamma B_i$", "*γB*_{i}", size=4.7, ha="left")
    S.text(rx0, ry0 + bh_ + 0.55, r"$\widetilde{S}(s_i)$", "*S̃*(*s*_{i})", size=5.2, color=INK, ha="left")

    ax_x = bx0 + 1.5
    S.arrow((ax_x, by0 - 0.8), (ax_x, ry0 + bh_ + 1.45), lw=1.1, shrink=0)
    S.text(ax_x - 0.45, 8.4, r"$+\,\gamma B_i$", "+ *γB*_{i}", size=5.6, color=INK, ha="right")
    return S


# ------------------------------------------------------------------- matplotlib
def render_mpl(S: Scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

    DASH = {"-": "-", "--": (0, (3, 2)), ":": (0, (1.5, 1.2))}
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": ["STIXGeneral", "DejaVu Sans"],
          "mathtext.fontset": "stix", "font.size": 7}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(W / 10, H / 10), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal"); ax.axis("off")
        for kind, o, z in S.ops:
            if kind == "rect":
                fc = o["fc"] or "none"; ec = o["ec"] or "none"
                if o["r"]:
                    ax.add_patch(FancyBboxPatch((o["x"], o["y"]), o["w"], o["h"], fc=fc, ec=ec, lw=o["lw"],
                                                ls=DASH[o["ls"]], zorder=z,
                                                boxstyle=f"round,pad=0,rounding_size={o['r']}"))
                else:
                    ax.add_patch(Rectangle((o["x"], o["y"]), o["w"], o["h"], fc=fc, ec=ec, lw=o["lw"],
                                           ls=DASH[o["ls"]], zorder=z))
            elif kind == "poly":
                ax.add_patch(Polygon(o["pts"], closed=True, fc=o["fc"] or "none", ec=o["ec"] or "none",
                                     lw=o["lw"], ls=DASH[o["ls"]], zorder=z))
            elif kind == "line":
                xs, ys = zip(*o["pts"])
                ax.plot(xs, ys, color=o["color"], lw=o["lw"], ls=DASH[o["ls"]], zorder=z, solid_capstyle="butt")
            elif kind == "arrow":
                ax.add_patch(FancyArrowPatch(o["p"], o["q"], arrowstyle="-|>" if o["head"] else "-",
                                             mutation_scale=6, lw=o["lw"], color=o["color"], zorder=z,
                                             connectionstyle=f"arc3,rad={o['rad']}", linestyle=DASH[o["ls"]],
                                             shrinkA=0, shrinkB=0))
            elif kind == "dot":
                ax.add_patch(Circle((o["x"], o["y"]), o["r"], fc=o["fc"], ec=o["ec"] or "none", lw=o["lw"], zorder=z))
            elif kind == "text":
                ax.text(o["x"], o["y"], o["mpl"], fontsize=o["size"], color=o["color"], ha=o["ha"], va=o["va"],
                        fontweight="bold" if o["bold"] else "normal", rotation=o["rot"],
                        rotation_mode="anchor", zorder=z)
        for key in ("pdf", "pdf2"):            # exact canvas size, so every copy matches the pptx
            fig.savefig(OUT[key], facecolor="white")
        fig.savefig(OUT["png"], dpi=300, facecolor="white")
        plt.close(fig)


# ------------------------------------------------------------------------ pptx
_RUN = re.compile(r"\*|_\{[^}]*\}|\^\{[^}]*\}")


def _runs(plain: str):
    """Split the plain markup into (text, italic, baseline) runs.
    `*...*` toggles italic, `_{..}` is a subscript, `^{..}` a superscript."""
    out, italic, pos = [], False, 0
    for m in _RUN.finditer(plain):
        if m.start() > pos:
            out.append((plain[pos:m.start()], italic, 0))
        tok = m.group(0)
        if tok == "*":
            italic = not italic
        elif tok.startswith("_"):   # short subscripts are variables (italic), words are not
            out.append((tok[2:-1], len(tok) - 3 <= 3, -25000))
        else:
            out.append((tok[2:-1], italic, 30000))
        pos = m.end()
    if pos < len(plain):
        out.append((plain[pos:], italic, 0))
    return out


def render_pptx(S: Scene):
    from lxml import etree
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt

    FONT = "Times New Roman"
    DASH = {"-": None, "--": MSO_LINE_DASH_STYLE.DASH, ":": MSO_LINE_DASH_STYLE.ROUND_DOT}

    def X(x): return Inches(x / 10)
    def Y(y): return Inches((H - y) / 10)
    def rgb(h): return RGBColor.from_string(h.lstrip("#").upper())

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W / 10), Inches(H / 10)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shapes = slide.shapes

    def style(shape, fc, ec, lw, ls, head=False):
        if fc:
            shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fc)
        elif hasattr(shape, "fill"):
            shape.fill.background()
        ln = shape.line
        if ec:
            ln.color.rgb = rgb(ec); ln.width = Pt(lw)
            if DASH[ls] is not None:
                ln.dash_style = DASH[ls]
        else:
            ln.fill.background()
        if head:
            tail = etree.SubElement(ln._get_or_add_ln(), qn("a:tailEnd"))
            tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
        st = shape._element.find(qn("p:style"))   # theme style: LibreOffice paints its shadow
        if st is not None:
            shape._element.remove(st)

    def freeform(pts, closed):
        fb = shapes.build_freeform(X(pts[0][0]), Y(pts[0][1]), scale=1.0)
        fb.add_line_segments([(X(x), Y(y)) for x, y in pts[1:]], close=closed)
        return fb.convert_to_shape()

    for kind, o, z in sorted(S.ops, key=lambda op: op[2]):
        if kind == "rect":
            if o["r"]:
                sh = shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(o["x"]), Y(o["y"] + o["h"]), X(o["w"]), X(o["h"]))
                sh.adjustments[0] = min(0.5, o["r"] / min(o["w"], o["h"]))
            else:
                sh = shapes.add_shape(MSO_SHAPE.RECTANGLE, X(o["x"]), Y(o["y"] + o["h"]), X(o["w"]), X(o["h"]))
            style(sh, o["fc"], o["ec"], o["lw"], o["ls"])
        elif kind == "poly":
            sh = freeform(o["pts"], True)
            style(sh, o["fc"], o["ec"], o["lw"], o["ls"])
        elif kind == "line":
            sh = freeform(o["pts"], False)
            style(sh, None, o["color"], o["lw"], o["ls"])
        elif kind == "arrow":
            p, q = o["p"], o["q"]
            if abs(o["rad"]) < 1e-6:
                sh = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, X(p[0]), Y(p[1]), X(q[0]), Y(q[1]))
            else:   # the same quadratic bezier matplotlib's arc3 draws, sampled as a polyline
                p, q = np.array(p), np.array(q); m = (p + q) / 2; d = q - p
                c = np.array([m[0] + o["rad"] * d[1], m[1] - o["rad"] * d[0]])
                ts = np.linspace(0, 1, 24)[:, None]
                pts = (1 - ts) ** 2 * p + 2 * (1 - ts) * ts * c + ts ** 2 * q
                sh = freeform([tuple(v) for v in pts], False)
            style(sh, None, o["color"], o["lw"], o["ls"], head=o["head"])
        elif kind == "dot":
            r = o["r"]
            sh = shapes.add_shape(MSO_SHAPE.OVAL, X(o["x"] - r), Y(o["y"] + r), X(2 * r), X(2 * r))
            style(sh, o["fc"], o["ec"], o["lw"], "-")
        elif kind == "text":
            runs = _runs(o["plain"])
            nchar = sum(len(t) for t, _, _ in runs)
            size = o["psize"]
            tw = nchar * size * 0.46 / 7.2 + 0.5               # width estimate, canvas units
            th = size * 1.35 / 7.2
            x, y = o["x"], o["y"]
            if o["rot"]:
                left, top = x - tw / 2, y + th / 2
            else:
                left = {"left": x, "center": x - tw / 2, "right": x - tw}[o["ha"]]
                top = {"top": y, "center": y + th / 2, "bottom": y + th}[o["va"]]
            tb = shapes.add_textbox(X(left), Y(top), X(tw), X(th))
            tf = tb.text_frame; tf.word_wrap = False
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
            tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "center": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[o["va"]]
            para = tf.paragraphs[0]
            para.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[o["ha"]]
            for text, italic, baseline in runs:
                run = para.add_run(); run.text = text
                f = run.font; f.name = FONT; f.size = Pt(size); f.bold = o["bold"]; f.italic = italic
                f.color.rgb = rgb(o["color"])
                if baseline:
                    run._r.get_or_add_rPr().set("baseline", str(baseline))
            if o["rot"]:
                tb.rotation = (360 - o["rot"]) % 360
    prs.save(OUT["pptx"])


def main():
    for p in OUT.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    S = build()
    render_mpl(S)
    render_pptx(S)
    for p in OUT.values():
        print(p.relative_to(REPO))


if __name__ == "__main__":
    main()
