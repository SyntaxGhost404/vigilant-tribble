"""Procedural baroque engraving: vine scrolls (rinceaux), palmettes, a
grotesque mask and border lines, rasterised into height maps."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class Canvas:
    """Drawing surface addressed in millimetres (x right, y up)."""

    def __init__(self, w_mm, h_mm, px_per_mm, ss=2, x0=0.0, y0=0.0,
                 sy=None):
        self.sx = px_per_mm * ss
        self.sy = (sy if sy is not None else px_per_mm) * ss
        self.ss = ss
        self.x0, self.y0 = x0, y0
        self.W = int(round(w_mm * self.sx))
        self.H = int(round(h_mm * self.sy))
        self.img = Image.new('L', (self.W, self.H), 0)
        self.d = ImageDraw.Draw(self.img)

    def P(self, x, y):
        return ((x - self.x0) * self.sx, self.H - (y - self.y0) * self.sy)

    def stroke(self, pts, w, value=255):
        pts = np.asarray(pts, float)
        if np.isscalar(w):
            w = np.full(len(pts), float(w))
        w = np.asarray(w, float)
        q = [self.P(*p) for p in pts]
        for i in range(len(q) - 1):
            (x0, y0), (x1, y1) = q[i], q[i + 1]
            dx, dy = x1 - x0, y1 - y0
            L = max(np.hypot(dx, dy), 1e-6)
            nx, ny = -dy / L, dx / L
            a = 0.5 * w[i] * self.sx
            b = 0.5 * w[i + 1] * self.sx
            self.d.polygon([(x0 + nx * a, y0 + ny * a), (x1 + nx * b, y1 + ny * b),
                            (x1 - nx * b, y1 - ny * b), (x0 - nx * a, y0 - ny * a)],
                           fill=value)
        for (x, y), ww in zip(q, w):
            r = 0.5 * ww * self.sx
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=value)

    def dot(self, x, y, r, value=255):
        X, Y = self.P(x, y)
        self.d.ellipse([X - r * self.sx, Y - r * self.sy, X + r * self.sx,
                        Y + r * self.sy], fill=value)

    def poly(self, pts, value=255):
        self.d.polygon([self.P(*p) for p in pts], fill=value)

    def height(self, blur_mm, out_size=None):
        im = self.img
        if blur_mm > 0:
            im = im.filter(ImageFilter.GaussianBlur(blur_mm * self.sx))
        if out_size is None:
            out_size = (self.W // self.ss, self.H // self.ss)
        im = im.resize(out_size, Image.LANCZOS)
        return np.asarray(im, np.float32) / 255.0


# ------------------------------------------------------------- motifs -----
def spiral(cx, cy, r0, turns, a0, sense=1, shrink=0.72, n=40):
    t = np.linspace(0, 1, n)
    r = r0 * (1 - shrink * t)
    a = a0 + sense * 2 * np.pi * turns * t
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def leaf(base, direction, length, width, bend=0.25, n=9):
    """Teardrop leaf as (centreline points, widths)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    t = np.linspace(0, 1, n)
    pts = base + np.outer(t * length, d) + np.outer(
        bend * length * np.sin(np.pi * t) * 0.35, nrm)
    w = width * np.sin(np.pi * np.clip(t * 0.92 + 0.04, 0, 1)) ** 0.8
    return pts, w


def tendril(S, T, N, R0, turns=1.2, shrink=0.78, n=48):
    """Curve leaving point S along tangent T and curling toward normal N."""
    S, T, N = (np.asarray(a, float) for a in (S, T, N))
    T = T / np.linalg.norm(T)
    N = N / np.linalg.norm(N)
    C = S + R0 * N
    a0 = np.arctan2(S[1] - C[1], S[0] - C[0])
    # direction of rotation so that the start tangent matches T
    d = S - C
    sense = 1.0 if (d[0] * T[1] - d[1] * T[0]) > 0 else -1.0
    t = np.linspace(0, 1, n)
    r = R0 * (1 - shrink * t)
    a = a0 + sense * 2 * np.pi * turns * t
    return np.stack([C[0] + r * np.cos(a), C[1] + r * np.sin(a)], 1)


def rinceau(cv, x0, x1, yc, amp, lam, stem_w, branch_w=None, phase=0.0,
            leaves=True, seed=1, half_h=5.0):
    """Running vine scroll between x0..x1 centred at yc: a wavy stem with a
    curling tendril in every bay and small leaves at the branch points."""
    rng = np.random.default_rng(seed)
    branch_w = branch_w or stem_w * 0.8
    def stem(x):
        return yc + amp * np.sin(2 * np.pi * (x - x0) / lam + phase)
    def dstem(x):
        return amp * 2 * np.pi / lam * np.cos(2 * np.pi * (x - x0) / lam +
                                             phase)
    xs = np.linspace(x0, x1, int((x1 - x0) / lam * 48) + 2)
    cv.stroke(np.stack([xs, stem(xs)], 1), stem_w)
    # extrema: phase + 2pi(x-x0)/lam = pi/2 + k pi
    k = np.arange(-2, int((x1 - x0) / lam * 2) + 3)
    xe = x0 + (np.pi / 2 + k * np.pi - phase) / (2 * np.pi) * lam
    for i, x in enumerate(xe):
        if x < x0 + 0.3 * lam or x > x1 - 0.35 * lam:
            continue
        top = stem(x) > yc            # crest -> curl into the lower bay
        nrm = np.array([0.0, -1.0 if top else 1.0])
        xs0 = x + 0.04 * lam
        S = np.array([xs0, stem(xs0)])
        T = np.array([1.0, dstem(xs0)])
        room = half_h - 0.6 + amp     # distance from stem extreme to border
        R0 = min(0.22 * lam, 0.5 * room * 0.9)
        sp = tendril(S, T, nrm, R0, turns=1.25, shrink=0.8)
        w = np.linspace(branch_w, branch_w * 0.5, len(sp))
        cv.stroke(sp, w)
        cv.dot(*sp[-1], branch_w * 0.85)
        if leaves:
            # leaf on the outside of the stem at the extreme
            dirn = np.array([-0.55, 0.85 if top else -0.85])
            pts, ww = leaf(np.array([x - 0.06 * lam, stem(x - 0.06 * lam)]),
                           dirn, lam * (0.17 + 0.03 * rng.random()),
                           stem_w * 1.7, bend=0.5)
            cv.stroke(pts, ww)
            # small leaf on the tendril's outer flank
            j = len(sp) // 4
            tdir = sp[j + 1] - sp[j]
            out = np.array([tdir[1], -tdir[0]])
            if np.dot(out, sp[j] - (S + R0 * nrm)) < 0:
                out = -out
            pts, ww = leaf(sp[j], out + tdir / (np.linalg.norm(tdir) + 1e-9),
                           lam * 0.12, stem_w * 1.3, bend=0.3)
            cv.stroke(pts, ww)


def lyre(cv, x, yc, half_h, size, lw, forward=1):
    """Mirrored pair of C-scrolls back to back on the centre line."""
    for s in (-1, 1):
        S = np.array([x, yc + s * 0.5])
        T = np.array([forward * 0.55, s * 1.0])
        N = np.array([forward * 1.0, -s * 0.35])
        sp = tendril(S, T, N, size * 0.42, turns=1.15, shrink=0.75)
        cv.stroke(sp, np.linspace(lw * 1.25, lw * 0.6, len(sp)))
        cv.dot(*sp[-1], lw * 1.0)


def vase(cv, x, yc, size, lw, forward=1):
    """Small baluster/vase on the centre line with a leaf crown."""
    body = []
    for t in np.linspace(0, 1, 9):
        r = size * (0.18 + 0.32 * np.sin(np.pi * t) ** 0.8)
        body.append((x + forward * t * size * 1.1, r))
    top = [(px_, yc + r) for px_, r in body]
    bot = [(px_, yc - r) for px_, r in body][::-1]
    cv.poly(top + bot)
    for a in (-0.6, 0.0, 0.6):
        pts, ww = leaf(np.array([x + forward * size * 1.1, yc]),
                       (forward * 1.0, a), size * 0.9, lw * 2.0, bend=-a)
        cv.stroke(pts, ww)


def palmette(cv, cx, cy, w, h, up=1, n=5, lw=0.5):
    base = np.array([cx, cy])
    for i in range(n):
        a = (i - (n - 1) / 2) / max((n - 1) / 2, 1)
        dirn = np.array([a * w / h * 0.9, up * 1.0])
        pts, ww = leaf(base, dirn, h * (1.0 - 0.25 * abs(a)), lw * 2.2,
                       bend=-0.5 * a)
        cv.stroke(pts, ww)
    cv.dot(cx, cy, lw * 1.4)


def mask(cv, cx, cy, s, lw):
    """Stylised grotesque face, s = half height (mm)."""
    # brow scrolls
    for sd in (-1, 1):
        cv.stroke(spiral(cx + sd * 0.42 * s, cy + 0.38 * s, 0.32 * s, 0.9,
                         np.pi / 2, sense=-sd, shrink=0.6), lw)
        cv.dot(cx + sd * 0.36 * s, cy + 0.12 * s, 0.13 * s)       # eyes
        # cheek / moustache scrolls
        cv.stroke(spiral(cx + sd * 0.38 * s, cy - 0.32 * s, 0.26 * s, 0.85,
                         np.pi / 2 + sd * 0.4, sense=sd, shrink=0.6), lw)
    cv.stroke([(cx, cy + 0.2 * s), (cx, cy - 0.25 * s)], [lw * 1.6, lw * 2.4])
    cv.stroke([(cx - 0.3 * s, cy - 0.55 * s), (cx, cy - 0.68 * s),
               (cx + 0.3 * s, cy - 0.55 * s)], lw)
    # beard of leaves
    for a in (-0.5, -0.2, 0.2, 0.5):
        pts, ww = leaf(np.array([cx + a * s * 0.6, cy - 0.7 * s]),
                       (a * 0.6, -1.0), 0.45 * s, lw * 1.8, bend=a)
        cv.stroke(pts, ww)
    # crown leaves
    for a in (-0.6, -0.2, 0.2, 0.6):
        pts, ww = leaf(np.array([cx + a * s * 0.5, cy + 0.68 * s]),
                       (a * 0.7, 1.0), 0.42 * s, lw * 1.8, bend=-a)
        cv.stroke(pts, ww)


def border(cv, x0, x1, y0, y1, lw, double=False):
    for yy in (y0, y1):
        cv.stroke([(x0, yy), (x1, yy)], lw)
    if double:
        g = lw * 2.4
        for yy in (y0 + g, y1 - g):
            cv.stroke([(x0, yy), (x1, yy)], lw * 0.55)


def band(cv, x, y0, y1, lw, n=3, gap=0.9):
    for i in range(n):
        cv.stroke([(x + i * gap, y0), (x + i * gap, y1)], lw)


# ------------------------------------------------------------- textures ---
def barrel_height(W=4096, H=2048, seed=3):
    """Height map in barrel UV space: u -> X (0..268 mm), v -> angle.
    Returns float array (H, W), row 0 = v=1."""
    L = 268.0
    # draw in (X mm, v*C) where C is a nominal circumference for the flats
    C = 103.7
    cv = Canvas(L, C, W / L, ss=2, sy=H / C)
    flat_w = 12.26
    # ---- top flat (v centre 0.25): mirrored arabesque + grotesque mask
    yc = 0.25 * C
    y0, y1 = yc - flat_w / 2 + 0.6, yc + flat_w / 2 - 0.6
    border(cv, 6.0, 192.0, y0, y1, 0.32)
    cv.stroke([(6.0, y0), (6.0, y1)], 0.32)
    cv.stroke([(192.0, y0), (192.0, y1)], 0.32)
    band(cv, 1.4, y0 - 0.3, y1 + 0.3, 0.45, n=3, gap=1.1)
    band(cv, 188.0, y0 - 0.3, y1 + 0.3, 0.45, n=3, gap=1.1)
    mask(cv, 101.0, yc, 4.9, 0.36)
    # mirrored double vine scroll running out from the mask to both ends
    cv.stroke([(8.0, yc), (93.0, yc)], 0.3)
    cv.stroke([(109.0, yc), (190.0, yc)], 0.3)
    for (a, b, ph) in ((8.5, 93.5, 0.0), (108.5, 190.0, np.pi)):
        rinceau(cv, a, b, yc + 2.75, 0.9, 10.4, 0.42, 0.38, phase=ph,
                seed=seed + 1, half_h=2.5)
        # mirror image for the lower half
        tmp = Canvas(L, C, W / L, ss=2, sy=H / C)
        rinceau(tmp, a, b, yc + 2.75, 0.9, 10.4, 0.42, 0.38, phase=ph,
                seed=seed + 1, half_h=2.5)
        mir = tmp.img.transpose(Image.FLIP_TOP_BOTTOM)
        # flip about the flat's centre line (yc) rather than the canvas
        shift = int(round((2 * yc - C) * cv.sy))
        mir = mir.transform(mir.size, Image.AFFINE, (1, 0, 0, 0, 1, shift))
        cv.img.paste(255, mask=mir)
    # ---- upper diagonal flats: running vine scrolls
    for vc, ph in [(0.125, 0.0), (0.375, np.pi)]:
        yc = vc * C
        y0, y1 = yc - flat_w / 2 + 0.6, yc + flat_w / 2 - 0.6
        border(cv, 6.0, 192.0, y0, y1, 0.32)
        cv.stroke([(6.0, y0), (6.0, y1)], 0.32)
        cv.stroke([(192.0, y0), (192.0, y1)], 0.32)
        rinceau(cv, 7.5, 190.5, yc, 2.0, 15.5, 0.62, 0.52, phase=ph,
                seed=seed, half_h=flat_w / 2 - 0.6)
        band(cv, 1.4, y0 - 0.3, y1 + 0.3, 0.45, n=3, gap=1.1)
        band(cv, 188.0, y0 - 0.3, y1 + 0.3, 0.45, n=3, gap=1.1)
    # ---- side flats: simple double border + small scrolls at the ends
    for vc in (0.0, 0.5, 1.0):
        yc = vc * C
        y0, y1 = yc - flat_w / 2 + 0.6, yc + flat_w / 2 - 0.6
        border(cv, 6.0, 192.0, y0, y1, 0.3)
        band(cv, 1.4, y0, y1, 0.45, n=3, gap=1.1)
    # ---- bell: volutes at the base on each side + acanthus collar
    Cb = C            # same nominal mapping (angle fraction)
    for vc in (0.0, 0.5, 1.0):
        yc = vc * Cb
        for sd in (-1, 1):
            cv.stroke(spiral(208.6, yc + sd * 3.6, 3.4, 1.35, -sd * np.pi / 2,
                             sense=sd, shrink=0.8, n=60), 0.75)
            cv.dot(208.6, yc + sd * 3.6, 0.9)
    for k in range(16):
        vc = k / 16.0
        p, w = leaf(np.array([203.6, vc * Cb]), (1.0, 0.0), 7.0, 2.4,
                    bend=0.2)
        cv.stroke(p, w * 0.55)
    return cv


# ------------------------------------------------------- final texture maps -
def _blur(a, sigma_px):
    from scipy import ndimage
    return ndimage.gaussian_filter(np.asarray(a, np.float32), sigma_px)


def barrel_maps(W=4096, H=2048, seed=3):
    """Returns (height, grime) arrays, row 0 = v=1 (image order)."""
    L, C = 268.0, 103.7
    flat_w = 12.26
    orn = barrel_height(W, H, seed).height(0.04)          # ornaments+borders
    pan = Canvas(L, C, W / L, ss=2, sy=H / C)
    for vc in (0.0, 0.125, 0.25, 0.375, 0.5, 1.0):
        yc = vc * C
        y0, y1 = yc - flat_w / 2 + 0.6, yc + flat_w / 2 - 0.6
        if vc in (0.0, 0.5, 1.0):
            continue                     # side flats: plain, border only
        pan.poly([(6.0, y0), (192.0, y0), (192.0, y1), (6.0, y1)])
    P_ = pan.height(0.05)
    # bell ornaments live on a plain (unrecessed) surface
    u = np.arange(W)[None, :] / W * L
    bell = (u > 199.0).astype(np.float32)
    h = 0.55 * (1.0 - P_)
    h = np.maximum(h, 0.78 * orn * (1 - bell) + 0.92 * orn * bell)
    grime = np.clip(P_ * (1.0 - orn * 1.4), 0, 1)
    # dark halo round the bell ornaments
    halo = np.clip(_blur(orn * bell, 6.0) * 2.5 - orn * bell * 2.0, 0, 1)
    grime = np.clip(grime + 0.5 * halo, 0, 1)
    return h, grime


def lock_maps(W=4096, H=2458):
    """Side-projection maps for the lock work (X -75..125, Z -40..80 mm).
    Returns (height, grime, inlay), row 0 = top (Z max)."""
    import gb_profiles as P
    import gb_sdf as S
    X0, X1, Z0, Z1 = -75.0, 125.0, -40.0, 80.0
    xs = X0 + (np.arange(W) + 0.5) / W * (X1 - X0)
    zs = Z1 - (np.arange(H) + 0.5) / H * (Z1 - Z0)
    XX, ZZ = np.meshgrid(xs, zs)
    ppm = W / (X1 - X0)
    lines = np.zeros((H, W), np.float32)

    def contour(poly, inset, width):
        d = S.polygon_sdf2(XX, ZZ, poly)
        return np.clip(1.0 - np.abs(d + inset) / (0.5 * width), 0, 1), d

    plate = P.to_mm(P.catmull_rom(P.LOCK_PLATE, 6, closed=True))
    b1, dplate = contour(plate, 1.25, 0.36)
    b2, _ = contour(plate, 1.95, 0.2)
    lines = np.maximum(lines, np.maximum(b1, b2))
    # scrolls along the plate, clipped inside the double border
    cv = Canvas(X1 - X0, Z1 - Z0, ppm, ss=2, x0=X0, y0=Z0)
    rinceau(cv, -66.0, 102.0, -15.4, 3.4, 20.5, 0.62, 0.5, seed=11,
            half_h=10.8)
    scr = cv.height(0.03, out_size=(W, H))
    # cock body, frizzen spring and pan sit over the plate in this
    # projection: keep the plate decoration off them
    body = P.to_mm(P.catmull_rom(P.COCK_BODY, 5, closed=True))
    cb, dbody = contour(body, 1.1, 0.32)
    sp_pts = np.array([(54.0, -21.4), (66.0, -21.6), (80.0, -20.6),
                       (90.5, -18.6), (95.0, -15.6), (93.6, -12.4),
                       (87.5, -10.4), (80.5, -9.2), (75.5, -8.4)])
    dsp, _, _ = S.polyline_dist2(XX, ZZ, sp_pts)
    sx, sz = P.to_mm([P.FRIZZEN_SPRING_SCREW[:2]])[0]
    spring = (dsp < 4.0) | (np.hypot(XX - sx, ZZ - sz) < 5.2)
    pan = (XX > 22.5) & (XX < 71.0) & (ZZ > -2.0)
    keep = (dbody > 0.8) & ~spring & ~pan
    lines = lines * keep
    lines = np.maximum(lines, scr * (dplate < -2.4) * keep)
    lines = np.maximum(lines, cb)
    cv = Canvas(X1 - X0, Z1 - Z0, ppm, ss=2, x0=X0, y0=Z0)
    spine = P.to_mm(P.catmull_rom(P.COCK_NECK_SPINE, 6))
    cv.stroke(spine, 0.3)
    bx, bz = P.to_mm([P.COCK_PIVOT[:2]])[0]
    for k in range(10):
        a = k / 10 * 2 * np.pi
        p0 = (bx + 9.0 * np.cos(a), bz + 9.0 * np.sin(a))
        p1 = (bx + 10.8 * np.cos(a + 0.25), bz + 10.8 * np.sin(a + 0.25))
        cv.stroke([p0, p1], 0.26)
    for (cx_, cz_, r_, a_, s_) in ((bx - 6.5, bz + 9.0, 2.6, 0.0, 1),
                                   (bx + 7.5, bz + 7.0, 2.4, 2.0, -1),
                                   (bx + 6.0, bz - 9.5, 2.4, 3.6, 1),
                                   (bx - 7.0, bz - 8.0, 2.2, 5.0, -1)):
        cv.stroke(spiral(cx_, cz_, r_, 1.2, a_, sense=s_, shrink=0.7), 0.34)
    # pan side lines and frizzen spring centre line
    cv.stroke([(25.0, 0.6), (69.0, 0.6)], 0.28)
    cv.stroke([(26.0, 5.4), (68.0, 4.6)], 0.28)
    cv.stroke([(56.0, -21.4), (80.0, -20.6), (90.5, -18.6), (95.0, -15.6),
               (93.6, -12.4), (87.5, -10.4), (77.0, -8.6)], 0.26)
    lines = np.maximum(lines, cv.height(0.03, out_size=(W, H)))
    # gold inlay: feathered leaf below the pan and a round boss
    ci = Canvas(X1 - X0, Z1 - Z0, ppm, ss=2, x0=X0, y0=Z0)
    pts, w = leaf(np.array([24.5, -8.0]), (1.0, -0.62), 26.0, 5.4, bend=0.5,
                  n=24)
    ci.stroke(pts, w)
    ci.dot(45.3, -20.1, 2.6)
    inlay = ci.height(0.04, out_size=(W, H))
    # the leaf's engraved veins
    cv = Canvas(X1 - X0, Z1 - Z0, ppm, ss=2, x0=X0, y0=Z0)
    cv.stroke(pts[2:-3], 0.22)
    for k in range(3, len(pts) - 4, 3):
        d = pts[k + 1] - pts[k]
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        cv.stroke([pts[k], pts[k] + 2.0 * n + 1.2 * d / np.linalg.norm(d)],
                  0.16)
        cv.stroke([pts[k], pts[k] - 2.0 * n + 1.2 * d / np.linalg.norm(d)],
                  0.16)
    lines = np.maximum(lines, cv.height(0.025, out_size=(W, H)))
    # relief style like the barrel: scrolls/borders raised, the ground
    # inside the decorated fields recessed and matted
    field = np.clip((-(dplate + 2.4)) / 0.25, 0, 1) * keep
    height = 0.55 * (1.0 - field) + 0.12 * field
    height = np.maximum(height, 0.82 * lines)
    grime = np.clip(field * (1.0 - 1.3 * lines), 0, 1)
    return height, grime, inlay


def write_textures(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    h, g = barrel_maps()
    Image.fromarray((np.clip(h, 0, 1) * 65535).astype(np.uint16)).save(
        os.path.join(out_dir, 'engrave_barrel_h.png'))
    Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, 'engrave_barrel_g.png'))
    h, g, ink = lock_maps()
    Image.fromarray((np.clip(h, 0, 1) * 65535).astype(np.uint16)).save(
        os.path.join(out_dir, 'engrave_lock_h.png'))
    Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, 'engrave_lock_g.png'))
    Image.fromarray((np.clip(ink, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, 'inlay_lock.png'))


if __name__ == '__main__':
    import os
    import sys
    write_textures(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'textures',
        'source'))
