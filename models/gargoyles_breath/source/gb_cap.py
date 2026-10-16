"""Silver gargoyle-head butt cap (SDF sculpt).

The head hangs nose-down at the end of the grip: its crown and crested,
scaled neck run up the back of the grip, the face looks down and forward,
and the open lower jaw sweeps forward where it joins the grip strap.
"""
import numpy as np
import gb_sdf as S
import gb_profiles as P
import gb_stock as ST


def mm(x, y):
    p = P.to_mm([(x, y)])[0]
    return float(p[0]), float(p[1])


def mm3(x, y, yy):
    a, b = mm(x, y)
    return (a, yy, b)


# ---------------------------------------------------------- silhouettes ---
HEAD_BASE = [  # ref px, without jaw / teeth / crest spikes
    (226, 516), (213, 531), (205, 537), (197, 544.5), (189.5, 552.5),
    (183, 561.5), (178, 571.5), (174.5, 582.5), (172.5, 594.5), (172, 607),
    (172.5, 619.5), (174.5, 631.5), (178, 643.5), (183.5, 655), (190.5, 665),
    (198.5, 674), (206.5, 681.5), (214.5, 686.5), (222, 688.5), (229, 687.2),
    (234, 683.4), (237, 677.2), (238.2, 670), (237.8, 662), (237, 654),
    (237.2, 646), (239, 639.5), (242.4, 633), (246, 625), (250, 617),
    (254.5, 609), (259, 603), (268, 590),
]
JAW = [
    (240.5, 631), (246.5, 629.5), (249.5, 639), (255, 645.6), (261.6, 649.6),
    (266.4, 654.4), (262.5, 657.4), (255, 660.4), (247.4, 664.6), (242, 664),
    (239, 656), (238.4, 645), (238.8, 637),
]
# (ref x, ref y, half width mm)
W_CTRL = [
    (222, 545, 22.6), (240, 575, 22.0), (200, 568, 21.2), (186, 590, 18.5),
    (206, 606, 20.8), (210, 622, 19.6), (184, 625, 15.5), (226, 632, 17.8),
    (243, 627, 14.6), (198, 648, 12.2), (212, 646, 13.6), (228, 648, 13.2),
    (200, 664, 10.6), (214, 664, 11.2), (230, 664, 10.8), (212, 680, 9.6),
    (222, 682, 9.4), (231, 683, 9.0),
]
# (ref x, ref y, rounding ratio r/W): round crown, boxy snout
RR_CTRL = [
    (222, 548, 0.90), (196, 580, 0.88), (206, 612, 0.80), (240, 610, 0.85),
    (190, 640, 0.62), (220, 645, 0.58), (205, 664, 0.50), (228, 665, 0.50),
    (218, 683, 0.58),
]
SIGMA = 12.0

# medial axis of the head (ref px) used to lay out the scale rows
AXIS = [(247, 552), (235.5, 571.5), (220, 590), (212, 606), (209, 622),
        (207.5, 642), (213, 664), (223, 683)]

EYE_C = (207.5, 614.5)


def _cap_region():
    b = P.to_mm(P.catmull_rom(P.CAP_BOUNDARY, 6))
    d0 = b[0] - b[1]
    d0 /= np.linalg.norm(d0)
    d1 = b[-1] - b[-2]
    d1 /= np.linalg.norm(d1)
    start = b[0] + d0 * 30.0
    end = b[-1] + d1 * 30.0
    far = [end + np.array([20.0, -90.0]), np.array([-280.0, -260.0]),
           np.array([-280.0, -20.0])]
    return np.vstack([start, b, end, far])


def _ctrl_mm(ctrl):
    return [(*mm(a, b), c) for a, b, c in ctrl]


def _flat_cone(x, y, z, a, b, ra, rb, thin_axis, thin=1.8):
    """Round cone squashed along thin_axis -> blade."""
    t = np.asarray(thin_axis, float)
    t = t / np.linalg.norm(t)
    c = (np.asarray(a, float) + np.asarray(b, float)) * 0.5
    px, py, pz = x - c[0], y - c[1], z - c[2]
    dt = px * t[0] + py * t[1] + pz * t[2]
    k = thin - 1.0
    qx, qy, qz = px + dt * t[0] * k, py + dt * t[1] * k, pz + dt * t[2] * k
    return S.round_cone(qx, qy, qz, np.asarray(a) - c, np.asarray(b) - c,
                        ra, rb) / (0.5 * (1.0 + thin))


def _chain(x, y, z, pts, radii):
    f = None
    for i in range(len(pts) - 1):
        g = S.round_cone(x, y, z, pts[i], pts[i + 1], radii[i], radii[i + 1])
        f = g if f is None else np.minimum(f, g)
    return f


def _bbox_mask(X, Y, Z, pts, pad):
    p = np.asarray(pts, float)
    lo = p.min(0) - pad
    hi = p.max(0) + pad
    return ((X > lo[0]) & (X < hi[0]) & (Y > lo[1]) & (Y < hi[1]) &
            (Z > lo[2]) & (Z < hi[2]))


class Head2D:
    """2D maps / helpers shared by the cap and the relief placement."""

    def __init__(self):
        self.poly = P.to_mm(P.catmull_rom(HEAD_BASE, 5))
        self.wc = _ctrl_mm(W_CTRL)
        self.rc = _ctrl_mm(RR_CTRL)

    def maps(self, X2, Z2):
        d = S.polygon_sdf2(X2, Z2, self.poly)
        W = S.rbf_field(X2, Z2, self.wc, SIGMA)[..., 0]
        R = S.rbf_field(X2, Z2, self.rc, SIGMA)[..., 0] * W
        return d, W, R


H2 = Head2D()


JAW_POLY = P.to_mm(P.catmull_rom(JAW, 5, closed=True))
JAW_HINGE = np.array(mm(240, 640))
JAW_TIP = np.array(mm(266.4, 654.4))


def _jaw_width(X2, Z2):
    hv = JAW_TIP - JAW_HINGE
    tt = np.clip(((X2 - JAW_HINGE[0]) * hv[0] + (Z2 - JAW_HINGE[1]) * hv[1]) /
                 hv.dot(hv), 0, 1)
    return 6.2 - 3.9 * tt ** 1.2


def _skull(X, ay, Z):
    """Brow ridge and cheekbone masses."""
    brow = _chain(X, ay, Z, [mm3(196.5, 595, 17.8), mm3(198.5, 605, 19.6),
                             mm3(200.0, 615, 20.2), mm3(203.5, 625, 19.2),
                             mm3(208.5, 631, 17.6)],
                  [2.8, 3.9, 3.7, 2.9, 1.8])
    cx, cz = mm(226, 622)
    cheek = S.ellipsoid(X, ay, Z, (cx, 15.0, cz), (6.0, 3.0, 6.5))
    return brow, cheek


def _base_pts(X, Y, Z):
    """Head, jaw, brow and cheek (no scales, eye or reliefs) at points."""
    ay = np.abs(Y)
    d, W, R = H2.maps(X, Z)
    W = W * (0.80 + 0.20 * S.smoothstep(0.0, -11.0, d))
    R = np.minimum(R, 0.97 * W)
    f = S.rounded_extrusion(d, ay - W, R)
    wj = _jaw_width(X, Z)
    dj = S.polygon_sdf2(X, Z, JAW_POLY)
    f = S.smin(f, S.rounded_extrusion(dj, ay - wj, 0.9 * wj), 1.6)
    brow, cheek = _skull(X, ay, Z)
    f = S.smin(f, brow, 2.2)
    return S.smin(f, cheek, 3.0)


def _on_surface(pts_px, side, depth):
    """Drop side-view points (ref px) onto the head surface on one side
    (+1 / -1) and push them `depth` mm under it along the surface normal."""
    xz = P.to_mm(pts_px)
    n = len(xz)
    ys = np.linspace(34.0, 0.0, 681)
    X = np.repeat(xz[:, 0:1], len(ys), 1)
    Z = np.repeat(xz[:, 1:2], len(ys), 1)
    Y = np.broadcast_to(ys, X.shape)
    f = _base_pts(X, Y, Z).astype(np.float64)
    inside = f < 0
    i = np.clip(np.argmax(inside, axis=1), 1, len(ys) - 1)
    rows = np.arange(n)
    f0, f1 = f[rows, i - 1], f[rows, i]
    a = np.clip(f0 / np.maximum(f0 - f1, 1e-9), 0, 1)
    ysurf = ys[i - 1] + (ys[i] - ys[i - 1]) * a
    ysurf = np.where(inside.any(axis=1), ysurf, ys[np.argmin(f, axis=1)])
    p = np.stack([xz[:, 0], side * ysurf, xz[:, 1]], 1)
    e = 0.05
    g = np.zeros_like(p)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = e
        hi = _base_pts(*(p + dp).T)
        lo = _base_pts(*(p - dp).T)
        g[:, k] = (hi - lo) / (2 * e)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    return p - g * np.asarray(depth, float)[:, None]


# --------------------------------------------------------------- reliefs --
def _flame(p0, p1, bend, curl, turn, rmax, n=22):
    """S-shaped flame lock from p0 to p1 (ref px) ending in a spiral curl.

    bend: sideways throw of the S (px); curl: start radius of the end
    spiral (px, 0 = tapered end); turn: spiral direction (+1 / -1).
    Returns (points px (N, 2), radii mm (N,)).
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    c1 = p0 + d / 3.0 + nrm * bend
    c2 = p0 + 2.0 * d / 3.0 - nrm * bend
    t = np.linspace(0.0, 1.0, n)[:, None]
    body = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 +
            3 * (1 - t) * t ** 2 * c2 + t ** 3 * p1)
    u = t[:, 0]
    rad = rmax * (0.35 + 0.65 * S.smoothstep(0.0, 0.35, u))
    if curl <= 0:
        return body, rad * (1.0 - 0.6 * S.smoothstep(0.65, 1.0, u))
    tan = body[-1] - body[-2]
    tan /= np.linalg.norm(tan)
    centre = p1 + turn * np.array([-tan[1], tan[0]]) * curl
    th = np.linspace(0.0, 1.7 * np.pi, 18)[1:]
    rho = curl * 0.3 ** (th / th[-1])
    v = p1 - centre
    ang = np.arctan2(v[1], v[0]) + turn * th
    spiral = centre + rho[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    srad = np.linspace(rad[-1], 0.45 * rmax, len(th))
    return np.vstack([body, spiral]), np.concatenate([rad, srad])


# (p0, p1, bend, curl, turn, radius mm), ref px of the right side view
LOCKS = [
    # flame locks flowing from under the eye down to the nose
    ((191, 641), (201, 677), 3.5, 2.6, -1, 1.1),
    ((199, 627), (198, 652), 4.0, 3.0, -1, 1.2),
    ((204, 629), (205, 668), 4.5, 3.4, -1, 1.3),
    ((210, 630), (214, 659), 4.2, 3.2, -1, 1.3),
    ((216, 631), (223, 671), 4.5, 3.2, -1, 1.25),
    ((222, 633), (230, 660), 3.6, 3.0, -1, 1.2),
    ((228, 636), (236, 651), 2.0, 2.6, -1, 1.05),
    # brow flame curling back over the eye
    ((215, 607), (198, 602), 2.6, 3.6, 1, 1.25),
    # cheek flames toward the mouth corner
    ((222, 604), (233, 628), 2.8, 3.2, -1, 1.2),
    ((228, 602), (238, 617), -1.6, 2.6, -1, 1.0),
]


def _flame_strokes():
    """Raised flame locks (ridges) and the fine incised lines between them.
    Returns (ridges, lines): lists of (points px, radii mm)."""
    ridges = [_flame(*lk) for lk in LOCKS]
    lines = []
    face = ridges[:7]
    n = 22                           # points in each lock's S-curve body
    for a, b in zip(face[:-1], face[1:]):
        mid = 0.5 * (a[0][:n] + b[0][:n])
        u = np.linspace(0, 1, n)
        lines.append((mid[2:-1], 0.42 * np.sin(np.pi * u[2:-1]) ** 0.5 + 0.08))
    for a in face:
        off = a[0][:n] + np.array([-1.7, 0.4])
        u = np.linspace(0, 1, n)
        lines.append((off[3:-3], 0.36 * np.sin(np.pi * u[3:-3]) ** 0.5 + 0.06))
    return ridges, lines


RIDGES, LINES = _flame_strokes()


class CapField:
    def __init__(self, grid):
        self.grid = grid
        X2, Z2 = np.meshgrid(grid.x.astype(np.float64),
                             grid.z.astype(np.float64), indexing='ij')
        d, W, R = H2.maps(X2, Z2)
        self.d_head = d.astype(np.float32)
        self.W = W.astype(np.float32)
        self.R = R.astype(np.float32)
        self.d_jaw = S.polygon_sdf2(X2, Z2, JAW_POLY)
        self.Wj = _jaw_width(X2, Z2).astype(np.float32)
        self.region = S.polygon_sdf2(X2, Z2, _cap_region())
        # scale layout: arc-length + offset relative to the medial axis
        ax = P.to_mm(P.catmull_rom(AXIS, 10))
        dist, par, total = S.polyline_dist2(X2, Z2, ax)
        # signed side: + toward the dorsal (rear) side
        seg = np.diff(ax, axis=0)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        i = np.clip(np.searchsorted(cum, par) - 1, 0, len(seg) - 1)
        tx, tz = seg[i, 0], seg[i, 1]
        nl = np.sqrt(tx * tx + tz * tz) + 1e-9
        nx, nz = tz / nl, -tx / nl          # rotate tangent -> dorsal normal
        side = (X2 - ax[i, 0]) * nx + (Z2 - ax[i, 1]) * nz
        self.ax_s = par.astype(np.float32)
        self.ax_t = side.astype(np.float32)
        self.stock = ST.StockField(grid)
        self._prep_reliefs()

    def _prep_reliefs(self):
        self.ridges = []
        self.lines = []
        for sd in (-1, 1):
            for pts, rr in RIDGES:
                self.ridges.append((_on_surface(pts, sd, 0.42 * rr), rr))
            for pts, rr in LINES:
                self.lines.append((_on_surface(pts, sd, 0.05 * rr), rr))

    # ----------------------------------------------------------- scales ---
    def scale_height(self, X, Y, Z, s, t):
        """Regular rows of overlapping scales, free edges toward the nose."""
        ds_row = 5.0
        nphi = 24
        dphi = 2 * np.pi / nphi
        rho = np.sqrt(t * t + Y * Y) + 1e-6
        phi = np.arctan2(Y, t)                      # 0 = dorsal
        h = np.zeros_like(s)
        k0 = np.floor(s / ds_row)
        for dk in (-1, 0, 1):
            k = k0 + dk
            sc = (k + 0.5) * ds_row
            off = (np.mod(k, 2)) * 0.5 * dphi
            j = np.round((phi - off) / dphi)
            for dj in (-1, 0, 1):
                pc = (j + dj) * dphi + off
                dl = np.angle(np.exp(1j * (phi - pc))) * rho
                ds = s - sc
                A = 0.80 * ds_row
                Bw = 0.62 * dphi * np.maximum(rho, 6.0)
                rr = np.sqrt(((ds + 0.10 * A) / A) ** 2 + (dl / Bw) ** 2)
                u = np.clip((ds + 1.1 * A) / (2.0 * A), 0.0, 1.0)
                prof = S.smoothstep(1.0, 0.88, rr) * \
                    (0.18 + 0.82 * np.sin(0.5 * np.pi * u) ** 1.6) * \
                    (1.0 - 0.45 * np.clip(dl / Bw, -1, 1) ** 2)
                h = np.maximum(h, prof)
        return h

    def scale_mask(self, s, t, X, Y, Z):
        phi = np.abs(np.arctan2(Y, t))
        # s runs from inside the grip toward the nose: the scales cover the
        # bridge of the nose and the upper cheek, the mouth side stays bare
        lim = np.interp(phi, [0.0, 0.9, 1.35, 1.8, 2.25],
                        [84.0, 82.0, 72.0, 58.0, 32.0])
        m = S.smoothstep(lim, lim - 5.0, s)
        m = m * S.smoothstep(2.65, 2.25, phi)
        ex, ez = mm(*EYE_C)
        de = np.sqrt((X - ex) ** 2 + (np.abs(Y) - 18.0) ** 2 + (Z - ez) ** 2)
        return m * S.smoothstep(5.5, 9.0, de)

    # ----------------------------------------------------------- field ----
    def eval_points(self, X, Y, Z, ix, iz):
        ay = np.abs(Y)
        dh = self.d_head[ix, iz]
        W = self.W[ix, iz] * (0.80 + 0.20 * S.smoothstep(0.0, -11.0,
                                                         self.d_head[ix, iz]))
        R = np.minimum(self.R[ix, iz], 0.97 * W)
        f = S.rounded_extrusion(dh, ay - W, R)
        jaw = S.rounded_extrusion(self.d_jaw[ix, iz], ay - self.Wj[ix, iz],
                                  0.9 * self.Wj[ix, iz])
        f = S.smin(f, jaw, 1.6)
        fs = self.stock.eval_points(X, Y, Z, ix, iz)
        f = S.smin(f, fs - 1.05, 0.8)
        reg = self.region[ix, iz]
        f = np.maximum(f, reg)
        rim = np.maximum(fs - 1.45, np.abs(reg + 0.9) - 0.9)
        f = np.minimum(f, rim)
        # scales on crown / neck / dorsal ridge
        s = self.ax_s[ix, iz].astype(np.float64)
        t = self.ax_t[ix, iz].astype(np.float64)
        m = self.scale_mask(s, t, X, Y, Z)
        near = (m > 0.01) & (np.abs(f) < 1.8)
        if np.any(near):
            h = self.scale_height(X[near], Y[near], Z[near], s[near], t[near])
            f = f.copy()
            f[near] = f[near] - (1.35 * h * m[near]).astype(f.dtype)
        f = self._face(f, X, Y, Z, ay)
        f = self._crest(f, X, Y, Z)
        return f.astype(np.float32)

    def _face(self, f, X, Y, Z, ay):
        ex, ez = mm(*EYE_C)
        # skull masses: brow ridge, cheekbone, jowl
        brow, cheek = _skull(X, ay, Z)
        f = S.smin(f, brow, 2.2)
        f = S.smin(f, cheek, 3.0)
        # round eye with a ring lid, heavier on the dorsal side, set in a
        # socket under the brow
        ax, az = -0.218, -0.976
        def eyept(a, b, yy):      # a along the head axis, b toward dorsal
            return (ex + a * ax + b * (-az), yy, ez + a * az + b * ax)
        sock = S.capsule(X, ay, Z, eyept(-1.8, 0.0, 21.2),
                         eyept(2.0, 0.2, 21.2), 3.1)
        f = S.smax(f, -sock, 0.8)
        eye = S.ellipsoid(X, ay, Z, eyept(0.1, 0.1, 17.7), (2.7, 2.4, 3.2))
        f = np.minimum(f, eye)
        th = np.linspace(0.0, 2.0 * np.pi, 17)
        ring = [eyept(3.7 * np.cos(q), 3.2 * np.sin(q) + 0.1,
                      19.3 + 0.7 * max(np.sin(q), 0.0)) for q in th]
        lids = _chain(X, ay, Z, ring,
                      [0.5 + 0.5 * max(np.sin(q), 0.0) for q in th])
        f = S.smin(f, lids, 0.45)
        pupil = S.capsule(X, ay, Z, eyept(-1.7, 0.2, 19.55),
                          eyept(1.7, 0.25, 19.55), 0.38)
        f = np.maximum(f, -pupil)
        # swept-back ears: concave blades rising toward the grip
        eb = mm3(213.5, 600.0, 19.0)
        et = mm3(226.5, 578.5, 25.6)
        ear = _flat_cone(X, ay, Z, eb, et, 5.0, 0.35, (0.35, 1.0, -0.1),
                         thin=1.9)
        cup = _flat_cone(X, ay, Z, (eb[0] + 0.5, 21.9, eb[2] + 0.3),
                         (et[0] - 1.4, 26.6, et[2] - 1.6), 3.0, 0.15,
                         (0.35, 1.0, -0.1), thin=2.4)
        f = S.smin(f, S.smax(ear, -cup, 0.35), 1.5)
        # nose pad and flared nostrils
        nx, nz = mm(226.0, 683.0)
        pad = S.ellipsoid(X, Y, Z, (nx + 0.6, 0, nz + 0.4), (4.6, 7.4, 3.6))
        f = S.smin(f, pad, 1.4)
        flare = S.ellipsoid(X, ay, Z, (nx + 1.2, 5.2, nz - 0.2),
                            (2.6, 2.3, 2.8))
        f = S.smin(f, flare, 1.0)
        nos = S.ellipsoid(X, ay, Z, (nx + 2.2, 4.4, nz - 1.8),
                          (2.0, 1.6, 2.6))
        f = S.smax(f, -nos, 0.35)
        nos_curl = S.torus_y(X, ay - 6.4, Z, (nx + 1.0, 0, nz - 0.4), 2.6, 0.6)
        f = S.smin(f, np.maximum(nos_curl, -(Z - nz + 1.2)), 0.4)
        # mouth corner groove and lip line
        lip = _chain(X, ay, Z, [mm3(239.5, 641, 12.9), mm3(237.6, 652, 11.9),
                                mm3(236.6, 664, 10.6), mm3(234.0, 676, 8.8)],
                     [0.55, 0.62, 0.58, 0.45])
        f = S.smax(f, -lip, 0.3)
        # teeth
        teeth = []
        for sd in (-1, 1):
            r0 = mm(237.4, 667.0)
            t0 = mm(247.0, 671.4)
            teeth.append(S.round_cone(X, Y, Z, (r0[0] - 1.2, sd * 5.8, r0[1]),
                                      (t0[0], sd * 4.6, t0[1]), 2.0, 0.1))
            r1 = mm(251.6, 661.2)
            t1 = mm(244.0, 657.4)
            teeth.append(S.round_cone(X, Y, Z, (r1[0], sd * 3.4, r1[1]),
                                      (t1[0], sd * 2.8, t1[1]), 1.45, 0.08))
            for k, (rx, ry) in enumerate([(237.6, 658.5), (237.2, 651.0),
                                          (237.8, 644.0)]):
                q = mm(rx, ry)
                teeth.append(S.round_cone(
                    X, Y, Z, (q[0] - 0.8, sd * (6.6 + 0.6 * k), q[1]),
                    (q[0] + 2.4, sd * (6.1 + 0.5 * k), q[1] - 0.8), 0.9, 0.08))
            for k, (rx, ry) in enumerate([(247.0, 650.0), (244.5, 643.5)]):
                q = mm(rx, ry)
                teeth.append(S.round_cone(
                    X, Y, Z, (q[0] + 0.6, sd * 4.2, q[1]),
                    (q[0] - 2.2, sd * 4.0, q[1] + 0.4), 0.8, 0.08))
        f = np.minimum(f, np.min(np.stack(teeth), axis=0))
        # chin barb at the jaw tip
        barb = S.round_cone(X, Y, Z, mm3(257.5, 657.8, 0.0),
                            mm3(268.2, 655.0, 0.0), 2.5, 0.12)
        f = S.smin(f, barb, 0.8)
        # flame locks in low relief, fine incised lines between them
        for pts, rr in self.ridges:
            m = _bbox_mask(X, Y, Z, pts, float(np.max(rr)) + 1.0)
            if np.any(m):
                g = _chain(X[m], Y[m], Z[m], pts, rr)
                f = f.copy()
                f[m] = S.smin(f[m], g.astype(f.dtype), 0.3)
        for pts, rr in self.lines:
            m = _bbox_mask(X, Y, Z, pts, float(np.max(rr)) + 0.6)
            if np.any(m):
                g = _chain(X[m], Y[m], Z[m], pts, rr)
                f = f.copy()
                f[m] = S.smax(f[m], -g.astype(f.dtype), 0.12)
        return f

    def _crest(self, f, X, Y, Z):
        """Blade-like crest spikes sweeping up the back of the head."""
        dors = P.to_mm(P.catmull_rom(HEAD_BASE[1:14], 10))
        seg = np.diff(dors, axis=0)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        total = cum[-1]
        center = np.array(mm(214, 612))
        rows = [(0.0, 0.0, 1.45, 6), (7.6, 0.5, 1.2, 5), (-7.6, 0.5, 1.2, 5)]
        for ry, phase, scl, n in rows:
            for k in range(n):
                sp = (k + 0.35 + phase) / (n + 0.2) * total * 0.97
                if sp > total - 3.0:
                    continue
                i = int(np.clip(np.searchsorted(cum, sp) - 1, 0, len(seg) - 1))
                tt = (sp - cum[i]) / max(np.linalg.norm(seg[i]), 1e-6)
                p = dors[i] + seg[i] * tt
                tan = -seg[i] / np.linalg.norm(seg[i])
                nrm = np.array([-tan[1], tan[0]])
                if nrm @ (p - center) < 0:
                    nrm = -nrm
                size = scl * (1.05 - 0.40 * (sp / total)) * \
                    (0.9 + 0.18 * np.sin(k * 2.7 + ry))
                inset = 0.9 + 0.2 * abs(ry)
                base = p - nrm * inset
                d1 = tan * 0.72 + nrm * 0.69
                d1 /= np.linalg.norm(d1)
                d2 = tan * 0.985 + nrm * 0.17
                d2 /= np.linalg.norm(d2)
                m1 = base + d1 * 4.4 * size
                tip = m1 + d2 * 7.0 * size
                yb, yt = ry, ry * 1.28
                a = (base[0], yb, base[1])
                b = (m1[0], 0.5 * (yb + yt), m1[1])
                c = (tip[0], yt, tip[1])
                thin = (0.0, 1.0, 0.0) if ry == 0 else (0.0, 1.0, 0.3 * np.sign(ry))
                m = _bbox_mask(X, Y, Z, [a, b, c], 5.0 * size + 1.5)
                if not np.any(m):
                    continue
                g = np.minimum(
                    _flat_cone(X[m], Y[m], Z[m], a, b, 3.6 * size, 2.6 * size,
                               thin, 3.3),
                    _flat_cone(X[m], Y[m], Z[m], b, c, 2.6 * size, 0.2,
                               thin, 3.3))
                # shallow groove down the blade
                gr = _flat_cone(X[m], Y[m], Z[m],
                                (a[0], a[1] - 0.0, a[2]), c, 0.5 * size, 0.1,
                                thin, 1.0)
                g = S.smax(g, -(gr + 0.25), 0.2)
                ff = f[m]
                f = f.copy()
                f[m] = S.smin(ff, g.astype(ff.dtype), 1.0)
        return f


LO = (-192.0, -30.0, -170.0)
HI = (-98.0, 30.0, -38.0)


def build(h=0.2, factor=4):
    vol, grid = S.evaluate_narrowband(CapField, LO, HI, h, factor=factor,
                                      band=2.0 * h * factor + 2.0,
                                      verbose=True)
    return S.mesh_volume(vol, grid)


if __name__ == '__main__':
    import sys
    import time
    t = time.time()
    hh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    v, f = build(hh)
    np.savez(sys.argv[2] if len(sys.argv) > 2 else 'cap.npz', v=v, f=f)
    print(len(v), len(f), time.time() - t)
