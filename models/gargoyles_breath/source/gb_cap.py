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

EYE_C = (206.0, 615.0)


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

    def surface_y(self, x, z):
        """|Y| of the base head surface at side-view point (x, z) mm."""
        d, W, R = self.maps(np.array([x]), np.array([z]))
        d, W, R = float(d[0]), float(W[0]), float(R[0])
        if d + R <= 0:
            return W
        q = R * R - (d + R) ** 2
        return W - R + np.sqrt(max(q, 0.0))


H2 = Head2D()


def surf(x, y, yy_off=0.0, side=1):
    """Point on the right/left head surface over ref px (x, y)."""
    a, b = mm(x, y)
    return (a, side * (H2.surface_y(a, b) + yy_off), b)


# --------------------------------------------------------------- reliefs --
def _flame_strokes():
    """Flowing flame/fur lines down the snout and around the eye (ref px
    paths, radius mm).  Generated as offset S-curves with curled ends."""
    out = []
    # snout flames: start under the eye / cheek, flow to the nose
    base = np.array([(0.0, 0.0), (2.0, 9.0), (1.0, 18.0), (2.6, 27.0),
                     (5.6, 35.0), (9.0, 41.0)])
    starts = [(203.0, 631.0), (207.5, 633.5), (212.5, 634.0), (217.5, 633.0),
              (222.0, 634.5), (226.5, 637.0), (231.0, 641.0)]
    lens = [0.95, 1.0, 0.98, 0.92, 0.84, 0.72, 0.55]
    for k, (sx, sy) in enumerate(starts):
        p = base * np.array([1.0 + 0.04 * k, lens[k]]) + np.array([sx, sy])
        end = p[-1]
        d = p[-1] - p[-2]
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) * (1 if k % 2 else -1)
        curl = [end + 1.6 * d + 1.2 * n, end + 1.4 * d + 2.8 * n,
                end + 0.2 * d + 3.1 * n, end - 0.5 * d + 2.0 * n]
        path = np.vstack([p, curl if k % 3 != 2 else curl[:2]])
        out.append((path, 1.05 - 0.04 * k))
    # eye flames: curl from the brow end round the eye and back to the crown
    out.append((np.array([(199, 627), (203, 631.5), (209, 631.5), (214, 627),
                          (216.5, 620), (216, 612), (213.5, 606)]), 1.0))
    out.append((np.array([(213, 604), (210, 597.5), (205, 593.5),
                          (199.5, 593.5), (196.5, 597), (198.5, 600.5)]), 0.95))
    # nasal bridge wrinkles (short arcs across the top of the snout)
    for k, yy in enumerate((660.0, 667.0, 673.5)):
        out.append((np.array([(186.0 + 4.5 * k, yy - 4.0),
                              (189.5 + 4.5 * k, yy - 1.0),
                              (194.0 + 4.6 * k, yy)]), 0.8))
    # flame tuft behind the eye toward the crown
    out.append((np.array([(196, 628), (191, 621), (189.5, 612), (192, 604),
                          (196.5, 601.5), (199, 605), (196.6, 607.6)]), 0.95))
    # cheek tuft sweeping back toward the ear
    out.append((np.array([(236, 646), (231, 636), (226.5, 627), (224.5, 618),
                          (226, 610.5), (229.5, 608), (231.5, 611.5),
                          (229.4, 614)]), 1.0))
    out.append((np.array([(239, 652), (234.6, 642), (230.5, 633),
                          (229.6, 625)]), 0.85))
    # jowl flames between the eye and the throat
    out.append((np.array([(242.5, 629), (237.6, 620), (234.6, 611),
                          (235.4, 603), (239.0, 599.4), (241.6, 602.6),
                          (239.6, 605.6)]), 0.9))
    out.append((np.array([(246.0, 619.5), (242.0, 611.5), (240.6, 604.0)]),
                0.7))
    out.append((np.array([(221.5, 640.0), (218.6, 630.4), (219.6, 621.0),
                          (223.6, 616.2), (227.0, 618.6), (225.4, 622.0)]),
                0.85))
    return out


RELIEFS = _flame_strokes()


class CapField:
    def __init__(self, grid):
        self.grid = grid
        X2, Z2 = np.meshgrid(grid.x.astype(np.float64),
                             grid.z.astype(np.float64), indexing='ij')
        d, W, R = H2.maps(X2, Z2)
        self.d_head = d.astype(np.float32)
        self.W = W.astype(np.float32)
        self.R = R.astype(np.float32)
        self.d_jaw = S.polygon_sdf2(X2, Z2, P.to_mm(
            P.catmull_rom(JAW, 5, closed=True)))
        hinge = np.array(mm(240, 640))
        tip = np.array(mm(266.4, 654.4))
        tt = np.clip(((X2 - hinge[0]) * (tip[0] - hinge[0]) +
                      (Z2 - hinge[1]) * (tip[1] - hinge[1])) /
                     np.sum((tip - hinge) ** 2), 0, 1)
        self.Wj = (6.2 - 3.9 * tt ** 1.2).astype(np.float32)
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
        self.reliefs = []
        for path, r in RELIEFS:
            for sd in (-1, 1):
                pts = [surf(x, y, -0.42 * r, sd) for x, y in
                       P.catmull_rom(path, 4)]
                nn = len(pts)
                rr = [1.25 * r * (0.45 + 0.55 * np.sin(np.pi * min(
                    1.0, (k + 0.7) / nn)) ** 0.6) for k in range(nn)]
                self.reliefs.append((pts, rr))

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

    def scale_mask(self, s, t, Y):
        phi = np.abs(np.arctan2(Y, t))
        # s measured from the axis start (inside the grip)
        lim = np.where(phi < 0.95, 70.0, np.where(phi < 1.75, 50.0, 30.0))
        m = S.smoothstep(lim, lim - 5.0, s)
        m = m * S.smoothstep(2.65, 2.25, phi)
        return m

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
        m = self.scale_mask(s, t, Y)
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
        brow = _chain(X, ay, Z, [mm3(196.5, 595, 17.8), mm3(198.5, 605, 19.6),
                                 mm3(200.0, 615, 20.2), mm3(203.5, 625, 19.2),
                                 mm3(208.5, 631, 17.6)],
                      [2.8, 3.9, 3.7, 2.9, 1.8])
        f = S.smin(f, brow, 2.2)
        cheek = S.ellipsoid(X, ay, Z, (*[mm(226, 622)[0]], 15.0,
                                       mm(226, 622)[1]), (6.0, 3.0, 6.5))
        f = S.smin(f, cheek, 3.0)
        # almond eye in a socket under the brow (long axis along the head)
        ax, az = -0.218, -0.976
        def eyept(a, b, yy):      # a along the head axis, b toward dorsal
            return (ex + a * ax + b * (-az), yy, ez + a * az + b * ax)
        sock = S.capsule(X, ay, Z, eyept(-3.2, 0.0, 21.0),
                         eyept(3.4, 0.3, 21.0), 2.7)
        f = S.smax(f, -sock, 0.8)
        ec = eyept(0.1, 0.1, 17.6)
        eye = S.ellipsoid(X, ay, Z, ec, (2.5, 2.3, 4.2))
        f = np.minimum(f, eye)
        lid_u = _chain(X, ay, Z, [eyept(-5.0, 0.4, 18.6), eyept(-1.6, 2.6, 20.3),
                                  eyept(2.2, 2.8, 20.4), eyept(5.4, 0.6, 18.8),
                                  eyept(7.6, -1.6, 17.6)],
                       [0.6, 1.45, 1.4, 0.8, 0.4])
        lid_l = _chain(X, ay, Z, [eyept(-4.2, -0.5, 18.4),
                                  eyept(-1.0, -2.0, 19.3),
                                  eyept(2.4, -2.0, 19.3),
                                  eyept(4.6, -0.6, 18.5)],
                       [0.45, 0.7, 0.7, 0.45])
        f = S.smin(f, np.minimum(lid_u, lid_l), 0.45)
        pupil = S.capsule(X, ay, Z, eyept(-1.4, 0.2, 19.45),
                          eyept(1.6, 0.25, 19.45), 0.36)
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
        # relief flames
        for pts, rr in self.reliefs:
            m = _bbox_mask(X, Y, Z, pts, 2.0)
            if np.any(m):
                g = _chain(X[m], Y[m], Z[m], pts, rr)
                ff = f[m]
                f = f.copy()
                f[m] = S.smin(ff, g.astype(ff.dtype), 0.75)
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
