"""Walnut stock: inflated side silhouette with variable width/rounding."""
import numpy as np
import gb_sdf as S
import gb_profiles as P

# half-width / edge-radius control points (X mm, Z mm, half width, radius)
WIDTH_CTRL = [
    # fore-end, widening into the nose that cradles the bell
    (216, -12, 20.7, 4.2), (204, -12, 20.6, 4.6), (186, -12, 19.6, 5.0),
    (160, -12, 18.9, 5.2), (138, -16, 18.8, 5.2), (122, -24, 19.2, 5.0),
    # lock region (the lock/side-plate panels are added separately)
    (100, -18, 19.8, 3.4), (60, -18, 19.9, 3.2), (20, -18, 19.9, 3.2),
    (-10, -16, 19.6, 3.6), (-38, -18, 18.6, 5.0),
    # wrist
    (-30, 8, 16.6, 7.0), (-55, 4, 16.2, 8.5), (-62, -30, 17.0, 9.0),
    (-80, -14, 16.4, 10.5), (-92, -42, 17.4, 11.0),
    # grip / butt
    (-104, -30, 17.6, 11.5), (-108, -60, 18.6, 12.5), (-118, -76, 19.8, 13.2),
    (-128, -88, 20.6, 13.8), (-138, -98, 21.2, 14.2), (-146, -66, 20.4, 13.0),
    (-150, -84, 21.0, 13.8),
]
SIGMA = 22.0
PANEL_GROW = 3.6      # lock panel outline = lock plate outline grown by this
PANEL_RAISE = 1.0     # panel stands proud of the surrounding wood by this


def outline_mm():
    """Closed side silhouette of the stock in model mm."""
    wrist = P.catmull_rom(P.to_mm(P.STOCK_WRIST_TOP[1:]), 6)
    butt = P.catmull_rom(P.to_mm(P.STOCK_BUTT_END), 6)
    front = P.catmull_rom(P.to_mm(P.STOCK_GRIP_FRONT), 6)
    bottom = P.catmull_rom(P.to_mm(P.STOCK_BOTTOM), 6)
    top = P.to_mm(P.STOCK_TOP_FOREEND)
    step = P.to_mm([P.STOCK_WRIST_TOP[0]])
    pts = np.vstack([top, step, wrist, butt[1:], front[1:], bottom[1:-1]])
    # drop near-duplicates
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.25:
            keep.append(i)
    return pts[keep]


def panel_outline_mm():
    return P.catmull_rom(P.to_mm(P.LOCK_PLATE), 6, closed=True)


def muzzle_radius(x):
    prof = np.array(P.MUZZLE_PROFILE)
    return np.interp(x, prof[:, 0], prof[:, 1])


def octagon_sdf2(y, z, inr):
    """2D SDF of a regular octagon (flats on the axes) with inradius inr."""
    k = np.array([-0.9238795325, 0.3826834323, 0.4142135623])
    py, pz = np.abs(y), np.abs(z)
    d = np.minimum(k[0] * py + k[1] * pz, 0.0)
    py = py - 2.0 * d * k[0]
    pz = pz - 2.0 * d * k[1]
    d = np.minimum(-k[0] * py + k[1] * pz, 0.0)    # second reflection
    py = py - 2.0 * d * (-k[0])
    pz = pz - 2.0 * d * k[1]
    py = py - np.clip(py, -k[2] * inr, k[2] * inr)
    pz = pz - inr
    return np.sqrt(py * py + pz * pz) * np.sign(pz)


def barrel_channel(x, y, z, clearance=0.12):
    """Solid occupied by the barrel (+clearance) for channel cutting."""
    octa = octagon_sdf2(y, z, P.BARREL_INRADIUS + clearance)
    octa = np.maximum(octa, np.maximum(-0.3 - x, x - 196.0))
    r = muzzle_radius(np.clip(x, 194.0, 268.0)) + clearance
    bell = np.sqrt(y * y + z * z) - r
    bell = np.maximum(bell, np.maximum(194.0 - x, x - 280.0))
    return np.minimum(octa, bell)


RAMROD_Z = (P.Z0 - P.RAMROD_Z_PX) * P.K
RAMROD_R = P.RAMROD_DIAM_PX * 0.5 * P.K


class StockField:
    def __init__(self, grid, recess=True):
        self.grid = grid
        self.recess = recess
        X2, Z2 = np.meshgrid(grid.x.astype(np.float64),
                             grid.z.astype(np.float64), indexing='ij')
        poly = outline_mm()
        self.d2 = S.polygon_sdf2(X2, Z2, poly)
        wr = S.rbf_field(X2, Z2, [(a, b, c, d) for a, b, c, d in WIDTH_CTRL],
                         SIGMA)
        W, R = wr[..., 0], wr[..., 1]
        # crisp top edge along the barrel channel (fore-end + lock region)
        crisp = (1.0 - S.smoothstep(-3.0, -11.0, Z2)) * \
            S.smoothstep(-6.0, 2.0, X2)
        R = R - (R - 0.9) * crisp
        # raised lock / side-plate panels
        pd = S.polygon_sdf2(X2, Z2, panel_outline_mm()) - PANEL_GROW
        W = W + PANEL_RAISE * S.smoothstep(1.4, -1.4, pd)
        self.W = W.astype(np.float32)
        self.R = R.astype(np.float32)

    def eval_points(self, X, Y, Z, ix, iz):
        d2 = self.d2[ix, iz]
        W = self.W[ix, iz]
        R = self.R[ix, iz]
        f = S.rounded_extrusion(d2, np.abs(Y) - W, R)
        # barrel channel and ramrod bore
        f = np.maximum(f, -barrel_channel(X, Y, Z))
        rod = np.sqrt(Y * Y + (Z - RAMROD_Z) ** 2) - (RAMROD_R + 0.25)
        rod = np.maximum(rod, np.maximum(-22.0 - X, X - 135.0))
        f = np.maximum(f, -rod)
        # recess for the barrel tang on top of the wrist
        if self.recess:
            f = np.maximum(f, -tang_recess(X, Y, Z))
        return f.astype(np.float32)


# barrel tang: brass strap on top of the wrist, X from 0 back to TANG_END
TANG_END = -46.0
TANG_HALF_W = 7.8


def tang_plan(x, y):
    """2D plan-view SDF of the tang (rounded, tapering to a point)."""
    t = np.clip(-x / -TANG_END, 0.0, 1.0)
    hw = TANG_HALF_W * (1.0 - 0.55 * t ** 1.6)
    d = np.abs(y) - hw
    d = np.maximum(d, x - 0.2)
    tip = np.sqrt((x - (TANG_END + 4.0)) ** 2 + y * y) - 4.2
    body = np.maximum(d, (TANG_END + 4.0) - x)
    return np.minimum(body, tip)


def tang_recess(x, y, z):
    zt = wrist_top_z(x)
    plan = tang_plan(x, y)
    return np.maximum(plan, np.maximum(z - (zt + 5.0), (zt - 1.1) - z))


_WT = None


def wrist_top_z(x):
    """Height of the wrist top line (side silhouette) as a function of X."""
    global _WT
    if _WT is None:
        pts = P.catmull_rom(P.to_mm(P.STOCK_WRIST_TOP[:12]), 8)
        o = np.argsort(pts[:, 0])
        _WT = (pts[o, 0], pts[o, 1])
    return np.interp(x, _WT[0], _WT[1])


LO = (-160.0, -24.0, -120.0)
HI = (225.0, 24.0, 20.0)


def build(h=0.3, factor=4):
    vol, grid = S.evaluate_narrowband(StockField, LO, HI, h, factor=factor,
                                      band=2.0 * h * factor + 0.6)
    return S.mesh_volume(vol, grid)


if __name__ == '__main__':
    import time
    t = time.time()
    v, f = build(0.6)
    print(len(v), len(f), time.time() - t)
