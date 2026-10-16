"""Small signed-distance-field toolkit (numpy) used for the organic parts.

Fields are evaluated on regular grids and meshed with marching cubes.  All
functions take coordinate arrays (broadcastable) and return float32 arrays.
"""
import numpy as np
from skimage import measure


# ----------------------------------------------------------------- basics --
def length(*c):
    return np.sqrt(sum(np.square(a) for a in c))


def smin(a, b, k):
    """Polynomial smooth minimum (union) with blend radius k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------- primitives ----
def sphere(x, y, z, c, r):
    return length(x - c[0], y - c[1], z - c[2]) - r


def ellipsoid(x, y, z, c, r):
    """Approximate SDF of an axis aligned ellipsoid (iq)."""
    px, py, pz = (x - c[0]), (y - c[1]), (z - c[2])
    k0 = length(px / r[0], py / r[1], pz / r[2])
    k1 = length(px / r[0] ** 2, py / r[1] ** 2, pz / r[2] ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def round_cone(x, y, z, a, b, ra, rb):
    """Capsule with different radii at the two ends (iq's sdRoundCone)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = ba.dot(ba)
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    px = np.asarray(x - a[0], np.float64)
    py = np.asarray(y - a[1], np.float64)
    pz = np.asarray(z - a[2], np.float64)
    yv = px * ba[0] + py * ba[1] + pz * ba[2]
    zv = yv - l2
    xx = (px * l2 - ba[0] * yv) ** 2 + (py * l2 - ba[1] * yv) ** 2 + \
         (pz * l2 - ba[2] * yv) ** 2
    y2 = yv * yv * l2
    z2 = zv * zv * l2
    k = np.sign(rr) * rr * rr * xx
    c1 = np.sign(zv) * a2 * z2 > k
    c2 = np.sign(yv) * a2 * y2 < k
    o1 = np.sqrt(xx + z2) * il2 - rb
    o2 = np.sqrt(xx + y2) * il2 - ra
    o3 = (np.sqrt(np.maximum(xx * a2 * il2, 0.0)) + yv * rr) * il2 - ra
    return np.where(c1, o1, np.where(c2, o2, o3)).astype(np.float32)


def capsule(x, y, z, a, b, r):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    px, py, pz = x - a[0], y - a[1], z - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / ba.dot(ba), 0, 1)
    return length(px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h) - r


def torus_y(x, y, z, c, R, r):
    """Torus whose axis is the Y axis, centred at c."""
    px, py, pz = x - c[0], y - c[1], z - c[2]
    q = length(px, pz) - R
    return length(q, py) - r


# --------------------------------------------------------- 2D polygons -----
def polygon_sdf2(px, pz, poly):
    """Exact signed distance to a closed polygon (negative inside)."""
    poly = np.asarray(poly, float)
    d = np.full(np.broadcast(px, pz).shape, np.inf, dtype=np.float64)
    s = np.ones_like(d)
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        bx, bz = poly[i - 1]
        ex, ez = bx - ax, bz - az
        wx, wz = px - ax, pz - az
        ee = ex * ex + ez * ez
        if ee < 1e-12:
            continue
        t = np.clip((wx * ex + wz * ez) / ee, 0.0, 1.0)
        dx, dz = wx - ex * t, wz - ez * t
        d = np.minimum(d, dx * dx + dz * dz)
        c1 = pz >= az
        c2 = pz < bz
        c3 = ex * wz > ez * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return (s * np.sqrt(d)).astype(np.float32)


def polyline_dist2(px, pz, line):
    """Unsigned distance to an open polyline, plus the arc-length parameter
    of the closest point (useful for width profiles)."""
    line = np.asarray(line, float)
    seg = np.diff(line, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    best = np.full(np.broadcast(px, pz).shape, np.inf)
    par = np.zeros_like(best)
    for i in range(len(seg)):
        ax, az = line[i]
        ex, ez = seg[i]
        ee = max(seglen[i] ** 2, 1e-12)
        t = np.clip(((px - ax) * ex + (pz - az) * ez) / ee, 0, 1)
        dx, dz = px - ax - ex * t, pz - az - ez * t
        dd = dx * dx + dz * dz
        m = dd < best
        best = np.where(m, dd, best)
        par = np.where(m, cum[i] + t * seglen[i], par)
    return np.sqrt(best), par, cum[-1]


def rounded_extrusion(d2, w, r):
    """Combine a 2D side-profile distance d2 with lateral distance w=|y|-W
    into a slab with edges rounded by radius r (r may vary)."""
    q1 = d2 + r
    q2 = w + r
    out = np.minimum(np.maximum(q1, q2), 0.0) + \
        np.sqrt(np.maximum(q1, 0.0) ** 2 + np.maximum(q2, 0.0) ** 2) - r
    return out


def rbf_field(px, pz, ctrl, sigma):
    """Gaussian-weighted (Shepard) interpolation of control values.
    ctrl: list of (x, z, v0, v1, ...)."""
    ctrl = np.asarray(ctrl, float)
    num = 0.0
    den = 0.0
    for row in ctrl:
        w = np.exp(-((px - row[0]) ** 2 + (pz - row[1]) ** 2) / (sigma * sigma))
        w = np.maximum(w, 1e-30)
        num = num + w[..., None] * row[2:]
        den = den + w
    return num / den[..., None]


# ------------------------------------------------------------- meshing -----
class Grid:
    def __init__(self, lo, hi, h):
        self.lo = np.asarray(lo, float)
        self.h = float(h)
        self.n = np.ceil((np.asarray(hi, float) - self.lo) / h).astype(int) + 1
        self.x = (self.lo[0] + np.arange(self.n[0]) * h).astype(np.float32)
        self.y = (self.lo[1] + np.arange(self.n[1]) * h).astype(np.float32)
        self.z = (self.lo[2] + np.arange(self.n[2]) * h).astype(np.float32)

    def axes(self, i0=0, i1=None):
        """Broadcastable coordinate arrays for x-slab [i0, i1)."""
        x = self.x[i0:i1][:, None, None]
        return x, self.y[None, :, None], self.z[None, None, :]

    def full(self, i0=0, i1=None):
        x, y, z = self.axes(i0, i1)
        return np.broadcast_arrays(x, y, z)

    def evaluate(self, fn, slab=64):
        vol = np.empty(tuple(self.n), dtype=np.float32)
        for i0 in range(0, self.n[0], slab):
            i1 = min(self.n[0], i0 + slab)
            vol[i0:i1] = fn(self, i0, i1)
        return vol


def mesh_volume(vol, grid, level=0.0, step=1):
    """Marching cubes -> (verts (N,3) model units, faces (M,3))."""
    v, f, n, _ = measure.marching_cubes(
        vol, level=level, spacing=(grid.h,) * 3, step_size=step,
        allow_degenerate=False)
    v = v + grid.lo[None, :]
    f = f.astype(np.int64)
    tri = v[f]
    signed = np.einsum('ij,ij->i', tri[:, 0],
                       np.cross(tri[:, 1], tri[:, 2])).sum()
    if signed < 0:          # make normals point out of the solid
        f = f[:, ::-1].copy()
    return v.astype(np.float64), f


# ------------------------------------------------- point-based evaluation --
def evaluate_dense(field, grid, slab=16):
    """field.eval_points(X, Y, Z, ix, iz) on every voxel of grid."""
    vol = np.empty(tuple(grid.n), dtype=np.float32)
    ny, nz = grid.n[1], grid.n[2]
    iy, iz = np.meshgrid(np.arange(ny), np.arange(nz), indexing='ij')
    iy = iy.ravel()
    iz = iz.ravel()
    for i0 in range(0, grid.n[0], slab):
        i1 = min(grid.n[0], i0 + slab)
        ix = np.repeat(np.arange(i0, i1), ny * nz)
        jy = np.tile(iy, i1 - i0)
        jz = np.tile(iz, i1 - i0)
        val = field.eval_points(grid.x[ix].astype(np.float64),
                                grid.y[jy].astype(np.float64),
                                grid.z[jz].astype(np.float64), ix, jz)
        vol[i0:i1] = val.reshape(i1 - i0, ny, nz)
    return vol


def evaluate_narrowband(field_cls, lo, hi, h, factor=4, band=None, slab=8,
                        verbose=False):
    """Evaluate exactly only near the surface; elsewhere use the trilinear
    upsampled coarse field (sign-correct, |f| > band so no surfaces)."""
    from scipy import ndimage
    gc = Grid(lo, hi, h * factor)
    fc = field_cls(gc)
    volc = evaluate_dense(fc, gc, slab=8)
    if band is None:
        band = 2.0 * h * factor + 1.0
    gf = Grid(lo, hi, h)
    ff = field_cls(gf)
    vol = np.empty(tuple(gf.n), dtype=np.float32)
    ny, nz = gf.n[1], gf.n[2]
    ys = np.arange(ny) / factor
    zs = np.arange(nz) / factor
    YY, ZZ = np.meshgrid(ys, zs, indexing='ij')
    n_active = 0
    for i0 in range(0, gf.n[0], slab):
        i1 = min(gf.n[0], i0 + slab)
        xs = np.arange(i0, i1) / factor
        cx = np.repeat(xs, ny * nz)
        cy = np.tile(YY.ravel(), i1 - i0)
        cz = np.tile(ZZ.ravel(), i1 - i0)
        coarse = ndimage.map_coordinates(volc, [cx, cy, cz], order=1,
                                         mode='nearest').astype(np.float32)
        act = np.abs(coarse) < band
        if np.any(act):
            idx = np.nonzero(act)[0]
            ix = i0 + idx // (ny * nz)
            rem = idx % (ny * nz)
            jy = rem // nz
            jz = rem % nz
            coarse[idx] = ff.eval_points(gf.x[ix].astype(np.float64),
                                         gf.y[jy].astype(np.float64),
                                         gf.z[jz].astype(np.float64), ix, jz)
            n_active += len(idx)
        vol[i0:i1] = coarse.reshape(i1 - i0, ny, nz)
    if verbose:
        print('narrowband active voxels: %d of %d (%.1f%%)' % (
            n_active, vol.size, 100.0 * n_active / vol.size))
    return vol, gf
