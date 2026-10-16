"""Hard-surface parts: barrel, muzzle, ramrod, pipes, lock work, furniture.

Lathe / prism parts are generated as clean quad meshes with UVs; small
sculpted parts (lock plate, cock, frizzen, guard, side plate...) are SDFs
meshed by marching cubes.
"""
import numpy as np
import gb_sdf as S
import gb_profiles as P
import gb_stock as ST

K = P.K


def px(x, y):
    p = P.to_mm([(x, y)])[0]
    return float(p[0]), float(p[1])


def pxs(pts):
    return P.to_mm(pts)


# =========================================================== mesh helpers ==
def _rot_from_x(axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    up = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([0, 1.0, 0])
    b = np.cross(up, a)
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=1)        # columns: x->a, y->b, z->c


def lathe(profile, n=96, origin=(0, 0, 0), axis=(1, 0, 0), cap0=False,
          cap1=False, uv_rect=(0, 0, 1, 1), phase=0.0):
    """Revolve (axial, radius) profile.  Returns verts, faces, loop_uvs."""
    prof = np.asarray(profile, float)
    K_ = len(prof)
    th = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(th), np.sin(th)], 1)
    v = np.zeros((K_ * n, 3))
    for i, (a, r) in enumerate(prof):
        v[i * n:(i + 1) * n, 0] = a
        v[i * n:(i + 1) * n, 1] = r * ring[:, 0]
        v[i * n:(i + 1) * n, 2] = r * ring[:, 1]
    seg = np.linalg.norm(np.diff(prof, axis=0), axis=1)
    vv = np.concatenate([[0], np.cumsum(seg)])
    vv = vv / max(vv[-1], 1e-9)
    u0, v0, u1, v1 = uv_rect
    faces, uvs = [], []
    for i in range(K_ - 1):
        for j in range(n):
            j1 = (j + 1) % n
            faces.append([i * n + j, (i + 1) * n + j, (i + 1) * n + j1,
                          i * n + j1])
            uu0 = u0 + (u1 - u0) * j / n
            uu1 = u0 + (u1 - u0) * (j + 1) / n
            va = v0 + (v1 - v0) * vv[i]
            vb = v0 + (v1 - v0) * vv[i + 1]
            uvs += [(uu0, va), (uu0, vb), (uu1, vb), (uu1, va)]
    verts = [v]
    if cap0 and prof[0, 1] > 1e-6:
        c = len(np.vstack(verts))
        verts.append(np.array([[prof[0, 0], 0, 0]]))
        for j in range(n):
            j1 = (j + 1) % n
            faces.append([c, j1, j])
            uvs += [(0.5 * (u0 + u1), v0), (u0, v0), (u0, v0)]
    if cap1 and prof[-1, 1] > 1e-6:
        c = len(np.vstack(verts))
        verts.append(np.array([[prof[-1, 0], 0, 0]]))
        b = (K_ - 1) * n
        for j in range(n):
            j1 = (j + 1) % n
            faces.append([c, b + j, b + j1])
            uvs += [(0.5 * (u0 + u1), v1), (u1, v1), (u1, v1)]
    v = np.vstack(verts)
    R = _rot_from_x(axis)
    v = v @ R.T + np.asarray(origin, float)
    return v, faces, np.asarray(uvs)


def fix_winding(v, faces):
    """Flip all faces if the mesh's signed volume is negative."""
    tot = 0.0
    for f in faces:
        p0 = v[f[0]]
        for k in range(1, len(f) - 1):
            tot += np.dot(p0, np.cross(v[f[k]], v[f[k + 1]]))
    if tot < 0:
        return [list(reversed(f)) for f in faces], True
    return faces, False


def flip_uv_loops(faces, uvs):
    out = []
    i = 0
    for f in faces:
        n = len(f)
        out.extend(list(uvs[i:i + n])[::-1])
        i += n
    return np.asarray(out)


def rounded_octagon(inr, rc, m_flat=8, m_corner=5):
    """Perimeter points of an octagon (flats on the axes) with rounded
    corners, plus their arc-length parameter."""
    pts = []
    for k in range(8):
        a = k * np.pi / 4            # flat normal angle (0 = +Y side)
        nrm = np.array([np.cos(a), np.sin(a)])
        tan = np.array([-np.sin(a), np.cos(a)])
        half = inr * np.tan(np.pi / 8) - rc * np.tan(np.pi / 8)
        for t in np.linspace(-half, half, m_flat, endpoint=False):
            pts.append(nrm * inr + tan * t)
        # corner arc between this flat and the next
        cc = (nrm * (inr - rc) + tan * half)
        for t in np.linspace(0, np.pi / 4, m_corner, endpoint=False):
            ang = a + t
            pts.append(cc + rc * np.array([np.cos(ang), np.sin(ang)]))
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    return pts, s / s[-1]


# ================================================================ barrel ===
BARREL_UV_LEN = 268.0


def barrel_octagon():
    """Octagonal breech section X 0..194 with octagon UVs (u along X)."""
    ring, sp = rounded_octagon(P.BARREL_INRADIUS, 0.9)
    n = len(ring)
    # v = k/8 at the centre of flat k (the engraving is laid out that way);
    # rounded_octagon starts its arc length at the corner before flat 0
    sp = np.append(sp, 1.0) - sp[4]
    xs = np.linspace(0.0, 194.5, 40)
    v = []
    for x in xs:
        # ring coordinates: (a=y, b=z) with flat 0 at +Y; rotate so a flat
        # is on top (+Z): angle 0 -> +Y side flat, pi/2 -> top flat
        v.append(np.stack([np.full(n, x), ring[:, 0], ring[:, 1]], 1))
    v = np.vstack(v)
    faces, uvs = [], []
    for i in range(len(xs) - 1):
        for j in range(n):
            j1 = (j + 1) % n
            faces.append([i * n + j, i * n + j1, (i + 1) * n + j1,
                          (i + 1) * n + j])
            ua, ub = xs[i] / BARREL_UV_LEN, xs[i + 1] / BARREL_UV_LEN
            va, vb = sp[j], sp[j + 1]
            uvs += [(ua, va), (ua, vb), (ub, vb), (ub, va)]
    # breech face and front face (fans)
    c0 = len(v)
    v = np.vstack([v, [[0.0, 0, 0], [194.5, 0, 0]]])
    b = (len(xs) - 1) * n
    for j in range(n):
        j1 = (j + 1) % n
        faces.append([c0, j1, j])
        uvs += [(0.0, 0.0)] * 3
        faces.append([c0 + 1, b + j, b + j1])
        uvs += [(194.5 / BARREL_UV_LEN, 0.0)] * 3
    faces, flipped = fix_winding(v, faces)
    uvs = np.asarray(uvs)
    if flipped:
        uvs = flip_uv_loops(faces, uvs)
    return v, faces, uvs


def bore_radius(x):
    t = np.clip((x - 194.0) / (P.MUZZLE_X - 194.0), 0, 1)
    return P.BORE_RADIUS_BREECH + (P.BORE_RADIUS_MUZZLE -
                                   P.BORE_RADIUS_BREECH) * t ** 2.4


def muzzle():
    """Ring moulding + bell + mouth lip + trumpet bore (one closed lathe)."""
    outer = [list(p) for p in P.MUZZLE_PROFILE]
    lip = [(268.0, 21.2), (267.8, 20.3), (267.2, 19.7), (266.2, 19.35)]
    xs = np.linspace(265.0, 194.0, 30)
    inner = [(x, float(bore_radius(x)) - 0.12 * (x > 260) * 0) for x in xs]
    prof = outer + lip + inner + [(194.0, 12.0), (194.0, 16.1)]
    # u = angle; v along the profile.  Map into the barrel texture strip
    v, f, uv = lathe(prof, n=128, uv_rect=(0, 0, 1, 1), phase=np.pi / 2)
    # remap uvs: u <- X position (matches octagon u), v <- angle
    uvx = []
    i = 0
    flat = [p for p in f]
    for face in flat:
        for idx in face:
            uvx.append((v[idx, 0] / BARREL_UV_LEN,
                        (np.arctan2(v[idx, 2], v[idx, 1]) / (2 * np.pi)) % 1.0))
        i += 1
    uvx = np.asarray(uvx)
    # fix wrap-around seams within each face
    k = 0
    for face in flat:
        n = len(face)
        us = uvx[k:k + n, 1]
        if us.max() - us.min() > 0.5:
            us[us < 0.5] += 1.0
            uvx[k:k + n, 1] = us
        k += n
    faces, flipped = fix_winding(v, flat)
    if flipped:
        uvx = flip_uv_loops(faces, uvx)
    return v, faces, uvx


# =========================================================== ramrod, pipes ==
RAMROD_Z = (P.Z0 - P.RAMROD_Z_PX) * K
RAMROD_R = 2.75


def ramrod():
    head = [((x - P.X0) * K, h * K) for x, h in P.RAMROD_HEAD]
    prof = [(-16.0, 0.0), (-16.0, 1.8), (-15.6, 2.5), (-15.0, RAMROD_R)]
    prof += [(head[0][0] - 2.0, RAMROD_R)]
    prof += [(head[0][0], 2.85), (head[1][0], 3.4), (head[2][0], 4.9),
             (head[3][0] - 0.6, 6.25), (head[3][0], 6.45),
             (head[4][0], 6.2), (head[4][0] + 0.35, 4.0),
             (head[4][0] + 0.45, 0.0)]
    v, f, uv = lathe(prof, n=48, origin=(0, 0, RAMROD_Z))
    f, fl = fix_winding(v, f)
    if fl:
        uv = flip_uv_loops(f, uv)
    return v, f, uv


def ramrod_pipe(x0, x1, entry=False):
    r_o, r_i = 4.55, RAMROD_R + 0.2
    prof = [(x0, r_i + 0.4), (x0, 5.0), (x0 + 1.4, 5.05), (x0 + 1.8, r_o)]
    if entry:
        prof = [(x0, r_i + 0.4), (x0, 5.3), (x0 + 1.0, 5.4), (x0 + 1.6, 4.9),
                (x0 + 2.4, r_o)]
    prof += [(x1 - 2.6, r_o), (x1 - 2.2, 5.05), (x1 - 1.4, 5.1),
             (x1 - 1.0, 4.85), (x1 - 0.6, 5.05), (x1, 5.0), (x1, r_i + 0.4),
             (x1 - 0.4, r_i), (x0 + 0.4, r_i), (x0, r_i + 0.4)]
    v, f, uv = lathe(prof, n=48, origin=(0, 0, RAMROD_Z))
    f, fl = fix_winding(v, f)
    if fl:
        uv = flip_uv_loops(f, uv)
    return v, f, uv


PIPES = [((a - P.X0) * K, (b - P.X0) * K) for a, b in P.RAMROD_PIPES]


# ======================================================== SDF part fields ==
def field_class(fn, needs_stock=False, recess=True):
    class F:
        def __init__(self, grid):
            self.stock = ST.StockField(grid, recess) if needs_stock else None

        def eval_points(self, X, Y, Z, ix, iz):
            if needs_stock:
                return fn(X, Y, Z, self.stock.eval_points(X, Y, Z, ix, iz))
            return fn(X, Y, Z).astype(np.float32)
    return F


def sdf_mesh(fn, lo, hi, h, needs_stock=False, factor=4, band=None,
             recess=True):
    cls = field_class(fn, needs_stock, recess)
    vol, grid = S.evaluate_narrowband(cls, lo, hi, h, factor=factor,
                                      band=band)
    return S.mesh_volume(vol, grid)


def profile_slab(X, Y, Z, poly_mm, y0, y1, r, dome=0.0):
    """Side-profile polygon extruded between y0..y1 with rounded edges."""
    d2 = S.polygon_sdf2(X, Z, poly_mm)
    yc = 0.5 * (y0 + y1)
    hw = 0.5 * abs(y1 - y0)
    if dome:
        hw = hw + dome * S.smoothstep(0.0, -6.0, d2)
    return S.rounded_extrusion(d2, np.abs(Y - yc) - hw, r)


def chain(X, Y, Z, pts, radii, flat_axis=None, flat=1.0):
    f = None
    for i in range(len(pts) - 1):
        a, b = np.asarray(pts[i], float), np.asarray(pts[i + 1], float)
        if flat_axis is None or flat == 1.0:
            g = S.round_cone(X, Y, Z, a, b, radii[i], radii[i + 1])
        else:
            g = flat_cone(X, Y, Z, a, b, radii[i], radii[i + 1], flat_axis,
                          flat)
        f = g if f is None else np.minimum(f, g)
    return f


def flat_cone(x, y, z, a, b, ra, rb, thin_axis, thin):
    t = np.asarray(thin_axis, float)
    t = t / np.linalg.norm(t)
    c = (np.asarray(a, float) + np.asarray(b, float)) * 0.5
    px_, py_, pz_ = x - c[0], y - c[1], z - c[2]
    dt = px_ * t[0] + py_ * t[1] + pz_ * t[2]
    k = thin - 1.0
    qx, qy, qz = px_ + dt * t[0] * k, py_ + dt * t[1] * k, pz_ + dt * t[2] * k
    return S.round_cone(qx, qy, qz, np.asarray(a) - c, np.asarray(b) - c,
                        ra, rb) / (0.5 * (1.0 + thin))


def revolve_sdf(X, Y, Z, origin, axis, prof_poly):
    """SDF of a solid of revolution; prof_poly is a closed (axial, radius)
    polygon (radius >= 0)."""
    o = np.asarray(origin, float)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    px_, py_, pz_ = X - o[0], Y - o[1], Z - o[2]
    ax = px_ * a[0] + py_ * a[1] + pz_ * a[2]
    rx = px_ - ax * a[0]
    ry = py_ - ax * a[1]
    rz = pz_ - ax * a[2]
    rad = np.sqrt(rx * rx + ry * ry + rz * rz)
    return S.polygon_sdf2(ax, rad, prof_poly)


# ---------------------------------------------------------- lock plate ----
LOCK_OUT = -23.15       # outer face of the lock plate (Y)
LOCK_IN = -20.35


def lock_plate_fn(X, Y, Z):
    poly = pxs(P.catmull_rom(P.LOCK_PLATE, 6, closed=True))
    f = profile_slab(X, Y, Z, poly, LOCK_IN, LOCK_OUT + 0.25, 0.9, dome=0.25)
    # tumbler boss under the cock
    cx, cz = px(*P.COCK_PIVOT[:2])
    boss = S.round_cone(X, Y, Z, (cx, LOCK_OUT + 0.5, cz),
                        (cx, LOCK_OUT - 0.9, cz), 9.6, 9.0)
    f = S.smin(f, boss, 0.8)
    # bevelled border line (engraved) handled in texture
    return f


# ------------------------------------------------------- frizzen spring ---
def frizzen_spring_fn(X, Y, Z):
    y = LOCK_OUT - 1.15
    pts = [(54.0, y, -21.4), (66.0, y, -21.6), (80.0, y, -20.6),
           (90.5, y, -18.6), (95.0, y, -15.6), (93.6, y, -12.4),
           (87.5, y, -10.4), (80.5, y, -9.2), (75.5, y, -8.4)]
    rad = [3.6, 3.0, 2.7, 2.5, 2.3, 2.1, 1.9, 1.7, 1.5]
    f = chain(X, Y, Z, pts, rad, flat_axis=(0, 1, 0), flat=2.2)
    sx, sz = px(*P.FRIZZEN_SPRING_SCREW[:2])
    boss = S.round_cone(X, Y, Z, (sx, LOCK_OUT + 0.4, sz),
                        (sx, LOCK_OUT - 1.9, sz), 4.6, 4.3)
    return S.smin(f, boss, 1.2)


# ------------------------------------------------------------------ pan ---
PAN_Y = -21.5           # centre line of pan / frizzen / jaws / flint


def pan_fn(X, Y, Z):
    # body: side profile slab between the barrel flat and outside
    prof = pxs([(447.5, 470.0), (447.5, 463.0), (452, 458.6), (458, 457.0),
                (505, 459.0), (511, 462.6), (511, 470.0)])
    body = profile_slab(X, Y, Z, prof, -15.6, -27.6, 0.8)
    # bowl: rounded trough along X
    bowl = S.capsule(X, Y, Z, (33.0, PAN_Y, 7.9), (62.0, PAN_Y, 7.4), 4.0)
    body = S.smax(body, -bowl, 0.6)
    # bridle carrying the frizzen screw
    bx, bz = px(*P.FRIZZEN_PIVOT[:2])
    bridle = S.round_cone(X, Y, Z, (bx - 4.0, -27.0, bz - 0.6),
                          (bx + 1.0, -27.4, bz), 3.0, 3.6)
    bridle = np.maximum(bridle, -27.2 - Y - 1.2)
    return S.smin(body, bridle, 1.0)


# -------------------------------------------------------------- frizzen ---
def frizzen_fn(X, Y, Z):
    sp = pxs(P.catmull_rom(P.FRIZZEN_SPINE, 4))
    th = np.interp(np.linspace(0, 1, len(sp)),
                   np.linspace(0, 1, len(P.FRIZZEN_THICK)),
                   np.array(P.FRIZZEN_THICK) * K * 0.5)
    pts = [(x, PAN_Y, z) for x, z in sp]
    # wide in Y (11 mm), thin in-plane: model as a swept rounded rectangle
    f = None
    for i in range(len(pts) - 1):
        a, b = np.array(pts[i]), np.array(pts[i + 1])
        seg = S.capsule(X, PAN_Y + 0 * Y, Z, a, b, 0.0)  # in-plane distance
        r = 0.5 * (th[i] + th[i + 1])
        g = S.rounded_extrusion(seg - r, np.abs(Y - PAN_Y) - 5.6, 0.7)
        f = g if f is None else np.minimum(f, g)
    # pivot knuckle + toe resting on the spring
    bx, bz = px(*P.FRIZZEN_PIVOT[:2])
    knuckle = S.round_cone(X, Y, Z, (bx, -16.4, bz), (bx, -26.6, bz), 3.2, 3.2)
    toe = S.round_cone(X, Y, Z, (bx, PAN_Y, bz), (bx + 6.0, PAN_Y, bz - 4.8),
                       2.6, 1.8)
    toe = np.maximum(toe, np.abs(Y - PAN_Y) - 4.6)
    return np.minimum(S.smin(f, knuckle, 1.2), toe)


# ----------------------------------------------------------------- cock ---
COCK_BODY_Y = (-23.25, -31.0)


def cock_fn(X, Y, Z):
    # bulbous breast: domed outer face, generously rounded edges
    body = profile_slab(X, Y, Z, pxs(P.catmull_rom(P.COCK_BODY, 5,
                                                   closed=True)),
                        COCK_BODY_Y[0], COCK_BODY_Y[1] + 1.2, 3.4, dome=1.6)
    spine = pxs(P.catmull_rom(P.COCK_NECK_SPINE, 4))
    wid = np.interp(np.linspace(0, 1, len(spine)),
                    np.linspace(0, 1, len(P.COCK_NECK_WIDTH)),
                    np.array(P.COCK_NECK_WIDTH) * K * 0.5)
    yc0 = 0.5 * sum(COCK_BODY_Y)
    ys = np.interp(np.linspace(0, 1, len(spine)), [0, 0.35, 1.0],
                   [yc0, yc0 + 1.0, PAN_Y])
    pts = [(x, y, z) for (x, z), y in zip(spine, ys)]
    neck = chain(X, Y, Z, pts, list(wid), flat_axis=(0, 1, 0), flat=1.15)
    f = S.smin(body, neck, 1.6)
    lj = profile_slab(X, Y, Z, pxs(P.COCK_LOWER_JAW), PAN_Y - 6.0,
                      PAN_Y + 6.0, 0.8)
    f = S.smin(f, lj, 1.0)
    tj = profile_slab(X, Y, Z, pxs(P.COCK_TOP_JAW), PAN_Y - 5.7, PAN_Y + 5.7,
                      0.9)
    (c0x, c0z), (c1x, c1z) = pxs(P.COCK_COMB)
    comb = S.round_cone(X, Y, Z, (c0x, PAN_Y, c0z), (c1x, PAN_Y, c1z), 1.4,
                        1.8)
    bx, bz = px(P.COCK_COMB_BALL[0], P.COCK_COMB_BALL[1])
    ball = S.sphere(X, Y, Z, (bx, PAN_Y, bz), P.COCK_COMB_BALL[2] * K * 0.5)
    f = np.minimum(f, S.smin(S.smin(comb, ball, 0.6), lj, 0.8))
    f = np.minimum(f, tj)
    # small hook under the lower jaw (decorative)
    hk = S.round_cone(X, Y, Z, (px(406, 414)[0], PAN_Y, px(406, 414)[1]),
                      (px(409, 423)[0], PAN_Y, px(409, 423)[1]), 1.6, 0.8)
    f = S.smin(f, hk, 0.6)
    return f


def jaw_screw_fn(X, Y, Z):
    x0 = (P.COCK_JAW_SCREW[0][0] - P.X0) * K
    prof = [(78.5, 0.0), (78.3, 1.6), (77.4, 3.0), (75.6, 4.4), (74.2, 4.6),
            (72.6, 3.9), (71.2, 2.2), (70.4, 1.8), (69.6, 3.2), (68.6, 3.8),
            (65.2, 3.8), (64.4, 3.1), (63.6, 2.35), (30.6, 2.35), (29.6, 1.6),
            (29.2, 0.0)]
    return revolve_sdf(X, Y, Z, (x0, PAN_Y, 0.0), (0, 0, 1), prof)


def flint_fn(X, Y, Z):
    fl = profile_slab(X, Y, Z, pxs(P.FLINT), PAN_Y - 4.6, PAN_Y + 4.6, 0.35)
    # knapped edge facets
    return fl


def leather_fn(X, Y, Z):
    a = px(401.0, 398.6)
    b = px(412.5, 411.8)
    box = S.rounded_extrusion(
        np.maximum(np.maximum(a[0] - X, X - b[0]),
                   np.maximum(b[1] - Z, Z - a[1])), np.abs(Y - PAN_Y) - 5.1,
        0.4)
    inner = np.maximum(np.maximum(a[0] - 0.5 - X, X - b[0] - 0.5),
                       np.abs(Z - 0.5 * (a[1] + b[1])) - 0.5 * abs(a[1] - b[1])
                       + 0.55)
    return np.maximum(box, -inner)


# ------------------------------------------------------- trigger, guard ---
def sweep_planar(X, Y, Z, path_mm, half_in, half_y, r, y0=0.0):
    """Swept rounded rectangle along a side-view path."""
    f = None
    for i in range(len(path_mm) - 1):
        a = (path_mm[i][0], y0, path_mm[i][1])
        b = (path_mm[i + 1][0], y0, path_mm[i + 1][1])
        d = S.capsule(X, y0 + 0 * Y, Z, a, b, 0.0)
        hi_ = 0.5 * (half_in[i] + half_in[i + 1])
        g = S.rounded_extrusion(d - hi_, np.abs(Y - y0) - half_y, r)
        f = g if f is None else np.minimum(f, g)
    return f


def trigger_fn(X, Y, Z):
    path = pxs(P.catmull_rom([(370.8, 516.0)] + P.TRIGGER, 4))
    w = np.interp(np.linspace(0, 1, len(path)),
                  np.linspace(0, 1, len(P.TRIGGER_WIDTH) + 1),
                  [P.TRIGGER_WIDTH[0]] + P.TRIGGER_WIDTH) * K * 0.5
    return sweep_planar(X, Y, Z, path, list(w), 2.3, 0.8)


def guard_fn(X, Y, Z, fs):
    path = pxs(P.catmull_rom(P.GUARD_BOW, 5))
    w = [P.GUARD_WIRE * K * 0.5] * len(path)
    # bow: slightly oval wire, wider at the bottom
    bow = sweep_planar(X, Y, Z, path, w, 2.6, 1.2)
    # front tang hugging the stock bottom + finial
    tp = pxs(P.catmull_rom(P.GUARD_FRONT_TANG, 5))
    d, par, tot = S.polyline_dist2(X, Z, tp)
    tang = np.maximum(np.maximum(fs - 1.45, -(fs + 0.35)),
                      np.maximum(np.abs(Y) - (3.6 + 0.9 * S.smoothstep(
                          tot - 14, tot - 2, par)), d - 5.0))
    tang = np.maximum(tang, np.maximum(-6.5 - X, X - 106.0))
    fin = S.ellipsoid(X, Y, Z, (101.5, 0.0, -36.2), (4.8, 4.6, 2.0))
    tang = np.minimum(tang, np.maximum(fin, -(fs + 0.35)))
    f = S.smin(bow, tang, 1.2)
    # rear: bow joins the grip strap; little rear finial scroll
    return f


def strap_fn(X, Y, Z, fs):
    path = pxs(P.catmull_rom(P.GRIP_STRAP, 5))
    d, par, tot = S.polyline_dist2(X, Z, path)
    t = par / tot
    hw = 3.4 + 3.2 * S.smoothstep(0.15, 0.85, t)
    f = np.maximum(np.maximum(fs - 1.25, -(fs + 0.45)),
                   np.maximum(np.abs(Y) - hw, d - 6.0))
    # stop where it joins the guard bow (top) and into the cap (bottom)
    a = path[0]
    f = np.maximum(f, -((X - a[0]) * 0.7 + (Z - a[1]) * -0.7) - 1.0)
    return f


def tang_fn(X, Y, Z, fs_norecess):
    plan = ST.tang_plan(X, Y)
    f = np.maximum(np.maximum(fs_norecess - 0.25, -(fs_norecess + 1.1)),
                   plan)
    return f


# ----------------------------------------------------------- side plate ---
def _spiral(c, r0, turns, start_ang, sense=1, n=18, shrink=0.55):
    out = []
    for i in range(n):
        t = i / (n - 1)
        r = r0 * (1 - shrink * t)
        a = start_ang + sense * 2 * np.pi * turns * t
        out.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
    return out


def side_plate_strokes():
    """Dragon side plate (left side): list of (pts p4-px, half widths px)."""
    S_ = []
    # body: head end -> loop round the rear side nail -> tail
    body = [(648, 658), (657, 651.5), (668, 648.5), (680, 650), (691, 655.5),
            (702, 661.5), (714, 666), (726, 667.2), (737, 664.5), (746, 659),
            (753, 652), (758.5, 645), (764, 639.5), (771, 638), (777, 641.5),
            (779, 648), (776, 654), (770, 657.5), (777.5, 660.5),
            (788, 662.5), (798, 661.5), (805, 657.5), (807.5, 652.5),
            (804.5, 649.5), (801, 651.5)]
    bw = np.interp(np.arange(len(body)), [0, 4, 9, 13, 18, 24],
                   [4.6, 4.4, 3.9, 3.0, 2.2, 1.2])
    S_.append((body, bw))
    # head (snout pointing at the front nail), open jaws
    S_.append(([(650, 657), (642, 654.5), (633, 654.5), (626, 656.5)],
               [4.2, 3.6, 2.6, 1.3]))
    S_.append(([(647, 662), (639, 664), (632, 666.5), (627, 669.5)],
               [2.6, 2.2, 1.6, 0.9]))
    S_.append(([(646, 652), (650, 646.5), (656, 643.5)], [1.6, 1.4, 0.8]))
    # mane / fins: curls above and below the body
    for c, r, ang, sense in [((690, 645.5), 5.0, -0.4, -1),
                             ((711, 650.5), 4.2, -0.2, -1),
                             ((737, 649.0), 4.6, 0.1, -1),
                             ((663, 669.5), 4.6, 3.0, 1),
                             ((683, 668.5), 4.0, 2.9, 1),
                             ((757, 667.0), 4.0, 3.4, 1),
                             ((792, 650.5), 4.0, -0.6, -1)]:
        sp = _spiral(c, r, 1.05, ang, sense)
        # grow the curl out of the body: stem from the nearest body point
        bpts = np.asarray(body, float)
        j = int(np.argmin(np.hypot(bpts[:, 0] - sp[0][0],
                                   bpts[:, 1] - sp[0][1])))
        root = tuple(bpts[j])
        mid = (0.5 * (root[0] + sp[0][0]), 0.5 * (root[1] + sp[0][1]))
        sp = [root, mid] + list(sp)
        w = [2.6, 2.2] + list(np.linspace(1.9, 0.9, len(sp) - 2))
        S_.append((sp, w))
    # legs / claws
    S_.append(([(700, 662), (697, 669), (692, 673), (687, 673.5)],
               [1.8, 1.5, 1.1, 0.7]))
    S_.append(([(731, 666), (735, 672), (741, 674.5), (746, 673.5)],
               [1.7, 1.4, 1.0, 0.7]))
    # escutcheon ring round the front side nail
    ring = _spiral((614, 666.5), 9.4, 1.0, 0.0, 1, n=40, shrink=0.0)
    S_.append((ring, [1.5] * len(ring)))
    return S_


SIDE_Y = None


def side_plate_fn(X, Y, Z):
    """Dragon side plate: flat-topped silver ribbons (rounded edges) standing
    proud of the left lock panel, with a shallow spine groove."""
    y0 = SIDE_Y
    top = y0 + 1.7                      # relief surface
    f = np.full(np.shape(X), 8.0, np.float64)
    for pts, w in SIDE_STROKES_MM:
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            r = max(w[i], w[i + 1]) + 1.2
            m = ((X > min(a[0], b[0]) - r) & (X < max(a[0], b[0]) + r) &
                 (Z > min(a[1], b[1]) - r) & (Z < max(a[1], b[1]) + r))
            if not np.any(m):
                continue
            # in-plane distance to the segment, varying half width
            ax, az = a
            ex, ez = b[0] - a[0], b[1] - a[1]
            ee = max(ex * ex + ez * ez, 1e-12)
            t = np.clip(((X[m] - ax) * ex + (Z[m] - az) * ez) / ee, 0, 1)
            d = np.hypot(X[m] - ax - ex * t, Z[m] - az - ez * t)
            hw = w[i] + (w[i + 1] - w[i]) * t
            # slightly domed top: thinner strokes stand a little lower
            ht = np.minimum(1.0, 0.35 + 0.4 * hw)
            g = S.rounded_extrusion(d - hw, np.abs(Y[m] - (y0 + top) * 0.5 -
                                                   0.0) -
                                    (0.5 * (top - y0) * ht + 0.25), 0.45)
            f[m] = S.smin(f[m], g, 0.2)
    return np.maximum(f, y0 - 0.6 - Y)


SIDE_STROKES_MM = None


def _prep_side_plate():
    global SIDE_STROKES_MM, SIDE_Y
    out = []
    for pts, w in side_plate_strokes():
        p3 = P.left_view_to_right(pts)
        out.append((pxs(P.catmull_rom(p3, 3)),
                    np.interp(np.linspace(0, 1, (len(pts) - 1) * 3 + 1),
                              np.linspace(0, 1, len(w)), np.asarray(w) * K)))
    SIDE_STROKES_MM = out
    SIDE_Y = 20.85


_prep_side_plate()


# --------------------------------------------------------------- screws ---
def screw_fn(X, Y, Z, c, axis, r, h=None, slot_ang=0.4, dome=0.55):
    """Domed slotted screw head sitting on a surface at c, axis outward."""
    h = h or r * dome
    prof = [(-1.2, 0.0), (-1.2, r * 0.92), (0.0, r), (h * 0.35, r * 0.96),
            (h * 0.7, r * 0.78), (h * 0.92, r * 0.48), (h, 0.0)]
    f = revolve_sdf(X, Y, Z, c, axis, prof)
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    up = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, up)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    du = np.cos(slot_ang) * u + np.sin(slot_ang) * v
    dv = np.cross(a, du)
    px_, py_, pz_ = X - c[0], Y - c[1], Z - c[2]
    along = px_ * a[0] + py_ * a[1] + pz_ * a[2]
    across = px_ * dv[0] + py_ * dv[1] + pz_ * dv[2]
    slot = np.maximum(np.abs(across) - r * 0.13, h * 0.32 - along)
    return S.smax(f, -slot, 0.08)


def screws():
    """(name, centre, axis, head radius, material)"""
    out = []
    lx, lz = px(*P.LOCK_SCREW_REAR[:2])
    out.append(('Screw_LockTail', (lx, LOCK_OUT - 0.1, lz), (0, -1, 0), 2.9,
                'silver'))
    sx, sz = px(*P.FRIZZEN_SPRING_SCREW[:2])
    out.append(('Screw_FrizzenSpring', (sx, LOCK_OUT - 2.2, sz), (0, -1, 0),
                3.2, 'silver'))
    bx, bz = px(*P.FRIZZEN_PIVOT[:2])
    out.append(('Screw_FrizzenPivot', (bx, -28.2, bz), (0, -1, 0), 2.5,
                'brass'))
    cx, cz = px(*P.COCK_PIVOT[:2])
    out.append(('Screw_Cock', (cx, COCK_BODY_Y[1] + 0.05, cz), (0, -1, 0),
                7.6, 'silver'))
    fx, fz = P.to_mm([P.SIDE_SCREW_FRONT])[0]
    out.append(('Screw_SideNailFront', (fx, SIDE_Y + 0.5, fz), (0, 1, 0),
                3.4, 'silver'))
    rx, rz = P.to_mm([P.SIDE_SCREW_REAR])[0]
    out.append(('Screw_SideNailRear', (rx, SIDE_Y + 0.5, rz), (0, 1, 0),
                3.4, 'silver'))
    tz = float(ST.wrist_top_z(np.array([-34.0]))[0])
    out.append(('Screw_Tang', (-34.0, 0.0, tz + 0.15), (0, 0, 1), 3.1,
                'brass'))
    sb = pxs(P.STOCK_BOTTOM)
    order = np.argsort(sb[:, 0])
    for gx in (25.6, 62.0):
        gz = float(np.interp(gx, sb[order, 0], sb[order, 1])) - 1.3
        out.append(('Pin_Guard_%d' % int(gx), (gx, 0.0, gz), (0, 0, -1), 1.5,
                    'silver'))
    return out
