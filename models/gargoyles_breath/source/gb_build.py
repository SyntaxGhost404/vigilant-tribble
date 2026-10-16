"""Generate every part of The Gargoyle's Breath and assemble the Blender scene.

usage:  python3 gb_build.py <quality: draft|final> <cache_dir> <out.blend>
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_profiles as P            # noqa: E402
import gb_sdf as S                 # noqa: E402
import gb_stock as ST              # noqa: E402
import gb_cap as CAP               # noqa: E402
import gb_hard as H                # noqa: E402

QUALITY = {
    'draft': dict(stock=0.5, cap=0.4, hard=0.25, fine=0.12, screw=0.08,
                  target=None),
    # target triangle budgets per part after decimation
    'final': dict(stock=0.25, cap=0.13, hard=0.1, fine=0.07, screw=0.05,
                  target={'Stock': 180000, 'ButtCap_Gargoyle': 460000,
                          'LockPlate': 30000, 'Cock': 42000,
                          'SidePlate': 80000, 'TriggerGuard': 30000,
                          'GripStrap': 12000, 'Frizzen': 12000,
                          'FrizzenSpring': 8000, 'Pan': 12000,
                          'Trigger': 6000, 'BarrelTang': 8000,
                          'Cock_JawScrew': 6000, 'Flint': 3000,
                          'Flint_Leather': 2500, 'Screw_Cock': 9000,
                          'screw': 4000, 'pin': 1500}),
}
QUALITY['draft']['target'] = {k: v // 3 for k, v in
                              QUALITY['final']['target'].items()}

# part name -> (material, parent, pivot (mm) or None)
PIVOT_COCK = tuple(P.to_mm([P.COCK_PIVOT[:2]])[0])
PIVOT_FRIZZEN = tuple(P.to_mm([P.FRIZZEN_PIVOT[:2]])[0])
PIVOT_TRIGGER = tuple(P.to_mm([(370.6, 521.0)])[0])


def tri(faces):
    out = []
    for q in faces:
        for k in range(1, len(q) - 1):
            out.append([q[0], q[k], q[k + 1]])
    return np.asarray(out, np.int64)


def cached(cache, name, fn):
    path = os.path.join(cache, name + '.npz')
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        uv = d['uv'] if 'uv' in d.files else None
        faces = d['f']
        if faces.dtype == object:
            faces = list(faces)
        return d['v'], faces, uv
    t = time.time()
    res = fn()
    v, f = res[0], res[1]
    uv = res[2] if len(res) > 2 else None
    kw = dict(v=v)
    if isinstance(f, list):
        kw['f'] = np.array(f, dtype=object)
    else:
        kw['f'] = f
    if uv is not None:
        kw['uv'] = uv
    np.savez(path, **kw)
    nf = len(f)
    print('  built %-22s %8d faces  %6.1fs' % (name, nf, time.time() - t),
          flush=True)
    return v, f, uv


def part_specs(q):
    hd, fn_, sc = q['hard'], q['fine'], q['screw']
    sm = H.sdf_mesh
    specs = [
        ('Stock', 'walnut', None, lambda: ST.build(q['stock'])),
        ('ButtCap_Gargoyle', 'silver', None, lambda: CAP.build(q['cap'])),
        ('Barrel', 'brass', None, H.barrel_octagon),
        ('Barrel_Muzzle', 'brass', None, H.muzzle),
        ('Ramrod', 'steel', None, H.ramrod),
        ('RamrodPipe_Rear', 'brass', None,
         lambda: H.ramrod_pipe(*H.PIPES[0], entry=True)),
        ('RamrodPipe_Front', 'brass', None,
         lambda: H.ramrod_pipe(*H.PIPES[1])),
        ('BarrelTang', 'brass', None,
         lambda: sm(H.tang_fn, (-56, -11, -4), (2, 11, 21), fn_,
                    needs_stock=True, recess=False)),
        ('LockPlate', 'silver', None,
         lambda: sm(H.lock_plate_fn, (-72, -24.9, -33), (108, -19.8, 4), hd)),
        ('FrizzenSpring', 'silver', None,
         lambda: sm(H.frizzen_spring_fn, (48, -27.5, -27), (100, -22.5, -4),
                    fn_)),
        ('Pan', 'silver', None,
         lambda: sm(H.pan_fn, (20, -31.5, -6), (84, -14.5, 12), hd)),
        ('Frizzen', 'steel', None,
         lambda: sm(H.frizzen_fn, (66, -28.5, -9), (110, -14.5, 49), hd)),
        ('Cock', 'silver', None,
         lambda: sm(H.cock_fn, (-30, -33.5, -30), (16, -14.5, 46), hd)),
        ('Cock_JawScrew', 'silver', 'Cock',
         lambda: sm(H.jaw_screw_fn, (-21, -27.5, 27), (-9, -15.5, 80), fn_)),
        ('Flint', 'flint', 'Cock',
         lambda: sm(H.flint_fn, (-11, -27, 40), (15, -16, 50), fn_)),
        ('Flint_Leather', 'leather', 'Cock',
         lambda: sm(H.leather_fn, (-12, -28, 39), (-1, -15, 51), fn_)),
        ('Trigger', 'steel', None,
         lambda: sm(H.trigger_fn, (-38, -4, -76), (-27, 4, -30), fn_)),
        ('TriggerGuard', 'silver', None,
         lambda: sm(H.guard_fn, (-63, -6, -82), (108, 6, -33), hd,
                    needs_stock=True)),
        ('GripStrap', 'silver', None,
         lambda: sm(H.strap_fn, (-128, -9, -122), (-52, 9, -38), hd,
                    needs_stock=True)),
        ('SidePlate', 'silver', None,
         lambda: sm(H.side_plate_fn, (-60, 19.5, -33), (103, 24.5, 0), fn_)),
    ]
    for name, c, axis, r, mat in H.screws():
        lo = np.array(c) - (r + 1.6)
        hi = np.array(c) + (r + 1.6)
        parent = 'Cock' if name == 'Screw_Cock' else None
        specs.append((name, mat, parent,
                      (lambda c=c, axis=axis, r=r, lo=lo, hi=hi:
                       sm(lambda X, Y, Z: H.screw_fn(X, Y, Z, c, axis, r),
                          tuple(lo), tuple(hi), sc))))
    return specs


PIVOTS = {
    'Cock': (PIVOT_COCK[0], H.COCK_BODY_Y[1], PIVOT_COCK[1]),
    'Frizzen': (PIVOT_FRIZZEN[0], H.PAN_Y, PIVOT_FRIZZEN[1]),
    'Trigger': (PIVOT_TRIGGER[0], 0.0, PIVOT_TRIGGER[1]),
}

# stock spine used for the wood-grain coordinates (mm)
GRAIN_SPINE = [(222, -13), (160, -14), (100, -17), (40, -16), (0, -12),
               (-35, -9), (-62, -16), (-86, -30), (-103, -50), (-117, -74),
               (-128, -96), (-136, -118)]


def grain_coords(v):
    d, par, tot = S.polyline_dist2(v[:, 0], v[:, 2], GRAIN_SPINE)
    # signed offset: + above the spine (toward the top line)
    sp = np.asarray(GRAIN_SPINE, float)
    seg = np.diff(sp, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    i = np.clip(np.searchsorted(cum, par) - 1, 0, len(seg) - 1)
    tx, tz = seg[i, 0], seg[i, 1]
    side = np.sign((v[:, 0] - sp[i, 0]) * tz - (v[:, 2] - sp[i, 1]) * tx)
    return np.stack([par, -side * d], 1)


def build_all(quality, cache):
    os.makedirs(cache, exist_ok=True)
    q = QUALITY[quality]
    out = {}
    for name, mat, parent, fn in part_specs(q):
        v, f, uv = cached(cache, name, fn)
        out[name] = (v, f, uv, mat, parent)
    return out


def assemble(parts, quality, blend_path):
    import bpy
    import gb_blender as B
    import gb_materials as M
    B.reset_scene()
    sc = bpy.context.scene
    root = bpy.data.objects.new('GargoylesBreath', None)
    sc.collection.objects.link(root)
    mats = M.create_all()
    objs = {}
    q = QUALITY[quality]
    order = sorted(parts.keys(), key=lambda n: parts[n][4] is not None)
    for name in order:
        v, f, uv, mat, parent = parts[name]
        if not isinstance(f, np.ndarray) or f.ndim != 2:
            faces = list(f)
        else:
            faces = f
        pivot = PIVOTS.get(name)
        # children are stored relative to their parent's pivot; the parent
        # has no rotation/scale so a plain offset is enough
        ppiv = PIVOTS.get(parent) if parent else None
        vv = v.copy()
        if pivot is not None:
            vv = vv - np.asarray(pivot)
        elif ppiv is not None:
            vv = vv - np.asarray(ppiv)
        ob = B.add_mesh(name, vv, faces, uvs=uv, smooth=True,
                        material=mats[mat])
        if uv is not None:
            ob.data.uv_layers[0].name = 'Engrave'
        if pivot is not None:
            ob.location = np.asarray(pivot) * B.MM
            if parent and ppiv is not None:
                ob.location = (np.asarray(pivot) - np.asarray(ppiv)) * B.MM
        par = objs.get(parent) if parent else None
        ob.parent = par if par is not None else root
        objs[name] = ob
        key = name
        if name.startswith('Screw') and name not in q['target']:
            key = 'screw'
        elif name.startswith('Pin'):
            key = 'pin'
        target = q['target'].get(key)
        if target:
            ntri = sum(len(p.vertices) - 2 for p in ob.data.polygons)
            if ntri > target:
                B.decimate(ob, target / ntri)
        if mat in ('brass', 'silver', 'steel') and name not in (
                'ButtCap_Gargoyle',):
            ob.data.set_sharp_from_angle(angle=np.radians(50.0))
    M.add_aux_uvs(objs, grain_coords)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    return objs


if __name__ == '__main__':
    quality = sys.argv[1] if len(sys.argv) > 1 else 'draft'
    cache = sys.argv[2] if len(sys.argv) > 2 else './cache_' + quality
    blend = sys.argv[3] if len(sys.argv) > 3 else './GargoylesBreath.blend'
    t = time.time()
    parts = build_all(quality, cache)
    print('geometry done %.1fs' % (time.time() - t), flush=True)
    assemble(parts, quality, os.path.abspath(blend))
    print('saved', blend, '%.1fs' % (time.time() - t))
