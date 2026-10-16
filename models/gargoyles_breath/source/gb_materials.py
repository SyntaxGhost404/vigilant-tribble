"""Procedural PBR materials (Cycles) for The Gargoyle's Breath.

Every material ends in a Principled BSDF named 'BSDF' whose Base Color,
Metallic and Roughness inputs are driven by node chains, and whose Normal
input comes from Bump nodes.  gb_bake.py bakes those inputs to textures.
"""
import os
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, '..', 'textures', 'source')

# side projection window for the lock-work engraving (mm)
PROJ_X = (-75.0, 125.0)
PROJ_Z = (-40.0, 80.0)


class G:
    """Tiny node-graph builder."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        for n in list(self.nt.nodes):
            self.nt.nodes.remove(n)
        self.x = 0

    def n(self, kind, inputs=None, **props):
        node = self.nt.nodes.new(kind)
        node.location = (self.x, 0)
        self.x += 40
        for k, v in props.items():
            setattr(node, k, v)
        for k, v in (inputs or {}).items():
            node.inputs[k].default_value = v
        return node

    def l(self, a, b):
        self.nt.links.new(a, b)

    # --- helpers returning sockets
    def math(self, op, a, b=None, clamp=False):
        m = self.n('ShaderNodeMath', operation=op, use_clamp=clamp)
        self._feed(m.inputs[0], a)
        if b is not None:
            self._feed(m.inputs[1], b)
        return m.outputs[0]

    def mixc(self, fac, a, b):
        m = self.n('ShaderNodeMix', data_type='RGBA', clamp_factor=True)
        self._feed(m.inputs[0], fac)
        self._feed(m.inputs[6], a)
        self._feed(m.inputs[7], b)
        return m.outputs[2]

    def mixf(self, fac, a, b):
        m = self.n('ShaderNodeMix', data_type='FLOAT', clamp_factor=True)
        self._feed(m.inputs[0], fac)
        self._feed(m.inputs[2], a)
        self._feed(m.inputs[3], b)
        return m.outputs[0]

    def maprange(self, x, a, b, c=0.0, d=1.0, clamp=True, smooth=False):
        m = self.n('ShaderNodeMapRange', clamp=clamp,
                   interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        self._feed(m.inputs['Value'], x)
        m.inputs['From Min'].default_value = a
        m.inputs['From Max'].default_value = b
        m.inputs['To Min'].default_value = c
        m.inputs['To Max'].default_value = d
        return m.outputs['Result']

    def noise(self, vec, scale, detail=4.0, rough=0.55, dist=0.0, dims='3D',
              out='Fac'):
        m = self.n('ShaderNodeTexNoise', noise_dimensions=dims)
        if vec is not None:
            self._feed(m.inputs['Vector'], vec)
        m.inputs['Scale'].default_value = scale
        m.inputs['Detail'].default_value = detail
        m.inputs['Roughness'].default_value = rough
        m.inputs['Distortion'].default_value = dist
        return m.outputs[out]

    def ao(self, dist, local=False):
        m = self.n('ShaderNodeAmbientOcclusion', samples=16, only_local=local)
        m.inputs['Distance'].default_value = dist
        return m.outputs['AO']

    def geo(self, out):
        return self.n('ShaderNodeNewGeometry').outputs[out]

    def uv(self, name):
        return self.n('ShaderNodeUVMap', uv_map=name).outputs['UV']

    def image(self, img, vec, interp='Cubic', ext='REPEAT'):
        m = self.n('ShaderNodeTexImage', image=img, interpolation=interp,
                   extension=ext)
        self._feed(m.inputs['Vector'], vec)
        return m.outputs['Color']

    def bump(self, height, strength, dist, normal=None):
        m = self.n('ShaderNodeBump')
        m.inputs['Strength'].default_value = strength
        m.inputs['Distance'].default_value = dist
        self._feed(m.inputs['Height'], height)
        if normal is not None:
            self._feed(m.inputs['Normal'], normal)
        return m.outputs['Normal']

    def sep(self, vec):
        m = self.n('ShaderNodeSeparateXYZ')
        self._feed(m.inputs[0], vec)
        return m.outputs

    def comb(self, x, y, z):
        m = self.n('ShaderNodeCombineXYZ')
        for i, v in enumerate((x, y, z)):
            self._feed(m.inputs[i], v)
        return m.outputs[0]

    def vscale(self, vec, s):
        m = self.n('ShaderNodeVectorMath', operation='SCALE')
        self._feed(m.inputs[0], vec)
        m.inputs['Scale'].default_value = s
        return m.outputs[0]

    def _feed(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.l(val, sock)
        elif isinstance(val, (tuple, list)) and len(val) in (3, 4) and \
                sock.type == 'RGBA':
            sock.default_value = tuple(val) + ((1.0,) if len(val) == 3 else ())
        else:
            sock.default_value = val

    def bsdf(self, base, metal, rough, normal, coat=None):
        b = self.n('ShaderNodeBsdfPrincipled')
        b.name = 'BSDF'
        self._feed(b.inputs['Base Color'], base)
        self._feed(b.inputs['Metallic'], metal)
        self._feed(b.inputs['Roughness'], rough)
        if normal is not None:
            self.l(normal, b.inputs['Normal'])
        if coat is not None:
            b.inputs['Coat Weight'].default_value = coat
            b.inputs['Coat Roughness'].default_value = 0.35
        o = self.n('ShaderNodeOutputMaterial')
        self.l(b.outputs['BSDF'], o.inputs['Surface'])
        return b


def _img(name, colorspace='Non-Color'):
    path = os.path.join(TEX, name)
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    return img


def obj_mm(g):
    """Object-space position in millimetres (texture coordinates)."""
    tc = g.n('ShaderNodeTexCoord')
    return g.vscale(tc.outputs['Object'], 1000.0)


# ------------------------------------------------------------- metals -----
def _lookup(g, kind):
    """Engraving lookups -> (height 0..1 [1 = high], grime 0..1) or a mask."""
    if kind == 'lock':
        uv = g.uv('Proj')
        h = g.sep(g.image(_img('engrave_lock_h.png'), uv, ext='EXTEND'))[0]
        gr = g.sep(g.image(_img('engrave_lock_g.png'), uv, ext='CLIP'))[0]
        return h, gr
    if kind == 'inlay':
        uv = g.uv('Proj')
        return g.sep(g.image(_img('inlay_lock.png'), uv, ext='CLIP'))[0]
    if kind == 'barrel':
        uv = g.uv('Engrave')
        h = g.sep(g.image(_img('engrave_barrel_h.png'), uv))[0]
        gr = g.sep(g.image(_img('engrave_barrel_g.png'), uv))[0]
        return h, gr
    raise ValueError(kind)


def metal(name, albedo, tarnish, polish_gain, engrave=None, inlay=None,
          rough=(0.2, 0.55), ao_dist=0.004, micro=1.0, relief_depth=0.00018,
          patina=0.32, grime_gain=0.75):
    """Aged metal: cavity tarnish (AO), polished edges (pointiness),
    blotchy patina, optional engraving height + gold inlay mask."""
    mat = bpy.data.materials.new(name)
    g = G(mat)
    pos = obj_mm(g)
    ao = g.ao(ao_dist)
    cav = g.math('POWER', g.math('SUBTRACT', 1.0, ao, clamp=True), 0.8)
    edge = g.maprange(g.geo('Pointiness'), 0.5, 0.62, 0.0, 1.0)
    blot = g.maprange(g.noise(pos, 0.18, 6.0, 0.62), 0.35, 0.75, 0.0, 1.0,
                      smooth=True)
    fine = g.noise(pos, 2.2, 8.0, 0.7)
    t = g.math('ADD', g.math('MULTIPLY', cav, 1.5),
               g.math('MULTIPLY', blot, patina))
    eh, grime = _lookup(g, engrave) if engrave else (None, None)
    ink = _lookup(g, inlay) if inlay else None
    if eh is not None:
        # recessed ground / grooves hold grime, raised relief stays bright
        t = g.math('ADD', t, g.math('MULTIPLY', grime, grime_gain))
    t = g.math('SUBTRACT', t, g.math('MULTIPLY', edge, 0.5), clamp=True)
    tt = g.math('MULTIPLY', t, g.maprange(fine, 0.3, 0.7, 0.85, 1.1))
    base = g.mixc(tt, albedo, tarnish)
    base = g.mixc(g.math('MULTIPLY', edge, polish_gain), base,
                  tuple(min(1.0, c * 1.12) for c in albedo))
    if ink is not None:
        gold = g.mixc(tt, (0.86, 0.63, 0.27), (0.12, 0.085, 0.03))
        base = g.mixc(ink, base, gold)
    r0, r1 = rough
    rough_s = g.mixf(tt, r0, r1)
    rough_s = g.math('ADD', rough_s, g.math(
        'MULTIPLY', g.math('SUBTRACT', fine, 0.5), 0.12))
    metal_s = g.mixf(g.math('MULTIPLY', tt, 0.6), 1.0, 0.55)
    sc = g.noise(pos, 9.0, 2.0, 0.5, dist=0.6)
    nrm = g.bump(g.math('MULTIPLY', sc, 0.6), 0.08 * micro, 0.0004)
    if eh is not None:
        nrm = g.bump(eh, 1.0, relief_depth, nrm)
    g.bsdf(base, metal_s, rough_s, nrm)
    return mat


# --------------------------------------------------------------- walnut ---
def walnut():
    """Oiled walnut: fine straight grain following the stock's curve, dark
    figure streaks, open pores, scratches, grime in recesses, worn edges."""
    mat = bpy.data.materials.new('Walnut')
    g = G(mat)
    pos = obj_mm(g)
    px_, py_, pz_ = g.sep(pos)
    guv = g.sep(g.uv('Grain'))
    s = g.math('MULTIPLY', guv[0], 1000.0)          # along the grain (mm)
    t = g.math('MULTIPLY', guv[1], 1000.0)          # across (mm)
    warp = g.noise(g.comb(g.math('MULTIPLY', s, 0.008),
                          g.math('MULTIPLY', py_, 0.03),
                          g.math('MULTIPLY', t, 0.02)), 1.0, 2.0, 0.5)
    wig = g.noise(g.comb(g.math('MULTIPLY', s, 0.03),
                         g.math('MULTIPLY', t, 0.06),
                         g.math('MULTIPLY', py_, 0.06)), 1.0, 3.0, 0.5)
    gc = g.math('ADD', g.math('ADD', t, g.math('MULTIPLY', py_, 0.3)),
                g.math('MULTIPLY', warp, 14.0))
    gc = g.math('ADD', gc, g.math('MULTIPLY', wig, 3.0))
    # fine straight grain lines (noise squeezed along the grain)
    fine = g.noise(g.comb(g.math('MULTIPLY', s, 0.0035),
                          g.math('MULTIPLY', gc, 0.95), 0.0), 1.0, 8.0, 0.68)
    lines = g.maprange(fine, 0.47, 0.62, 0.0, 1.0, smooth=True)
    # medium streaks and large colour patches
    streak = g.noise(g.comb(g.math('MULTIPLY', s, 0.011),
                            g.math('MULTIPLY', gc, 0.2),
                            g.math('MULTIPLY', py_, 0.04)), 1.0, 6.0, 0.6)
    streak = g.maprange(streak, 0.4, 0.62, 0.0, 1.0, smooth=True)
    patch = g.maprange(g.noise(pos, 0.018, 3.0, 0.5), 0.35, 0.65, 0.0, 1.0,
                       smooth=True)
    fig = g.noise(g.comb(g.math('MULTIPLY', s, 0.006),
                         g.math('MULTIPLY', gc, 0.07), 0.0), 1.0, 5.0, 0.6,
                  dist=0.3)
    fig = g.maprange(fig, 0.5, 0.66, 0.0, 1.0, smooth=True)
    # broad dark bands running with the grain
    bands = g.noise(g.comb(g.math('MULTIPLY', s, 0.004),
                           g.math('MULTIPLY', gc, 0.045), 0.0), 1.0, 4.0, 0.55,
                    dist=0.4)
    bands = g.maprange(bands, 0.5, 0.64, 0.0, 1.0, smooth=True)
    # pores: short dark dashes along the grain
    pores = g.noise(g.comb(g.math('MULTIPLY', s, 0.5),
                           g.math('MULTIPLY', gc, 7.0),
                           g.math('MULTIPLY', py_, 7.0)), 1.0, 1.0, 0.5)
    pores = g.maprange(pores, 0.68, 0.8, 0.0, 1.0)
    # scratches: thin random lines in two families
    scr1 = g.noise(g.comb(g.math('MULTIPLY', px_, 0.05),
                          g.math('MULTIPLY', pz_, 1.6),
                          g.math('MULTIPLY', py_, 1.6)), 1.0, 2.0, 0.5,
                   dist=2.0)
    scr1 = g.maprange(scr1, 0.497, 0.503, 1.0, 0.0)
    scr1 = g.math('MULTIPLY', scr1, g.maprange(
        g.noise(pos, 0.08, 2.0, 0.5), 0.6, 0.66, 0.0, 1.0))
    scr2 = g.noise(g.comb(g.math('MULTIPLY', pz_, 0.07),
                          g.math('MULTIPLY', px_, 1.2),
                          g.math('MULTIPLY', py_, 1.2)), 1.0, 2.0, 0.5,
                   dist=2.0)
    scr2 = g.maprange(scr2, 0.4975, 0.5025, 1.0, 0.0)
    scr2 = g.math('MULTIPLY', scr2, g.maprange(
        g.noise(pos, 0.07, 2.0, 0.5, dist=1.0), 0.62, 0.68, 0.0, 1.0))
    scratches = g.math('MAXIMUM', scr1, scr2)
    ao = g.ao(0.006)
    cav = g.math('SUBTRACT', 1.0, ao, clamp=True)
    edge = g.maprange(g.geo('Pointiness'), 0.5, 0.6, 0.0, 1.0)
    # colours (linear)
    c_light = (0.34, 0.105, 0.036)
    c_base = (0.125, 0.034, 0.012)
    c_dark = (0.032, 0.009, 0.004)
    col = g.mixc(streak, c_light, c_base)
    col = g.mixc(g.math('MULTIPLY', streak, streak), col, (0.06, 0.017, 0.007))
    col = g.mixc(g.math('MULTIPLY', patch, 0.45), col, (0.12, 0.04, 0.016))
    col = g.mixc(g.math('MULTIPLY', bands, 0.75), col, (0.045, 0.014, 0.006))
    col = g.mixc(g.math('MULTIPLY', lines, 0.7), col, c_dark)
    col = g.mixc(g.math('MULTIPLY', fig, 0.85), col, (0.035, 0.012, 0.005))
    col = g.mixc(g.math('MULTIPLY', pores, 0.6), col, (0.03, 0.01, 0.005))
    col = g.mixc(g.math('MULTIPLY', cav, 1.5), col, (0.025, 0.01, 0.006))
    col = g.mixc(g.math('MULTIPLY', edge, 0.4), col, (0.36, 0.2, 0.11))
    col = g.mixc(g.math('MULTIPLY', scratches, 0.22), col, (0.36, 0.19, 0.10))
    rough = g.math('ADD', 0.44, g.math('MULTIPLY', lines, 0.06))
    rough = g.math('ADD', rough, g.math('MULTIPLY', pores, 0.2))
    rough = g.math('ADD', rough, g.math('MULTIPLY', cav, 0.25))
    rough = g.math('ADD', rough, g.math('MULTIPLY', scratches, 0.15))
    rough = g.math('SUBTRACT', rough, g.math('MULTIPLY', edge, 0.1))
    h = g.math('ADD', g.math('MULTIPLY', pores, -0.7),
               g.math('MULTIPLY', lines, -0.12))
    h = g.math('ADD', h, g.math('MULTIPLY', scratches, -0.8))
    dent = g.noise(pos, 0.35, 3.0, 0.5)
    h = g.math('ADD', h, g.math('MULTIPLY', dent, 0.8))
    nrm = g.bump(h, 0.35, 0.0003)
    g.bsdf(col, 0.0, rough, nrm, coat=0.2)
    return mat


def stone():
    mat = bpy.data.materials.new('Flint')
    g = G(mat)
    pos = obj_mm(g)
    n1 = g.noise(pos, 0.6, 6.0, 0.65)
    n2 = g.noise(pos, 3.0, 4.0, 0.6)
    col = g.mixc(n1, (0.20, 0.24, 0.19), (0.34, 0.37, 0.30))
    col = g.mixc(g.maprange(n2, 0.6, 0.8), col, (0.12, 0.13, 0.11))
    rough = g.maprange(n2, 0.2, 0.8, 0.25, 0.5)
    nrm = g.bump(n2, 0.4, 0.0004)
    g.bsdf(col, 0.0, rough, nrm)
    return mat


def leather():
    mat = bpy.data.materials.new('Leather')
    g = G(mat)
    pos = obj_mm(g)
    v = g.n('ShaderNodeTexVoronoi')
    g.l(pos, v.inputs['Vector'])
    v.inputs['Scale'].default_value = 3.0
    n = g.noise(pos, 1.5, 5.0, 0.6)
    col = g.mixc(n, (0.09, 0.05, 0.03), (0.17, 0.10, 0.06))
    nrm = g.bump(v.outputs['Distance'], 0.3, 0.0003)
    g.bsdf(col, 0.0, g.maprange(n, 0, 1, 0.55, 0.8), nrm)
    return mat


def create_all():
    brass_alb, brass_tar = (0.55, 0.38, 0.12), (0.075, 0.052, 0.02)
    silver_alb, silver_tar = (0.71, 0.70, 0.69), (0.035, 0.033, 0.032)
    return {
        'walnut': walnut(),
        'brass_engraved': metal('Brass_Engraved', brass_alb, brass_tar, 0.45,
                                engrave='barrel', rough=(0.3, 0.62),
                                patina=0.7, grime_gain=1.2,
                                relief_depth=0.00026),
        'brass': metal('Brass', brass_alb, brass_tar, 0.45,
                       rough=(0.3, 0.62), patina=0.7),
        'silver_engraved': metal('Silver_Engraved', silver_alb, silver_tar,
                                 0.6, engrave='lock', inlay='inlay',
                                 rough=(0.24, 0.58), relief_depth=0.00014,
                                 patina=0.15, grime_gain=0.8),
        'silver': metal('Silver', silver_alb, silver_tar, 0.6,
                        rough=(0.26, 0.6), patina=0.3),
        'steel': metal('Steel', (0.56, 0.56, 0.57), (0.07, 0.06, 0.05), 0.4,
                       rough=(0.3, 0.6), patina=0.25),
        'flint': stone(),
        'leather': leather(),
    }


# ----------------------------------------------------------- aux UVs ------
ENGRAVED_SILVER = ('LockPlate', 'Cock', 'Pan', 'FrizzenSpring')
ENGRAVED_BRASS = ('Barrel', 'Barrel_Muzzle')


def add_aux_uvs(objs, grain_fn):
    """'Grain' UVs on the stock; 'Proj' side projection on lock work."""
    import gb_blender as B
    mats = {m.name: m for m in bpy.data.materials}
    for name, ob in objs.items():
        me = ob.data
        if name == 'Stock':
            v = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get('co', v)
            v = v.reshape(-1, 3) / B.MM
            gc = grain_fn(v) * 0.001
            lp = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get('vertex_index', lp)
            uvl = me.uv_layers.new(name='Grain')
            uvl.data.foreach_set('uv', gc[lp].astype(np.float32).ravel())
        if name in ENGRAVED_SILVER:
            mw = np.array(ob.matrix_world)
            v = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get('co', v)
            v = v.reshape(-1, 3)
            vw = (np.c_[v, np.ones(len(v))] @ mw.T)[:, :3] / B.MM
            u = (vw[:, 0] - PROJ_X[0]) / (PROJ_X[1] - PROJ_X[0])
            w = (vw[:, 2] - PROJ_Z[0]) / (PROJ_Z[1] - PROJ_Z[0])
            lp = np.empty(len(me.loops), np.int32)
            me.loops.foreach_get('vertex_index', lp)
            uvl = me.uv_layers.new(name='Proj')
            uvl.data.foreach_set('uv', np.stack([u, w], 1)[lp].astype(
                np.float32).ravel())
            me.materials[0] = mats['Silver_Engraved']
        if name in ENGRAVED_BRASS:
            me.materials[0] = mats['Brass_Engraved']
