"""Bake the procedural materials to PBR texture atlases and export glTF.

usage: python3 gb_bake.py <in.blend> <out_dir> [atlas_res]

For each atlas group the script
  1. adds an 'Atlas' UV layer to every member object and packs all members
     into it with Smart UV Project,
  2. bakes ambient occlusion once and swaps the material AO nodes for the
     baked map (so the remaining bakes are noise-free),
  3. bakes base colour / roughness / metallic through an emission shader and
     the tangent-space normal (which includes all bump detail),
  4. builds glTF-ready materials (BaseColor, ORM, Normal) and exports a GLB.
"""
import os
import sys
import math
import numpy as np
import bpy

GROUPS = {
    'Walnut': (['Walnut'], 4096),
    'Brass': (['Brass_Engraved', 'Brass'], 4096),
    'Silver': (['Silver_Engraved', 'Silver'], 4096),
    'Steel': (['Steel', 'Flint', 'Leather'], 2048),
}
AO_DIST = 0.005


def log(*a):
    import time
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def group_objects(names):
    out = []
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.data.materials and \
                ob.data.materials[0].name in names:
            out.append(ob)
    return out


def select_only(objs, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def make_atlas_uvs(objs):
    for o in objs:
        me = o.data
        uv = me.uv_layers.get('Atlas') or me.uv_layers.new(name='Atlas')
        me.uv_layers.active = uv
    select_only(objs)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60.0),
                             island_margin=0.0025, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.pack_islands(margin=0.0025, rotate=True)
    bpy.ops.object.mode_set(mode='OBJECT')


def new_image(name, res, non_color, alpha=False):
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, res, res, alpha=alpha, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    return img


def set_bake_target(mats, img):
    for m in mats:
        nt = m.node_tree
        node = nt.nodes.get('BAKE_TARGET')
        if node is None:
            node = nt.nodes.new('ShaderNodeTexImage')
            node.name = 'BAKE_TARGET'
            uvn = nt.nodes.new('ShaderNodeUVMap')
            uvn.name = 'BAKE_UV'
            uvn.uv_map = 'Atlas'
            nt.links.new(uvn.outputs['UV'], node.inputs['Vector'])
        node.image = img
        for n in nt.nodes:
            n.select = False
        node.select = True
        nt.nodes.active = node


def bake(kind, samples, margin=12):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.render.bake.margin = margin
    sc.render.bake.margin_type = 'EXTEND'
    sc.render.bake.use_selected_to_active = False
    sc.render.bake.target = 'IMAGE_TEXTURES'
    if kind == 'NORMAL':
        bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT')
    else:
        bpy.ops.object.bake(type=kind)


def emit_channel(mats, channel):
    """Route a BSDF input to an emission output; returns restore info."""
    saved = []
    for m in mats:
        nt = m.node_tree
        bsdf = nt.nodes['BSDF']
        out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
        em = nt.nodes.get('BAKE_EMIT') or nt.nodes.new('ShaderNodeEmission')
        em.name = 'BAKE_EMIT'
        sock = bsdf.inputs[channel]
        if sock.links:
            src = sock.links[0].from_socket
            nt.links.new(src, em.inputs['Color'])
        else:
            v = sock.default_value
            if channel == 'Base Color':
                em.inputs['Color'].default_value = tuple(v)
            else:
                em.inputs['Color'].default_value = (v, v, v, 1.0)
            for lk in list(em.inputs['Color'].links):
                nt.links.remove(lk)
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
        saved.append((nt, bsdf, out))
    return saved


def plain_shader(mats):
    saved = []
    for m in mats:
        nt = m.node_tree
        bsdf = nt.nodes['BSDF']
        out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
        d = nt.nodes.get('BAKE_DIFFUSE') or nt.nodes.new(
            'ShaderNodeBsdfDiffuse')
        d.name = 'BAKE_DIFFUSE'
        nt.links.new(d.outputs['BSDF'], out.inputs['Surface'])
        saved.append((nt, bsdf, out))
    return saved


def restore(saved):
    for nt, bsdf, out in saved:
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])


def swap_ao(mats, ao_img):
    """Replace every AO node output with the baked AO map."""
    for m in mats:
        nt = m.node_tree
        aos = [n for n in nt.nodes if n.type == 'AMBIENT_OCCLUSION']
        if not aos:
            continue
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.name = 'BAKED_AO'
        tex.image = ao_img
        uvn = nt.nodes.new('ShaderNodeUVMap')
        uvn.uv_map = 'Atlas'
        nt.links.new(uvn.outputs['UV'], tex.inputs['Vector'])
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(tex.outputs['Color'], sep.inputs['Color'])
        for a in aos:
            for lk in list(a.outputs['AO'].links):
                nt.links.new(sep.outputs[0], lk.to_socket)


def denoise_ao(img, res):
    from scipy import ndimage
    a = np.empty(res * res * 4, np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(res, res, 4)
    g = ndimage.gaussian_filter(a[..., 0], 1.5)
    a[..., 0] = a[..., 1] = a[..., 2] = g
    img.pixels.foreach_set(a.ravel())
    img.update()


def pixels(img, res):
    a = np.empty(res * res * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(res, res, 4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_rgb(img, res, path, srgb, quality=92):
    from PIL import Image
    a = pixels(img, res)[..., :3]
    if srgb:
        a = linear_to_srgb(a)
    a = (np.clip(a[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    if path.lower().endswith('.jpg'):
        im.save(path, quality=quality, subsampling=0)
    else:
        im.save(path, optimize=True)


def write_orm(ao, rough, metal, res, path):
    from PIL import Image
    A = pixels(ao, res)[..., 0]
    R = pixels(rough, res)[..., 0]
    M = pixels(metal, res)[..., 0]
    rgb = np.stack([A, R, M], -1)
    rgb = (np.clip(rgb[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(path, quality=93, subsampling=0)


def export_material(group, tex_dir, base_path, orm_path, nrm_path):
    m = bpy.data.materials.new('M_' + group)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    uvn = nt.nodes.new('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'

    def tex(path, non_color):
        t = nt.nodes.new('ShaderNodeTexImage')
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
        t.image = img
        nt.links.new(uvn.outputs['UV'], t.inputs['Vector'])
        return t
    tb = tex(base_path, False)
    nt.links.new(tb.outputs['Color'], b.inputs['Base Color'])
    to = tex(orm_path, True)
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(to.outputs['Color'], sep.inputs['Color'])
    nt.links.new(sep.outputs['Green'], b.inputs['Roughness'])
    nt.links.new(sep.outputs['Blue'], b.inputs['Metallic'])
    tn = tex(nrm_path, True)
    nm = nt.nodes.new('ShaderNodeNormalMap')
    nm.uv_map = 'UVMap'
    nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    # occlusion for the glTF exporter
    grp = bpy.data.node_groups.get('glTF Material Output')
    if grp is None:
        grp = bpy.data.node_groups.new('glTF Material Output',
                                       'ShaderNodeTree')
        grp.interface.new_socket('Occlusion', in_out='INPUT',
                                 socket_type='NodeSocketFloat')
    gn = nt.nodes.new('ShaderNodeGroup')
    gn.node_tree = grp
    nt.links.new(sep.outputs['Red'], gn.inputs['Occlusion'])
    return m


def bake_group(group, matnames, res, tex_dir):
    objs = group_objects(matnames)
    if not objs:
        return None
    mats = [bpy.data.materials[n] for n in matnames if n in bpy.data.materials]
    log('[%s] %d objects, %d materials, %dpx' % (group, len(objs), len(mats),
                                                   res))
    make_atlas_uvs(objs)
    select_only(objs)
    # 1) ambient occlusion -- through a plain diffuse shader, so the
    #    materials' own AO nodes are not evaluated for every bake sample
    ao = new_image(group + '_AO', res, True)
    set_bake_target(mats, ao)
    bpy.context.scene.world.light_settings.distance = AO_DIST
    saved = plain_shader(mats)
    log('  uv ok, baking AO')
    bake('AO', 32)
    restore(saved)
    denoise_ao(ao, res)
    swap_ao(mats, ao)
    # 2) channels through emission
    imgs = {}
    # every bake target is a linear float buffer; colour encoding to sRGB
    # happens once, when the base colour is written to disk
    for ch, nc in (('Base Color', True), ('Roughness', True),
                   ('Metallic', True)):
        img = new_image('%s_%s' % (group, ch.replace(' ', '')), res, nc)
        set_bake_target(mats, img)
        saved = emit_channel(mats, ch)
        log('  baking', ch)
        bake('EMIT', 2)
        restore(saved)
        imgs[ch] = img
    # 3) tangent-space normal
    nrm = new_image(group + '_Normal', res, True)
    set_bake_target(mats, nrm)
    log('  baking normal')
    bake('NORMAL', 2)
    base_p = os.path.join(tex_dir, 'T_%s_BaseColor.jpg' % group)
    orm_p = os.path.join(tex_dir, 'T_%s_ORM.jpg' % group)
    nrm_p = os.path.join(tex_dir, 'T_%s_Normal.jpg' % group)
    save_rgb(imgs['Base Color'], res, base_p, srgb=True, quality=92)
    write_orm(ao, imgs['Roughness'], imgs['Metallic'], res, orm_p)
    save_rgb(nrm, res, nrm_p, srgb=False, quality=95)
    return objs, export_material(group, tex_dir, base_p, orm_p, nrm_p)


def main(blend_in, out_dir, max_res=None):
    bpy.ops.wm.open_mainfile(filepath=blend_in)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    if sc.world is None:
        sc.world = bpy.data.worlds.new('World')
    tex_dir = os.path.join(out_dir, 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    # untouched copies of the procedural materials, kept for re-baking
    for names, _ in GROUPS.values():
        for n in names:
            m = bpy.data.materials.get(n)
            if m is not None:
                c = m.copy()
                c.name = n + '_Procedural'
                c.use_fake_user = True
    results = {}
    for group, (names, res) in GROUPS.items():
        if max_res:
            res = min(res, max_res)
        r = bake_group(group, names, res, tex_dir)
        if r:
            results[group] = r
    # assign the baked materials; the atlas UV becomes the render UV
    for group, (objs, mat) in results.items():
        for o in objs:
            me = o.data
            uv = me.uv_layers['Atlas']
            uv.name = 'UVMap'
            uv.active_render = True
            me.uv_layers.active = uv
            me.materials[0] = mat
    # drop the bake-time working materials and generated buffers
    for m in list(bpy.data.materials):
        if m.users == 0 and not m.use_fake_user:
            bpy.data.materials.remove(m)
    for names, _ in GROUPS.values():
        for n in names:
            m = bpy.data.materials.get(n)
            if m is not None and not m.name.endswith('_Procedural'):
                bpy.data.materials.remove(m)
    for img in list(bpy.data.images):
        if img.source == 'GENERATED':
            bpy.data.images.remove(img)
    bpy.ops.file.make_paths_relative()
    blend_out = os.path.join(out_dir, 'GargoylesBreath.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend_out, compress=True,
                                relative_remap=True)
    # glTF: only the atlas UVs
    for ob in bpy.data.objects:
        if ob.type == 'MESH':
            for name in [u.name for u in ob.data.uv_layers]:
                if name != 'UVMap':
                    ob.data.uv_layers.remove(ob.data.uv_layers[name])
    glb = os.path.join(out_dir, 'GargoylesBreath.glb')
    bpy.ops.export_scene.gltf(
        filepath=glb, export_format='GLB', export_yup=True,
        export_apply=True, export_texcoords=True, export_normals=True,
        export_tangents=False, export_materials='EXPORT',
        export_image_format='AUTO', export_cameras=False,
        export_lights=False, use_selection=False)
    log('exported', glb, os.path.getsize(glb) / 1e6, 'MB')


if __name__ == '__main__':
    argv = sys.argv[1:]
    main(os.path.abspath(argv[0]), os.path.abspath(argv[1]),
         int(argv[2]) if len(argv) > 2 else None)
