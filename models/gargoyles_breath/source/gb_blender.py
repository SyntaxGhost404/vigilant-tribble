"""Blender helpers: mesh import, scene, cameras, lighting, rendering."""
import math
import numpy as np
import bpy

MM = 0.001   # model is authored in millimetres, Blender scene is in metres


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.length_unit = 'MILLIMETERS'
    return sc


def add_mesh(name, verts, faces, uvs=None, smooth=True, collection=None,
             parent=None, material=None, sharp_angle=None):
    """verts (N,3) in mm; faces: (M,k) int array or list of index lists;
    uvs: per-loop (L,2) array or per-vertex (N,2) array."""
    verts = np.asarray(verts, dtype=np.float64)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set('co', (verts * MM).astype(np.float32).ravel())
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        loops = faces.astype(np.int32).ravel()
        sizes = np.full(len(faces), faces.shape[1], dtype=np.int32)
    else:
        sizes = np.array([len(f) for f in faces], dtype=np.int32)
        loops = np.concatenate([np.asarray(f, np.int32) for f in faces])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me.loops.add(len(loops))
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(len(sizes))
    me.polygons.foreach_set('loop_start', starts)
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if uvs is not None:
        uvs = np.asarray(uvs, dtype=np.float32)
        uvl = me.uv_layers.new(name='UVMap')
        if len(uvs) == len(verts) and len(uvs) != len(loops):
            per_loop = uvs[loops]
        else:
            per_loop = uvs
        uvl.data.foreach_set('uv', per_loop.ravel())
    if smooth:
        me.shade_smooth()
    if sharp_angle is not None:
        me.set_sharp_from_angle(angle=math.radians(sharp_angle))
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
    if material is not None:
        me.materials.append(material)
    return ob


def decimate(ob, ratio):
    if ratio >= 1.0:
        return ob
    m = ob.modifiers.new('Decimate', 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = ratio
    m.use_collapse_triangulate = True
    apply_modifiers(ob)
    # collapsing can leave a few duplicate or zero-area faces behind
    ob.data.validate()
    return ob


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ob_eval, preserve_all_data_layers=True,
                                          depsgraph=dg)
    old = ob.data
    name = old.name
    mats = list(old.materials)
    ob.modifiers.clear()
    ob.data = me
    for m in mats:
        if m is not None and m.name not in [x.name for x in me.materials if x]:
            me.materials.append(m)
    bpy.data.meshes.remove(old)
    me.name = name
    return ob


def mesh_arrays(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', v)
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', t)
    return v.reshape(-1, 3) / MM, t.reshape(-1, 3)


# ----------------------------------------------------------- cameras -------
def ortho_camera(name, view, center_mm, width_mm, res, dist=1.0):
    """view: 'right' (looks +Y, muzzle right), 'left', 'top', 'bottom',
    'front' (looks -X into the muzzle), 'back'."""
    cam = bpy.data.cameras.new(name)
    cam.type = 'ORTHO'
    cam.ortho_scale = width_mm * MM
    cam.clip_start = 0.001
    cam.clip_end = 10.0
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    cx, cy, cz = (np.asarray(center_mm, float) * MM)
    rot = {
        'right': ((math.pi / 2, 0, 0), (cx, cy - dist, cz)),
        'left': ((math.pi / 2, 0, math.pi), (cx, cy + dist, cz)),
        'top': ((0, 0, 0), (cx, cy, cz + dist)),
        'bottom': ((math.pi, 0, 0), (cx, cy, cz - dist)),
        'front': ((math.pi / 2, 0, math.pi / 2), (cx + dist, cy, cz)),
        'back': ((math.pi / 2, 0, -math.pi / 2), (cx - dist, cy, cz)),
    }[view]
    ob.rotation_euler = rot[0]
    ob.location = rot[1]
    sc = bpy.context.scene
    sc.camera = ob
    sc.render.resolution_x, sc.render.resolution_y = res
    return ob


def persp_camera(name, loc_mm, target_mm, lens=85.0, res=(1600, 1000)):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.005
    cam.clip_end = 20.0
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = np.asarray(loc_mm, float) * MM
    look_at(ob, np.asarray(target_mm, float) * MM)
    sc = bpy.context.scene
    sc.camera = ob
    sc.render.resolution_x, sc.render.resolution_y = res
    return ob


def look_at(ob, target):
    from mathutils import Vector
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ---------------------------------------------------------- lighting -------
def area_light(name, loc_mm, target_mm, power, size_mm, color=(1, 1, 1)):
    li = bpy.data.lights.new(name, 'AREA')
    li.energy = power
    li.size = size_mm * MM
    li.color = color
    ob = bpy.data.objects.new(name, li)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = np.asarray(loc_mm, float) * MM
    look_at(ob, np.asarray(target_mm, float) * MM)
    return ob


def studio(world_strength=0.35, world_color=(0.045, 0.045, 0.05), hdri=True):
    """Dark product-shot studio similar to the reference renders."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new('World')
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes['Background']
    if hdri:
        # soft gradient "studio" environment so metals have something to
        # reflect: bright overhead softbox band, dark floor.
        tc = nt.nodes.new('ShaderNodeTexCoord')
        sep = nt.nodes.new('ShaderNodeSeparateXYZ')
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        nt.links.new(tc.outputs['Generated'], sep.inputs[0])
        nt.links.new(sep.outputs['Z'], ramp.inputs[0])
        el = ramp.color_ramp.elements
        el[0].position = 0.45
        el[0].color = (0.006, 0.006, 0.007, 1)
        el[1].position = 0.92
        el[1].color = (0.55, 0.55, 0.58, 1)
        nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
    else:
        bg.inputs['Color'].default_value = (*world_color, 1)
    bg.inputs['Strength'].default_value = world_strength
    return w


def setup_render(samples=64, res=None, transparent=False, exposure=0.0):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 8
    sc.cycles.glossy_bounces = 6
    sc.cycles.transparent_max_bounces = 8
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.view_settings.exposure = exposure
    if res:
        sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'


def render(path):
    sc = bpy.context.scene
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
