"""Studio renders of The Gargoyle's Breath.

usage: python3 gb_render.py <model.blend> <out_dir> [views...] [--samples N]
views: right left top front hero gargoyle lock muzzle (default: all)
"""
import os
import sys
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_blender as B     # noqa: E402
import gb_profiles as P    # noqa: E402

MM = B.MM
PCT = 100


def studio():
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new('World')
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground')
    lp = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    # what metals reflect: dim overhead gradient; what the camera sees: black
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    nt.links.new(tc.outputs['Generated'], sep.inputs[0])
    nt.links.new(sep.outputs['Z'], ramp.inputs[0])
    el = ramp.color_ramp.elements
    el[0].position, el[0].color = 0.42, (0.004, 0.004, 0.005, 1)
    el[1].position, el[1].color = 0.95, (0.30, 0.30, 0.32, 1)
    nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = 0.35
    black = nt.nodes.new('ShaderNodeBackground')
    black.inputs['Color'].default_value = (0.008, 0.008, 0.009, 1)
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(black.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    tgt = (40, 0, -40)
    B.area_light('Key', (-180, -420, 380), tgt, 11, 380, (1.0, 0.96, 0.9))
    B.area_light('Fill', (320, -380, -120), tgt, 2.5, 500, (0.9, 0.95, 1.0))
    B.area_light('Rim', (60, 520, 260), tgt, 8, 420, (1.0, 0.98, 0.95))
    B.area_light('Top', (80, -60, 560), tgt, 6, 600)
    B.area_light('Under', (-60, -200, -420), (-120, 0, -110), 1.2, 300)


VIEWS = {
    # ortho views matched to the reference page framing (ref px windows)
    'right': ('ortho', 'right', (150, 350, 800, 700), 3),
    'left': ('ortho', 'left', (150, 350, 800, 700), 3),
    'top': ('ortho', 'top', None, 3),
    'front': ('ortho', 'front', None, 3),
}


def ortho_from_ref(name, view, box, scale):
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2 - P.X0) * P.K
    cz = (P.Z0 - (y0 + y1) / 2) * P.K
    return B.ortho_camera(name, view, (cx, 0, cz), (x1 - x0) * P.K,
                          ((x1 - x0) * scale, (y1 - y0) * scale))


def render_view(name, out_dir, samples):
    sc = bpy.context.scene
    if name in ('right', 'left'):
        ortho_from_ref('cam_' + name, name, VIEWS[name][2], VIEWS[name][3])
    elif name == 'top':
        B.ortho_camera('cam_top', 'top', (50, 0, 0), 480, (1950, 600))
    elif name == 'front':
        B.ortho_camera('cam_front', 'front', (0, 0, -40), 220, (1100, 1100))
    elif name == 'hero':
        B.persp_camera('cam_hero', (330, -430, 230), (35, 0, -35), lens=55,
                       res=(2400, 1500))
    elif name == 'gargoyle':
        B.persp_camera('cam_garg', (-30, -215, -250), (-146, 0, -112),
                       lens=85, res=(1600, 1600))
    elif name == 'lock':
        B.persp_camera('cam_lock', (110, -260, 120), (10, -20, 5), lens=85,
                       res=(2000, 1300))
    elif name == 'muzzle':
        B.persp_camera('cam_muzzle', (470, -170, 120), (220, 0, 0), lens=70,
                       res=(1600, 1200))
    sc.cycles.samples = samples
    sc.render.resolution_percentage = PCT
    B.render(os.path.join(out_dir, '%s.png' % name))


def main():
    argv = sys.argv[1:]
    samples = 96
    pct = 100
    expo = -1.6
    for flag in ('--samples', '--pct', '--exposure'):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            del argv[i:i + 2]
            if flag == '--samples':
                samples = int(val)
            elif flag == '--pct':
                pct = int(val)
            else:
                expo = float(val)
    blend, out_dir = argv[0], argv[1]
    views = argv[2:] or ['right', 'left', 'top', 'hero', 'gargoyle', 'lock',
                         'muzzle', 'front']
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
    for ob in list(bpy.data.objects):
        if ob.type in ('LIGHT', 'CAMERA'):
            bpy.data.objects.remove(ob)
    os.makedirs(out_dir, exist_ok=True)
    studio()
    B.setup_render(samples=samples, exposure=expo)
    sc = bpy.context.scene
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.cycles.use_denoising = True
    global PCT
    PCT = pct
    for v in views:
        render_view(v, out_dir, samples)
        print('rendered', v, flush=True)


if __name__ == '__main__':
    main()
