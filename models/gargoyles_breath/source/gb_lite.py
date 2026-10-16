"""Export a lightweight GLB (decimated meshes, 2K textures) from the baked
model, for web viewers and quick sharing.

usage: python3 gb_lite.py <GargoylesBreath.blend> <out.glb> [ratio] [tex_px]
"""
import os
import sys
import tempfile
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_blender as B     # noqa: E402

KEEP_FULL = {'Flint', 'Flint_Leather', 'Ramrod', 'RamrodPipe_Front',
             'RamrodPipe_Rear', 'Barrel', 'Barrel_Muzzle'}


def main(blend, out_glb, ratio=0.35, tex_px=2048):
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for name in [u.name for u in ob.data.uv_layers]:
            if name != 'UVMap':
                ob.data.uv_layers.remove(ob.data.uv_layers[name])
        if ob.name not in KEEP_FULL:
            B.decimate(ob, ratio)
    tmp = tempfile.mkdtemp()
    for img in bpy.data.images:
        if img.source != 'FILE' or not img.filepath:
            continue
        if max(img.size) > tex_px:
            img.scale(tex_px, tex_px)
            path = os.path.join(tmp, os.path.basename(img.filepath_raw))
            img.filepath_raw = path
            img.file_format = 'JPEG'
            img.save(quality=88)
    bpy.ops.export_scene.gltf(
        filepath=os.path.abspath(out_glb), export_format='GLB',
        export_yup=True, export_apply=True, export_texcoords=True,
        export_normals=True, export_materials='EXPORT',
        export_image_format='JPEG', export_jpeg_quality=88,
        export_cameras=False, export_lights=False)
    print('lite glb', os.path.getsize(out_glb) / 1e6, 'MB')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]) if len(a) > 2 else 0.35,
         int(a[3]) if len(a) > 3 else 2048)
