"""Перегнать модель Poly Haven (.blend) в лёгкие .glb для сцены, по одному на каждое дерево в файле:
    blender -b model.blend --python sim/blend2glb.py -- <папка> [--tris 30000] [--tex 512]
Прореживает каждую сетку до --tris треугольников, уменьшает текстуры до --tex и заменяет сложные
материалы Poly Haven простыми (цвет + прозрачность листвы срезом): экспортёр glTF и Godot понимают
только такие."""
import os
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_dir = argv[0]
tris = int(argv[argv.index("--tris") + 1]) if "--tris" in argv else 30000
tex = int(argv[argv.index("--tex") + 1]) if "--tex" in argv else 512
os.makedirs(out_dir, exist_ok=True)


def find_image(mat, key):
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image and key in n.image.name.lower():
            return n.image
    return None


def simplify(mat):
    diff = find_image(mat, "diff") or find_image(mat, "col")
    alpha = find_image(mat, "alpha") or find_image(mat, "mask")
    nt = mat.node_tree
    nt.nodes.clear()
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    nt.links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
    if diff:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = diff
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if alpha:
        a = nt.nodes.new("ShaderNodeTexImage")
        a.image = alpha
        alpha.colorspace_settings.name = "Non-Color"
        nt.links.new(a.outputs["Color"], bsdf.inputs["Alpha"])
        mat.blend_method = "CLIP"
        mat.alpha_threshold = 0.5
    print("  материал", mat.name, "цвет", diff and diff.name, "прозрачность", alpha and alpha.name)


for mat in bpy.data.materials:
    if mat.node_tree is not None and mat.users:
        simplify(mat)
for img in bpy.data.images:
    if img.size[0] > tex:
        img.scale(tex, max(1, int(img.size[1] * tex / img.size[0])))

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get()]
for o in meshes:
    n = sum(len(p.vertices) - 2 for p in o.data.polygons)
    if n > tris:
        m = o.modifiers.new("dec", "DECIMATE")
        m.ratio = tris / n
        m.use_collapse_triangulate = True
    # каждое дерево отдельно и в начале координат (в файле Poly Haven варианты стоят рядом)
    loc = o.location.copy()
    o.location = (0, 0, 0)
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    path = os.path.join(out_dir, o.name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_image_format="AUTO", export_yup=True)
    o.location = loc
    print("записан", path, "треугольников было", n, "стало ~", min(n, tris), os.path.getsize(path) // 1024, "КБ")
