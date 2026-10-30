"""Перегнать модель Poly Haven (.blend) в лёгкие .glb для сцены, по одному на каждое дерево в файле:
    blender -b model.blend --python sim/blend2glb.py -- <папка> [--tris 30000] [--tex 512]
Уменьшает каждое дерево до --tris треугольников: листву (карточки с прозрачностью) прореживает
целыми карточками, а оставшиеся немного увеличивает, чтобы крона не стала прозрачной; ствол и ветки
упрощает. Текстуры уменьшает до --tex, сложные материалы Poly Haven заменяет простыми (цвет +
прозрачность листвы срезом): экспортёр glTF и Godot понимают только такие."""
import os
import random
import sys

import bmesh
import bpy
from mathutils import Vector

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

def uv_from_attributes(o):
    """UV, сохранённые как простой атрибут (FLOAT2 или вектор), — в слой UV (иначе экспортёр их не видит)"""
    me = o.data
    for a in list(me.attributes):
        if (a.data_type in ("FLOAT2", "FLOAT_VECTOR") and a.domain == "CORNER" and a.name not in me.uv_layers
                and not a.name.startswith(".") and "uv" in a.name.lower()):
            data = [tuple(x.vector)[:2] for x in a.data]
            name = a.name
            for aux in (".vs." + name, ".es." + name, ".pn." + name):
                if aux in me.attributes:
                    me.attributes.remove(me.attributes[aux])
            me.attributes.remove(me.attributes[name])
            uv = me.uv_layers.new(name=name)
            for i, v in enumerate(data):
                uv.data[i].uv = v
    if len(me.uv_layers):
        me.uv_layers.active_index = 0
        me.uv_layers[0].active_render = True


def thin(o, tris):
    """листву — целыми карточками, остальное — схлопыванием рёбер"""
    me = o.data
    leaf = {i for i, m in enumerate(me.materials) if m is not None and m.blend_method == "CLIP"}
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    leaf_faces = [f for f in bm.faces if f.material_index in leaf]
    n_leaf = sum(len(f.verts) - 2 for f in leaf_faces)
    n_all = sum(len(f.verts) - 2 for f in bm.faces)
    leaf_budget = 0.75 * tris
    if leaf and n_leaf > leaf_budget:
        seen, islands = set(), []
        for f in leaf_faces:
            if f.index in seen:
                continue
            seen.add(f.index)
            stack, isl = [f], []
            while stack:
                g = stack.pop()
                isl.append(g)
                for e in g.edges:
                    for h in e.link_faces:
                        if h.index not in seen and h.material_index in leaf:
                            seen.add(h.index)
                            stack.append(h)
            islands.append(isl)
        keep = leaf_budget / n_leaf
        grow = min(2.0, keep ** -0.5)
        rnd = random.Random(1)
        kill = []
        for isl in islands:
            if rnd.random() < keep:
                vs = list({v for f in isl for v in f.verts})
                c = sum((v.co for v in vs), Vector()) / len(vs)
                for v in vs:
                    v.co = c + (v.co - c) * grow
            else:
                kill.extend(isl)
        bmesh.ops.delete(bm, geom=kill, context="FACES")
        print("  листва:", len(islands), "карточек, оставлено", round(keep * 100, 1), "%, увеличены в", round(grow, 2))
    bm.to_mesh(me)
    bm.free()
    rest = sum(len(p.vertices) - 2 for p in me.polygons if p.material_index not in leaf)
    rest_budget = tris - min(n_leaf, leaf_budget)
    if rest > rest_budget > 0:
        vg = o.vertex_groups.new(name="bark")
        vg.add([v for p in me.polygons if p.material_index not in leaf for v in p.vertices], 1.0, "REPLACE")
        m = o.modifiers.new("dec", "DECIMATE")
        m.ratio = max(0.05, rest_budget / rest)
        m.vertex_group = "bark"
        m.use_collapse_triangulate = True
    return n_all


meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get()]
for o in meshes:
    uv_from_attributes(o)
    n = thin(o, tris)
    # каждое дерево отдельно и в начале координат (в файле Poly Haven варианты стоят рядом)
    loc = o.location.copy()
    o.location = (0, 0, 0)
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    path = os.path.join(out_dir, o.name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_image_format="AUTO", export_yup=True)
    o.location = loc
    print("записан", path, "треугольников было", n, "стало", sum(len(p.vertices) - 2 for p in o.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.polygons), os.path.getsize(path) // 1024, "КБ")
