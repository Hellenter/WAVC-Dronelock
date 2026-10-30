extends Node
# Виртуальная установка Дронолока: лес, дома, холмы, небо с облаками; широкая камера и точная
# на поворотке; дроны и птицы по траекториям. Кадры пишутся в PNG, разметка — в <видео>.gt.json
# (формат detector/synth_sky.py: по кадрам kind/bright/x/y), позы камер — в meta.json.
#
# Время модельное: каждый кадр каждой камеры рендерится в свой момент t = k / fps, сколько бы ни
# шёл рендер, поэтому ролик повторяется бит в бит при том же seed.
#
# Координаты в конфиге как в emulator/geom.py: x — восток, y — север, z — вверх (метры);
# азимут от севера по часовой, угол места от горизонта (градусы). В Godot: (x, y, z) = (восток, верх, -север).
#
# Запуск: godot --path sim/godot --rendering-driver opengl3 -- --config <сцена.json> --out <папка>
#   [--seconds N] [--scale K] — K < 1 уменьшает разрешение обеих камер (быстрый просмотр).

const DEG := PI / 180.0

var cfg := {}
var out_dir := ""
var rng := RandomNumberGenerator.new()
var noise := FastNoiseLite.new()

var world: Node3D
var wide_vp: SubViewport
var fine_vp: SubViewport
var wide_cam: Camera3D
var fine_cam: Camera3D
var wcfg := {}
var fcfg := {}
var rig_pos := Vector3.ZERO

var time_mats: Array = []      # материалы с параметром sim_time (ветер, облака)
var targets: Array = []        # {id, kind, cfg, node, wings, size}
var trees: Array = []          # для заслонения: [x, z, r, y0, y1] в координатах Godot
var tree_buckets: Array = []   # 360 корзин по азимуту от установки -> индексы trees
var boxes: Array = []          # AABB домов и столбов

# поворотка точной камеры
var f_az := 0.0
var f_el := 0.0
var f_vaz := 0.0
var f_vel := 0.0
var f_t := 0.0
var f_target := -1

var gt_wide: Array = []
var gt_fine: Array = []
var poses_wide: Array = []
var poses_fine: Array = []


func _ready() -> void:
	var args = parse_args()
	if not args.has("config"):
		push_error("нужен --config <сцена.json>")
		get_tree().quit(2)
		return
	var txt = FileAccess.get_file_as_string(args["config"])
	var parsed = JSON.parse_string(txt)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("не читается конфиг " + str(args["config"]))
		get_tree().quit(2)
		return
	cfg = parsed
	out_dir = args.get("out", "out")
	if args.has("seconds"):
		cfg["seconds"] = float(args["seconds"])
	var scale = float(args.get("scale", 1.0))
	wcfg = cfg.get("wide", {})
	fcfg = cfg.get("fine", {})
	if scale != 1.0:
		for c in [wcfg, fcfg]:
			c["w"] = int(round(float(c.get("w", 640)) * scale))
			c["h"] = int(round(float(c.get("h", 480)) * scale))
	rng.seed = int(cfg.get("seed", 1))
	noise.seed = int(cfg.get("seed", 1))
	noise.frequency = 0.004
	Engine.max_fps = 0
	DirAccess.make_dir_recursive_absolute(out_dir + "/wide")
	if fine_enabled():
		DirAccess.make_dir_recursive_absolute(out_dir + "/fine")

	setup_viewports()
	build_environment()
	build_ground()
	build_hills()
	build_trees()
	build_houses()
	build_poles()
	build_targets()
	await run()
	get_tree().quit()


func parse_args() -> Dictionary:
	var a = {}
	var list = OS.get_cmdline_user_args()
	var i = 0
	while i < list.size():
		var k: String = list[i]
		if k.begins_with("--") and i + 1 < list.size():
			a[k.substr(2)] = list[i + 1]
			i += 2
		else:
			i += 1
	return a


func fine_enabled() -> bool:
	return bool(fcfg.get("enabled", true))


# ---------- геометрия ----------

func enu(v) -> Vector3:
	return Vector3(float(v[0]), float(v[2]), -float(v[1]))


func dir_azel(az: float, el: float) -> Vector3:
	var a = az * DEG
	var e = el * DEG
	return Vector3(sin(a) * cos(e), sin(e), -cos(a) * cos(e))


func azel_of(d: Vector3) -> Vector2:
	var az = fposmod(atan2(d.x, -d.z) / DEG, 360.0)
	var el = asin(clampf(d.y / maxf(d.length(), 1e-9), -1.0, 1.0)) / DEG
	return Vector2(az, el)


func wrap180(a: float) -> float:
	return fposmod(a + 180.0, 360.0) - 180.0


func aim(cam: Camera3D, az: float, el: float, roll := 0.0) -> void:
	var b = Basis.looking_at(dir_azel(az, el), Vector3.UP)
	if roll != 0.0:
		b = b * Basis(Vector3(0, 0, 1), roll * DEG)
	cam.global_transform = Transform3D(b, rig_pos)


func make_shader(code: String) -> Shader:
	var s = Shader.new()
	s.code = code
	return s


func mat_color(c: Color, rough := 1.0) -> StandardMaterial3D:
	var m = StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = rough
	return m


func col(v, def: Color) -> Color:
	if v == null:
		return def
	return Color(float(v[0]), float(v[1]), float(v[2]))


# ---------- камеры ----------

func setup_viewports() -> void:
	rig_pos = Vector3(0, float(cfg.get("rig", {}).get("height", 4.0)), 0)
	var msaa = {0: Viewport.MSAA_DISABLED, 2: Viewport.MSAA_2X, 4: Viewport.MSAA_4X, 8: Viewport.MSAA_8X}
	wide_vp = SubViewport.new()
	wide_vp.size = Vector2i(int(wcfg.get("w", 4608)), int(wcfg.get("h", 3456)))
	wide_vp.world_3d = World3D.new()      # один мир на обе камеры
	wide_vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	wide_vp.msaa_3d = msaa.get(int(wcfg.get("msaa", 4)), Viewport.MSAA_4X)
	add_child(wide_vp)
	world = Node3D.new()
	wide_vp.add_child(world)
	wide_cam = Camera3D.new()
	wide_cam.keep_aspect = Camera3D.KEEP_WIDTH   # fov по ширине кадра
	wide_cam.fov = float(wcfg.get("hfov", 120.0))
	wide_cam.near = 0.3
	wide_cam.far = 9000.0
	wide_vp.add_child(wide_cam)
	wide_cam.current = true

	fine_vp = SubViewport.new()
	fine_vp.size = Vector2i(int(fcfg.get("w", 640)), int(fcfg.get("h", 480)))
	fine_vp.world_3d = wide_vp.world_3d
	fine_vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	fine_vp.msaa_3d = msaa.get(int(fcfg.get("msaa", 4)), Viewport.MSAA_4X)
	add_child(fine_vp)
	fine_cam = Camera3D.new()
	fine_cam.keep_aspect = Camera3D.KEEP_WIDTH
	fine_cam.fov = float(fcfg.get("hfov", 20.0))
	fine_cam.near = 0.3
	fine_cam.far = 9000.0
	fine_vp.add_child(fine_cam)
	fine_cam.current = true
	f_az = float(wcfg.get("az", 0.0))
	f_el = float(wcfg.get("el", 15.0))


func wide_pose(t: float) -> Vector2:
	var az0 = float(wcfg.get("az", 0.0))
	var el = float(wcfg.get("el", 15.0))
	var pan = float(wcfg.get("pan_dps", 0.0))
	var sweep = float(wcfg.get("sweep_deg", 0.0))
	if pan == 0.0:
		return Vector2(az0, el)
	if sweep <= 0.0:
		return Vector2(fposmod(az0 + pan * t, 360.0), el)
	# качание туда-обратно ±sweep с постоянной скоростью pan
	var period = 4.0 * sweep / absf(pan)
	var ph = fposmod(t, period) / period
	var tri = 4.0 * ph - 1.0 if ph < 0.5 else 3.0 - 4.0 * ph
	return Vector2(fposmod(az0 + sweep * tri, 360.0), el)


# ---------- окружение ----------

const SKY_SHADER := """
shader_type sky;
uniform vec3 zenith : source_color = vec3(0.22, 0.42, 0.78);
uniform vec3 horizon : source_color = vec3(0.72, 0.80, 0.90);
uniform vec3 cloud_lit : source_color = vec3(0.96, 0.96, 0.97);
uniform vec3 cloud_dark : source_color = vec3(0.58, 0.61, 0.66);
uniform float cloud_cover = 0.4;
uniform float cloud_soft = 0.18;
uniform float cloud_scale = 0.0008;
uniform float cloud_height = 1500.0;
uniform vec2 wind = vec2(6.0, 2.0);
uniform float sim_time = 0.0;
uniform float seed = 0.0;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
	vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) {
	float a = 0.5; float s = 0.0;
	for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.0, 9.0); a *= 0.5; }
	return s;
}
void sky() {
	vec3 d = EYEDIR;
	float h = max(d.y, 0.0);
	vec3 c = mix(horizon, zenith, pow(h, 0.45));
	float sd = max(dot(d, LIGHT0_DIRECTION), 0.0);
	c += LIGHT0_COLOR * (pow(sd, 900.0) * 6.0 + pow(sd, 12.0) * 0.12);
	if (d.y > 0.003) {
		vec2 uv = d.xz / d.y * cloud_height + wind * sim_time;
		float n = fbm(uv * cloud_scale + vec2(seed * 13.1, seed * 7.7));
		float k = smoothstep(1.0 - cloud_cover - cloud_soft, 1.0 - cloud_cover + cloud_soft, n);
		k *= smoothstep(0.003, 0.08, d.y);
		vec3 cc = mix(cloud_lit, cloud_dark, smoothstep(0.55, 0.95, n));
		cc += LIGHT0_COLOR * pow(sd, 6.0) * 0.25;
		c = mix(c, cc, k);
	}
	COLOR = c;
}
"""

func build_environment() -> void:
	var w: Dictionary = cfg.get("world", {})
	var env = Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky = Sky.new()
	var sm = ShaderMaterial.new()
	sm.shader = make_shader(SKY_SHADER)
	var hz = col(w.get("horizon"), Color(0.72, 0.80, 0.90))
	sm.set_shader_parameter("horizon", hz)
	sm.set_shader_parameter("zenith", col(w.get("zenith"), Color(0.22, 0.42, 0.78)))
	sm.set_shader_parameter("cloud_cover", float(w.get("cloud_cover", 0.4)))
	sm.set_shader_parameter("cloud_soft", float(w.get("cloud_soft", 0.18)))
	sm.set_shader_parameter("cloud_scale", float(w.get("cloud_scale", 0.0008)))
	var wind = enu(w.get("cloud_wind", [6.0, 2.0, 0.0]))
	sm.set_shader_parameter("wind", Vector2(wind.x, wind.z))
	sm.set_shader_parameter("seed", float(cfg.get("seed", 1)))
	time_mats.append(sm)
	sky.sky_material = sm
	# настоящее небо: панорама HDRI (.hdr/.exr, например Poly Haven) вместо нарисованного
	var hdri := str(w.get("sky_hdri", ""))
	if hdri != "":
		if hdri.is_relative_path():
			hdri = ProjectSettings.globalize_path("res://assets/sky/" + hdri)
		var img = Image.load_from_file(hdri)
		if img == null:
			push_warning("не открывается небо " + hdri)
		else:
			var pm = PanoramaSkyMaterial.new()
			pm.panorama = ImageTexture.create_from_image(img)
			pm.energy_multiplier = float(w.get("sky_energy", 1.0))
			sky.sky_material = pm
			env.sky_rotation = Vector3(0, -float(w.get("sky_rotation", 0.0)) * DEG, 0)
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = hz
	env.ambient_light_energy = float(w.get("ambient", 0.55))
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var fog = float(w.get("fog", 0.00025))
	if fog > 0.0:
		env.fog_enabled = true
		env.fog_light_color = hz
		env.fog_density = fog
		env.fog_sky_affect = 0.0
	var we = WorldEnvironment.new()
	we.environment = env
	world.add_child(we)

	var sun = DirectionalLight3D.new()
	var saz = float(w.get("sun_az", 150.0))
	var sel = clampf(float(w.get("sun_el", 35.0)), 1.0, 85.0)
	sun.light_energy = float(w.get("sun", 1.1))
	sun.light_color = col(w.get("sun_color"), Color(1.0, 0.97, 0.9))
	sun.shadow_enabled = bool(w.get("shadows", false))
	sun.directional_shadow_max_distance = 600.0
	world.add_child(sun)
	sun.global_transform = Transform3D(Basis.looking_at(-dir_azel(saz, sel), Vector3.UP), Vector3.ZERO)


const NOISE_FN := """
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
	vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
"""

const GROUND_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform vec3 c1 : source_color = vec3(0.30, 0.42, 0.18);
uniform vec3 c2 : source_color = vec3(0.45, 0.43, 0.25);
uniform float freq = 0.02;
varying vec3 wp;
%s
void vertex() { wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
void fragment() {
	float n = vnoise(wp.xz * freq) * 0.6 + vnoise(wp.xz * freq * 7.3) * 0.4;
	ALBEDO = mix(c1, c2, smoothstep(0.3, 0.8, n));
	ROUGHNESS = 1.0;
	if (!FRONT_FACING) { NORMAL = -NORMAL; }
}
"""

func ground_mat(c1: Color, c2: Color, freq: float) -> ShaderMaterial:
	var m = ShaderMaterial.new()
	m.shader = make_shader(GROUND_SHADER % NOISE_FN)
	m.set_shader_parameter("c1", c1)
	m.set_shader_parameter("c2", c2)
	m.set_shader_parameter("freq", freq)
	return m


func build_ground() -> void:
	var w: Dictionary = cfg.get("world", {})
	var mi = MeshInstance3D.new()
	var pm = PlaneMesh.new()
	pm.size = Vector2(16000, 16000)
	mi.mesh = pm
	mi.material_override = ground_mat(col(w.get("grass"), Color(0.30, 0.42, 0.18)),
		col(w.get("grass2"), Color(0.45, 0.43, 0.25)), 0.02)
	world.add_child(mi)


func build_hills() -> void:
	var w: Dictionary = cfg.get("world", {})
	var hh = float(w.get("hills", 120.0))
	if hh <= 0.0:
		return
	var r0 = float(w.get("hills_from", 1500.0))
	var r1 = r0 + 3000.0
	var st = SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var na = 180
	var nr = 12
	var hn = FastNoiseLite.new()
	hn.seed = int(cfg.get("seed", 1)) + 7
	hn.frequency = 0.0012
	var pts = []
	for j in nr + 1:
		var r = lerpf(r0, r1, float(j) / nr)
		var row = []
		for i in na + 1:
			var a = TAU * float(i % na) / na
			var x = sin(a) * r
			var z = -cos(a) * r
			var prof = smoothstep(r0, r0 + 800.0, r)
			var y = (hn.get_noise_2d(x, z) * 0.5 + 0.5) * hh * prof
			row.append(Vector3(x, y, z))
		pts.append(row)
	for j in nr:
		for i in na:
			var a: Vector3 = pts[j][i]
			var b: Vector3 = pts[j][i + 1]
			var c: Vector3 = pts[j + 1][i]
			var d: Vector3 = pts[j + 1][i + 1]
			for v in [a, c, b, b, c, d]:
				st.add_vertex(v)
	st.generate_normals()
	var mi = MeshInstance3D.new()
	mi.mesh = st.commit()
	mi.material_override = ground_mat(Color(0.20, 0.30, 0.16), Color(0.28, 0.34, 0.20), 0.004)
	world.add_child(mi)


# ---------- деревья ----------

const CROWN_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform vec3 albedo : source_color = vec3(0.16, 0.30, 0.12);
uniform float sim_time = 0.0;
uniform float sway = 1.0;
void vertex() {
	vec3 wp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	float ph = wp.x * 0.07 + wp.z * 0.05;
	float k = max(VERTEX.y - 0.35, 0.0);
	VERTEX.x += (sin(sim_time * 1.7 + ph) + 0.4 * sin(sim_time * 4.3 + ph * 3.0)) * sway * k * 0.012;
	VERTEX.z += (sin(sim_time * 2.3 + ph * 1.3) + 0.4 * sin(sim_time * 5.1 + ph * 2.0)) * sway * k * 0.009;
}
void fragment() { ALBEDO = albedo * COLOR.rgb; ROUGHNESS = 1.0; }
"""

func crown_mat(c: Color) -> ShaderMaterial:
	var m = ShaderMaterial.new()
	m.shader = make_shader(CROWN_SHADER)
	m.set_shader_parameter("albedo", c)
	m.set_shader_parameter("sway", float(cfg.get("world", {}).get("wind", 1.0)))
	time_mats.append(m)
	return m


# Дерево единичной высоты: поверхность 0 — ствол, 1 — крона. Возвращает [mesh, радиус кроны, низ кроны].
func proc_tree(kind: String) -> Array:
	var out = ArrayMesh.new()
	var st = SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var trunk = CylinderMesh.new()
	trunk.radial_segments = 6
	trunk.rings = 1
	if kind == "pine":
		trunk.top_radius = 0.012
		trunk.bottom_radius = 0.025
		trunk.height = 0.35
		st.append_from(trunk, 0, Transform3D(Basis(), Vector3(0, 0.175, 0)))
	else:
		trunk.top_radius = 0.02
		trunk.bottom_radius = 0.035
		trunk.height = 0.5
		st.append_from(trunk, 0, Transform3D(Basis(), Vector3(0, 0.25, 0)))
	st.commit(out)
	out.surface_set_material(0, mat_color(Color(0.30, 0.22, 0.15)))
	var st2 = SurfaceTool.new()
	st2.begin(Mesh.PRIMITIVE_TRIANGLES)
	var r = 0.0
	var y0 = 0.0
	if kind == "pine":
		var tiers = [[0.22, 0.42, 0.42], [0.17, 0.38, 0.62], [0.11, 0.32, 0.82]]
		for t in tiers:
			var cone = CylinderMesh.new()
			cone.top_radius = 0.0
			cone.bottom_radius = t[0]
			cone.height = t[1]
			cone.radial_segments = 8
			cone.rings = 1
			st2.append_from(cone, 0, Transform3D(Basis(), Vector3(0, t[2], 0)))
		r = 0.2
		y0 = 0.22
	else:
		var blobs = [[0.30, Vector3(0, 0.70, 0)], [0.22, Vector3(0.14, 0.58, 0.05)], [0.2, Vector3(-0.12, 0.62, -0.08)]]
		for b in blobs:
			var s = SphereMesh.new()
			s.radius = b[0]
			s.height = b[0] * 1.8
			s.radial_segments = 8
			s.rings = 5
			st2.append_from(s, 0, Transform3D(Basis(), b[1]))
		r = 0.3
		y0 = 0.42
	st2.commit(out)
	out.surface_set_material(1, crown_mat(Color(0.20, 0.34, 0.14) if kind == "pine" else Color(0.26, 0.40, 0.16)))
	return [out, r, y0]


# Модель из .glb (например Kenney Nature Kit), сведённая в одну сетку с основанием в 0 и высотой 1.
func glb_mesh(path: String) -> Array:
	var doc = GLTFDocument.new()
	var state = GLTFState.new()
	if doc.append_from_file(path, state) != OK:
		push_warning("не открывается " + path)
		return []
	var root: Node = doc.generate_scene(state)
	if root == null:
		return []
	var merged = ArrayMesh.new()
	for mi in root.find_children("*", "MeshInstance3D", true, false):
		var xf = Transform3D()
		var n: Node = mi
		while n != root and n is Node3D:
			xf = (n as Node3D).transform * xf
			n = n.get_parent()
		var m: Mesh = mi.mesh
		for s in m.get_surface_count():
			var st = SurfaceTool.new()
			st.begin(Mesh.PRIMITIVE_TRIANGLES)
			st.append_from(m, s, xf)
			st.commit(merged)
			var mat: Material = mi.get_surface_override_material(s)
			if mat == null:
				mat = m.surface_get_material(s)
			merged.surface_set_material(merged.get_surface_count() - 1, mat)
	root.free()
	if merged.get_surface_count() == 0:
		return []
	var bb = merged.get_aabb()
	var h = maxf(bb.size.y, 1e-3)
	var fix = Transform3D(Basis().scaled(Vector3.ONE / h), Vector3.ZERO) * \
		Transform3D(Basis(), Vector3(-bb.get_center().x, -bb.position.y, -bb.get_center().z))
	return [merged, maxf(bb.size.x, bb.size.z) * 0.5 / h, 0.3, fix]


func list_glb(sub: String) -> Array:
	var dir = str(cfg.get("world", {}).get("assets", "res://assets"))
	if dir.begins_with("res://"):
		dir = ProjectSettings.globalize_path(dir)
	var d = DirAccess.open(dir + "/" + sub)
	var res = []
	if d == null:
		return res
	for f in d.get_files():
		if f.to_lower().ends_with(".glb") or f.to_lower().ends_with(".gltf"):
			res.append(dir + "/" + sub + "/" + f)
	res.sort()
	return res


func tree_positions() -> Array:
	var w: Dictionary = cfg.get("world", {})
	var pos = []
	var clear = float(w.get("clear_radius", 35.0))
	# лесной массив вокруг: плотность по шуму (поляны и опушки)
	var n = int(w.get("trees", 2500))
	var rmin = maxf(float(w.get("trees_from", 60.0)), clear)
	var rmax = float(w.get("trees_to", 1200.0))
	var tries = 0
	while pos.size() < n and tries < n * 30:
		tries += 1
		var r = sqrt(rng.randf_range(rmin * rmin, rmax * rmax))
		var a = rng.randf() * TAU
		var x = sin(a) * r
		var z = -cos(a) * r
		if noise.get_noise_2d(x, z) * 0.5 + 0.5 < float(w.get("forest_threshold", 0.45)):
			continue
		pos.append(Vector2(x, z))
	# опушки: плотные полосы деревьев (кромка — главный источник ложных захватов)
	for e in w.get("tree_lines", []):
		var dist = float(e.get("dist", 150.0))
		var depth = float(e.get("depth", 30.0))
		var a0 = float(e.get("az_from", -60.0))
		var a1 = float(e.get("az_to", 60.0))
		var cnt = int(e.get("count", 400))
		for i in cnt:
			var a = lerpf(a0, a1, rng.randf()) * DEG
			var r = dist + rng.randf() * depth + noise.get_noise_1d(a * 300.0) * depth * 0.5
			pos.append(Vector2(sin(a) * r, -cos(a) * r))
	return pos


func build_trees() -> void:
	var w: Dictionary = cfg.get("world", {})
	var kinds = []    # [mesh, r, y0, fix, hmin, hmax]
	var assets = list_glb("trees")
	for p in assets:
		var g = glb_mesh(p)
		if g.size() > 0:
			kinds.append([g[0], g[1], g[2], g[3], 8.0, 22.0])
	if kinds.is_empty():
		var pine = proc_tree("pine")
		var leaf = proc_tree("leaf")
		var pf = float(w.get("pine_share", 0.5))
		var np = int(round(pf * 10))
		for i in np:
			kinds.append([pine[0], pine[1], pine[2], Transform3D(), 12.0, 28.0])
		for i in 10 - np:
			kinds.append([leaf[0], leaf[1], leaf[2], Transform3D(), 8.0, 20.0])
	var per = {}
	for p in tree_positions():
		var k = rng.randi() % kinds.size()
		if not per.has(k):
			per[k] = []
		var h = rng.randf_range(kinds[k][4], kinds[k][5]) * float(w.get("tree_scale", 1.0))
		var wide = rng.randf_range(0.8, 1.25)
		per[k].append([p, h, wide, rng.randf() * TAU, rng.randf_range(0.8, 1.15)])
	tree_buckets.resize(360)
	for i in 360:
		tree_buckets[i] = PackedInt32Array()
	for k in per.keys():
		var lst: Array = per[k]
		var mm = MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = true
		mm.mesh = kinds[k][0]
		mm.instance_count = lst.size()
		for i in lst.size():
			var it: Array = lst[i]
			var p: Vector2 = it[0]
			var h: float = it[1]
			var b = Basis(Vector3.UP, it[3]).scaled(Vector3(h * it[2], h, h * it[2]))
			mm.set_instance_transform(i, Transform3D(b, Vector3(p.x, 0, p.y)) * kinds[k][3])
			var c: float = it[4]
			mm.set_instance_color(i, Color(c, c * rng.randf_range(0.95, 1.05), c))
			add_tree_occluder(p, kinds[k][1] * h * it[2] * 0.85, kinds[k][2] * h, h)
		var mmi = MultiMeshInstance3D.new()
		mmi.multimesh = mm
		world.add_child(mmi)


func add_tree_occluder(p: Vector2, r: float, y0: float, y1: float) -> void:
	var idx = trees.size()
	trees.append([p.x, p.y, r, y0, y1])
	var d = Vector2(p.x - rig_pos.x, p.y - rig_pos.z)
	var az = fposmod(atan2(d.x, -d.y) / DEG, 360.0)
	var half = atan2(r, maxf(d.length(), 1e-3)) / DEG + 0.6
	if d.length() <= r:
		half = 180.0
	var lo = int(floor(az - half))
	var hi = int(floor(az + half))
	for a in range(lo, hi + 1):
		var arr: PackedInt32Array = tree_buckets[posmod(a, 360)]
		arr.append(idx)
		tree_buckets[posmod(a, 360)] = arr


# ---------- дома и столбы ----------

func build_houses() -> void:
	var w: Dictionary = cfg.get("world", {})
	var assets = list_glb("houses")
	var models = []
	for p in assets:
		var g = glb_mesh(p)
		if g.size() > 0:
			models.append(g)
	var walls = [Color(0.85, 0.82, 0.74), Color(0.70, 0.55, 0.42), Color(0.62, 0.66, 0.70), Color(0.90, 0.88, 0.85), Color(0.55, 0.40, 0.30)]
	var roofs = [Color(0.45, 0.18, 0.14), Color(0.30, 0.32, 0.35), Color(0.20, 0.35, 0.25), Color(0.55, 0.55, 0.55)]
	for v in w.get("villages", []):
		var vaz = float(v.get("az", 60.0)) * DEG
		var vd = float(v.get("dist", 250.0))
		var vr = float(v.get("radius", 80.0))
		var c = Vector2(sin(vaz) * vd, -cos(vaz) * vd)
		var placed = []
		for i in int(v.get("houses", 15)):
			var p = Vector2.ZERO
			var ok = false
			for tr in 40:
				p = c + Vector2(rng.randf_range(-vr, vr), rng.randf_range(-vr, vr))
				ok = true
				for q in placed:
					if p.distance_to(q) < 16.0:
						ok = false
						break
				if ok:
					break
			if not ok:
				continue
			placed.append(p)
			var rot = rng.randf() * TAU
			if models.size() > 0:
				var g: Array = models[rng.randi() % models.size()]
				var h = rng.randf_range(6.0, 10.0)
				var mi = MeshInstance3D.new()
				mi.mesh = g[0]
				mi.transform = Transform3D(Basis(Vector3.UP, rot).scaled(Vector3.ONE * h), Vector3(p.x, 0, p.y)) * g[3]
				world.add_child(mi)
				boxes.append(mi.transform * g[0].get_aabb())
				continue
			var bw = rng.randf_range(6.0, 12.0)
			var bd = rng.randf_range(6.0, 10.0)
			var bh = rng.randf_range(3.0, 7.0)
			var rh = rng.randf_range(2.0, 4.0)
			var node = Node3D.new()
			node.transform = Transform3D(Basis(Vector3.UP, rot), Vector3(p.x, 0, p.y))
			var body = MeshInstance3D.new()
			var bm = BoxMesh.new()
			bm.size = Vector3(bw, bh, bd)
			body.mesh = bm
			body.position.y = bh * 0.5
			body.material_override = mat_color(walls[rng.randi() % walls.size()])
			node.add_child(body)
			var roof = MeshInstance3D.new()
			var pr = PrismMesh.new()
			pr.size = Vector3(bw + 0.6, rh, bd + 0.6)
			roof.mesh = pr
			roof.position.y = bh + rh * 0.5
			roof.material_override = mat_color(roofs[rng.randi() % roofs.size()], 0.8)
			node.add_child(roof)
			world.add_child(node)
			boxes.append(node.transform * AABB(Vector3(-bw / 2, 0, -bd / 2), Vector3(bw, bh + rh, bd)))


func build_poles() -> void:
	# столбы и мачты: тонкие вертикальные цели у горизонта (верхушка мачты — частый ложный захват)
	var w: Dictionary = cfg.get("world", {})
	for p in w.get("poles", []):
		var pos = enu(p.get("pos", [0, 100, 0]))
		var h = float(p.get("height", 20.0))
		var r = float(p.get("radius", 0.15))
		var mi = MeshInstance3D.new()
		var cm = CylinderMesh.new()
		cm.top_radius = r * 0.7
		cm.bottom_radius = r
		cm.height = h
		cm.radial_segments = 8
		mi.mesh = cm
		mi.position = Vector3(pos.x, h * 0.5, pos.z)
		mi.material_override = mat_color(Color(0.35, 0.33, 0.30))
		world.add_child(mi)
		boxes.append(AABB(Vector3(pos.x - r, 0, pos.z - r), Vector3(2 * r, h, 2 * r)))


# ---------- цели ----------

func part(parent: Node3D, mesh: Mesh, pos: Vector3, mat: Material, rot := Basis()) -> MeshInstance3D:
	var mi = MeshInstance3D.new()
	mi.mesh = mesh
	mi.transform = Transform3D(rot, pos)
	mi.material_override = mat
	parent.add_child(mi)
	return mi


func box(sz: Vector3) -> BoxMesh:
	var b = BoxMesh.new()
	b.size = sz
	return b


# Случайные цели: "random_targets": {"drones": 6, "birds": 3, "range": [150, 1500], "seed": 5,
#   "types": {"quad": 0.5, "delta": 0.25, "plane": 0.25}} — разные типы, размеры и скорости,
#   пролёт по прямой через поле зрения широкой камеры.
func random_targets() -> Array:
	var rc: Dictionary = cfg.get("random_targets", {})
	var out = []
	if rc.is_empty():
		return out
	var r = RandomNumberGenerator.new()
	r.seed = int(rc.get("seed", int(cfg.get("seed", 1)) + 100))
	var rng_r = rc.get("range", [150, 1500])
	var types: Dictionary = rc.get("types", {"quad": 0.5, "delta": 0.25, "plane": 0.25})
	var tot = 0.0
	for k in types:
		tot += float(types[k])
	var half = float(wcfg.get("hfov", 120.0)) * 0.4
	var az0 = float(wcfg.get("az", 0.0))
	var n_dr = int(rc.get("drones", 6))
	for i in n_dr + int(rc.get("birds", 3)):
		var t = {}
		var model = "bird"
		var size = r.randf_range(0.3, 1.2)
		var speed = r.randf_range(8, 14)
		if i < n_dr:
			var x = r.randf() * tot
			for k in types:
				model = k
				x -= float(types[k])
				if x <= 0:
					break
			if model == "quad":
				size = r.randf_range(0.3, 0.8)
				speed = r.randf_range(5, 20)
			elif model == "delta":
				size = r.randf_range(1.8, 3.5)
				speed = r.randf_range(25, 45)
			else:
				size = r.randf_range(1.5, 4.0)
				speed = r.randf_range(15, 30)
			t["kind"] = "drone"
		else:
			t["kind"] = "bird"
			t["flap_hz"] = r.randf_range(3, 8)
			t["flap_period"] = r.randf_range(2, 5)
			t["flap_on"] = r.randf_range(0.8, 2)
		t["model"] = model
		t["size"] = size
		t["speed"] = speed
		var dist = r.randf_range(float(rng_r[0]), float(rng_r[1]))
		var az = (az0 + r.randf_range(-half, half)) * DEG
		var el = r.randf_range(3, 25) * DEG
		var c = Vector3(sin(az) * cos(el), cos(az) * cos(el), sin(el)) * dist   # восток, север, верх
		var hd = r.randf() * TAU
		var dirv = Vector3(cos(hd), sin(hd), r.randf_range(-0.05, 0.05))
		var secs = float(cfg.get("seconds", 10.0))
		var a = c - dirv * speed * secs * 0.5
		var b = c + dirv * speed * secs * 0.5
		t["path"] = {"type": "line", "from": [a.x, a.y, maxf(a.z, 15.0)], "to": [b.x, b.y, maxf(b.z, 15.0)]}
		out.append(t)
	return out


func build_targets() -> void:
	var id = 1
	for t in cfg.get("targets", []) + random_targets():
		var kind = str(t.get("kind", "drone"))
		var model = str(t.get("model", "quad" if kind == "drone" else "bird"))
		var s = float(t.get("size", 0.35 if kind == "drone" else 0.8))
		var c = col(t.get("color"), Color(0.12, 0.12, 0.13) if kind == "drone" else Color(0.10, 0.09, 0.08))
		var m = mat_color(c, 0.6)
		var root = Node3D.new()
		var wings = []
		if model == "quad":
			part(root, box(Vector3(0.32 * s, 0.11 * s, 0.32 * s)), Vector3.ZERO, m)
			var prop = CylinderMesh.new()
			prop.top_radius = 0.17 * s
			prop.bottom_radius = 0.17 * s
			prop.height = 0.01 * s
			prop.radial_segments = 12
			var pm = mat_color(Color(0.25, 0.25, 0.27), 0.5)
			pm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			pm.albedo_color.a = 0.35
			for k in 4:
				var a = PI / 4 + k * PI / 2
				var dirv = Vector3(sin(a), 0, cos(a))
				part(root, box(Vector3(0.04 * s, 0.03 * s, 0.36 * s)), dirv * 0.18 * s, m, Basis(Vector3.UP, a))
				part(root, box(Vector3(0.06 * s, 0.07 * s, 0.06 * s)), dirv * 0.35 * s + Vector3(0, 0.03 * s, 0), m)
				part(root, prop, dirv * 0.35 * s + Vector3(0, 0.07 * s, 0), pm)
		elif model == "delta":
			# летающее крыло: треугольник с вырезом сзади (размах s), кили на концах
			var L = 0.62 * s
			var st = SurfaceTool.new()
			st.begin(Mesh.PRIMITIVE_TRIANGLES)
			var nose = Vector3(0, 0, -0.55 * L)
			var lt = Vector3(-0.5 * s, 0, 0.45 * L)
			var rt = Vector3(0.5 * s, 0, 0.45 * L)
			var notch = Vector3(0, 0, 0.25 * L)
			for v in [nose, lt, notch, nose, notch, rt]:
				st.add_vertex(v)
			st.generate_normals()
			var dm = m.duplicate()
			dm.cull_mode = BaseMaterial3D.CULL_DISABLED
			part(root, st.commit(), Vector3.ZERO, dm)
			part(root, box(Vector3(0.06 * s, 0.07 * s, 0.8 * L)), Vector3(0, 0, 0.05 * L), m)
			for x in [-0.47 * s, 0.47 * s]:
				part(root, box(Vector3(0.01 * s, 0.1 * s, 0.2 * L)), Vector3(x, 0.05 * s, 0.35 * L), m)
		elif model == "wing" or model == "plane":
			part(root, box(Vector3(0.12 * s, 0.12 * s, 0.75 * s)), Vector3.ZERO, m)
			part(root, box(Vector3(s, 0.02 * s, 0.16 * s)), Vector3(0, 0.03 * s, -0.05 * s), m)
			part(root, box(Vector3(0.35 * s, 0.02 * s, 0.1 * s)), Vector3(0, 0.03 * s, 0.33 * s), m)
			part(root, box(Vector3(0.02 * s, 0.14 * s, 0.1 * s)), Vector3(0, 0.1 * s, 0.33 * s), m)
		else:
			# птица: тело и два крыла на шарнирах, size — размах крыльев
			var body = CapsuleMesh.new()
			body.radius = 0.06 * s
			body.height = 0.36 * s
			body.radial_segments = 8
			body.rings = 2
			part(root, body, Vector3.ZERO, m, Basis(Vector3.RIGHT, PI / 2))
			for side in [-1.0, 1.0]:
				var hinge = Node3D.new()
				hinge.position = Vector3(side * 0.04 * s, 0.02 * s, -0.02 * s)
				root.add_child(hinge)
				part(hinge, box(Vector3(0.46 * s, 0.012 * s, 0.17 * s)), Vector3(side * 0.23 * s, 0, 0), m)
				wings.append([hinge, side])
			part(root, box(Vector3(0.12 * s, 0.01 * s, 0.1 * s)), Vector3(0, 0, 0.2 * s), m)
		world.add_child(root)
		targets.append({"id": int(t.get("id", id)), "kind": kind, "cfg": t, "node": root, "wings": wings, "size": s})
		id += 1


# Положение цели по траектории (координаты Godot) или null, если цель ещё/уже не в небе.
func target_pos(tg: Dictionary, t: float):
	var c: Dictionary = tg["cfg"]
	var t0 = float(c.get("start", 0.0))
	var t1 = float(c.get("end", 1e9))
	if t < t0 or t > t1:
		return null
	var path: Dictionary = c.get("path", {})
	var tt = t - t0
	var sp = float(path.get("speed", c.get("speed", 10.0)))
	var typ = str(path.get("type", "line"))
	if typ == "orbit":
		var ctr = enu(path.get("center", [0, 300, 60]))
		var r = float(path.get("radius", 80.0))
		var dirn = float(path.get("dir", 1.0))
		var a = float(path.get("phase", 0.0)) * DEG + dirn * sp * tt / r
		return ctr + Vector3(sin(a) * r, 0, -cos(a) * r)
	if typ == "hover":
		var p0 = enu(path.get("pos", [0, 200, 50]))
		var j = float(path.get("jitter", 1.0))
		var ph = float(tg["id"]) * 1.7
		return p0 + Vector3(sin(tt * 0.7 + ph) + 0.5 * sin(tt * 1.9 + ph), 0.6 * sin(tt * 1.1 + ph * 2.0), cos(tt * 0.9 + ph) + 0.4 * sin(tt * 2.3)) * j
	if typ == "wander":
		var ctr = enu(path.get("center", [0, 200, 40]))
		var r = float(path.get("radius", 60.0))
		var ph = float(tg["id"]) * 2.3
		var w = sp / r
		return ctr + Vector3(r * sin(w * tt + ph), 0.25 * r * sin(0.6 * w * tt + ph * 1.3), -r * 0.7 * sin(0.77 * w * tt + ph * 0.7))
	# line / waypoints: ломаная с постоянной скоростью; loop — по кругу, pingpong — туда-обратно
	var pts = []
	if typ == "line":
		pts = [enu(path.get("from", [-100, 200, 50])), enu(path.get("to", [100, 200, 60]))]
	else:
		for p in path.get("points", []):
			pts.append(enu(p))
	if pts.size() == 0:
		return null
	if pts.size() == 1:
		return pts[0]
	var mode = str(path.get("mode", "once"))
	if mode == "loop":
		pts.append(pts[0])
	var total = 0.0
	for i in pts.size() - 1:
		total += (pts[i + 1] - pts[i]).length()
	var s = sp * tt
	if mode == "pingpong":
		s = fposmod(s, 2.0 * total)
		if s > total:
			s = 2.0 * total - s
	elif mode == "loop":
		s = fposmod(s, total)
	elif s > total:
		# дальше по прямой последнего отрезка
		var dl: Vector3 = (pts[-1] - pts[-2]).normalized()
		return pts[-1] + dl * (s - total)
	for i in pts.size() - 1:
		var l = (pts[i + 1] - pts[i]).length()
		if s <= l or i == pts.size() - 2:
			return pts[i].lerp(pts[i + 1], clampf(s / maxf(l, 1e-6), 0.0, 1.0))
		s -= l
	return pts[-1]


func update_targets(t: float) -> void:
	for tg in targets:
		var p = target_pos(tg, t)
		var node: Node3D = tg["node"]
		if p == null:
			node.visible = false
			continue
		node.visible = true
		var p2 = target_pos(tg, t + 0.1)
		var v = Vector3.ZERO
		if p2 != null:
			v = (p2 - p) / 0.1
		var fwd = Vector3(v.x, 0, v.z)
		var b = Basis()
		if fwd.length() > 0.05:
			b = Basis.looking_at(fwd.normalized(), Vector3.UP)
			if tg["kind"] == "drone" and str(tg["cfg"].get("model", "quad")) == "quad":
				b = b * Basis(Vector3.RIGHT, -minf(fwd.length() * 1.2, 25.0) * DEG)   # наклон вперёд
		node.global_transform = Transform3D(b, p)
		if tg["wings"].size() > 0:
			var c: Dictionary = tg["cfg"]
			var f = float(c.get("flap_hz", 4.0))
			var amp = float(c.get("flap_deg", 40.0)) * DEG
			var ang = amp * sin(TAU * f * t + float(tg["id"]))
			# планирование: взмахи пачками flap_on секунд из каждых flap_period
			var per = float(c.get("flap_period", 0.0))
			if per > 0.0 and fposmod(t + tg["id"] * 0.37, per) > float(c.get("flap_on", per * 0.5)):
				ang = 0.12
			for wg in tg["wings"]:
				var hinge: Node3D = wg[0]
				hinge.transform.basis = Basis(Vector3.BACK, ang * wg[1])


# ---------- заслонение ----------

func seg_hits_cyl(c: Vector3, p: Vector3, tr: Array) -> bool:
	var d = p - c
	var fx = c.x - float(tr[0])
	var fz = c.z - float(tr[1])
	var r = float(tr[2])
	var a = d.x * d.x + d.z * d.z
	if a < 1e-9:
		return false
	var b = 2.0 * (fx * d.x + fz * d.z)
	var cc = fx * fx + fz * fz - r * r
	var disc = b * b - 4.0 * a * cc
	if disc < 0.0:
		return false
	var sq = sqrt(disc)
	var t0 = maxf((-b - sq) / (2.0 * a), 0.0)
	var t1 = minf((-b + sq) / (2.0 * a), 1.0)
	if t0 > t1:
		return false
	var ya = c.y + d.y * t0
	var yb = c.y + d.y * t1
	return maxf(ya, yb) >= float(tr[3]) and minf(ya, yb) <= float(tr[4])


func occluded(c: Vector3, p: Vector3) -> bool:
	if p.y <= 0.0:
		return true
	var d = p - rig_pos
	var az = int(floor(fposmod(atan2(d.x, -d.z) / DEG, 360.0)))
	for i in tree_buckets[az]:
		if seg_hits_cyl(c, p, trees[i]):
			return true
	for bb in boxes:
		var ab: AABB = bb
		if ab.intersects_segment(c, p) != null:
			return true
	return false


func project(cam: Camera3D, w: int, h: int, tg: Dictionary):
	var node: Node3D = tg["node"]
	if not node.visible:
		return null
	var p = node.global_position
	if cam.is_position_behind(p):
		return null
	var px = cam.unproject_position(p) - Vector2(0.5, 0.5)   # центр пикселя 0 — координата 0
	if px.x < 0 or px.y < 0 or px.x > w - 1 or px.y > h - 1:
		return null
	if occluded(cam.global_position, p):
		return null
	var dist = (p - cam.global_position).length()
	var ppx = float(tg["size"]) / (2.0 * dist * tan(cam.fov * DEG * 0.5)) * w
	return [px, dist, ppx]


func gt_frame(cam: Camera3D, w: int, h: int) -> Array:
	var rec = []
	for tg in targets:
		var pr = project(cam, w, h, tg)
		if pr == null:
			continue
		rec.append({"kind": tg["kind"], "bright": false, "x": snappedf(pr[0].x, 0.01), "y": snappedf(pr[0].y, 0.01),
			"id": tg["id"], "dist": snappedf(pr[1], 0.1), "px": snappedf(pr[2], 0.01)})
	return rec


# ---------- поворотка точной камеры ----------

func pick_fine_target() -> int:
	var want = fcfg.get("target", "auto")
	if typeof(want) != TYPE_STRING:
		for i in targets.size():
			if targets[i]["id"] == int(want):
				return i if targets[i]["node"].visible else -1
		return -1
	if f_target >= 0 and project(wide_cam, wide_vp.size.x, wide_vp.size.y, targets[f_target]) != null:
		return f_target
	for i in targets.size():
		if targets[i]["kind"] == "drone" and project(wide_cam, wide_vp.size.x, wide_vp.size.y, targets[i]) != null:
			return i
	return -1


# Ведение: на цель с задержкой передачи latency, скорость ≤ max_dps, ускорение ≤ accel по каждой оси.
func turret_to(t: float) -> void:
	var vmax = float(fcfg.get("max_dps", 30.0))
	var acc = float(fcfg.get("accel", 150.0))
	var lat = float(fcfg.get("latency", 0.06))
	var kp = float(fcfg.get("gain", 4.0))
	while f_t < t - 1e-9:
		var dt = minf(1.0 / 240.0, t - f_t)
		f_t += dt
		var daz = 0.0
		var del = 0.0
		if f_target >= 0:
			var p = target_pos(targets[f_target], f_t - lat)
			var p2 = target_pos(targets[f_target], f_t - lat + 0.05)
			if p != null and p2 != null:
				var a1 = azel_of(p - rig_pos)
				var a2 = azel_of(p2 - rig_pos)
				var raz = wrap180(a2.x - a1.x) / 0.05
				var rel = (a2.y - a1.y) / 0.05
				daz = clampf(raz + kp * wrap180(a1.x - f_az), -vmax, vmax)
				del = clampf(rel + kp * (a1.y - f_el), -vmax, vmax)
		f_vaz = move_toward(f_vaz, daz, acc * dt)
		f_vel = move_toward(f_vel, del, acc * dt)
		f_az = fposmod(f_az + f_vaz * dt, 360.0)
		f_el = clampf(f_el + f_vel * dt, -10.0, 89.0)


# ---------- кадры ----------

func set_time(t: float) -> void:
	for m in time_mats:
		(m as ShaderMaterial).set_shader_parameter("sim_time", t)
	update_targets(t)
	var wp = wide_pose(t)
	aim(wide_cam, wp.x, wp.y, float(wcfg.get("roll", 0.0)))
	if fine_enabled():
		f_target = pick_fine_target()
		turret_to(t)
		aim(fine_cam, f_az, f_el)


func run() -> void:
	var secs = float(cfg.get("seconds", 10.0))
	var wfps = float(wcfg.get("fps", 14.0))
	var ffps = float(fcfg.get("fps", 60.0))
	var ev = []
	for k in int(round(secs * wfps)):
		ev.append([k / wfps, 0, k])
	if fine_enabled():
		for j in int(round(secs * ffps)):
			ev.append([j / ffps, 1, j])
	ev.sort_custom(func(a, b): return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]))

	# прогрев: шейдеры компилируются на первых кадрах
	set_time(0.0)
	for _w in 3:
		wide_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		fine_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		await RenderingServer.frame_post_draw

	var t_start = Time.get_ticks_msec()
	var i = 0
	while i < ev.size():
		var t: float = ev[i][0]
		var group = []
		while i < ev.size() and absf(ev[i][0] - t) < 1e-6:
			group.append(ev[i])
			i += 1
		set_time(t)
		for e in group:
			var vpu: SubViewport = wide_vp if e[1] == 0 else fine_vp
			vpu.render_target_update_mode = SubViewport.UPDATE_ONCE
		await RenderingServer.frame_post_draw
		for e in group:
			var vp: SubViewport = wide_vp if e[1] == 0 else fine_vp
			var cam: Camera3D = wide_cam if e[1] == 0 else fine_cam
			var img = vp.get_texture().get_image()
			if img.get_format() != Image.FORMAT_RGB8:
				img.convert(Image.FORMAT_RGB8)
			var sub = "wide" if e[1] == 0 else "fine"
			img.save_png("%s/%s/f_%06d.png" % [out_dir, sub, e[2]])
			var rec = gt_frame(cam, vp.size.x, vp.size.y)
			var pose = azel_of(-cam.global_transform.basis.z)
			if e[1] == 0:
				gt_wide.append(rec)
				poses_wide.append([snappedf(t, 1e-4), snappedf(pose.x, 1e-3), snappedf(pose.y, 1e-3)])
			else:
				gt_fine.append(rec)
				var tid = targets[f_target]["id"] if f_target >= 0 else null
				poses_fine.append([snappedf(t, 1e-4), snappedf(f_az, 1e-3), snappedf(f_el, 1e-3),
					snappedf(f_vaz, 1e-3), snappedf(f_vel, 1e-3), tid])
		if gt_wide.size() % 14 == 1 and group[0][1] == 0:
			print("кадр %d/%d широкой, %.1f с" % [gt_wide.size(), int(round(secs * wfps)), (Time.get_ticks_msec() - t_start) / 1000.0])
	write_outputs(secs, wfps, ffps)
	print("готово: %d кадров широкой, %d точной, %.1f с" % [gt_wide.size(), gt_fine.size(), (Time.get_ticks_msec() - t_start) / 1000.0])


func save_json(path: String, obj) -> void:
	var f = FileAccess.open(path, FileAccess.WRITE)
	f.store_string(JSON.stringify(obj))
	f.close()


func write_outputs(secs: float, wfps: float, ffps: float) -> void:
	var base = str(cfg.get("name", "sim"))
	var has_drone = false
	for tg in targets:
		if tg["kind"] == "drone":
			has_drone = true
	var gw = {"w": wide_vp.size.x, "h": wide_vp.size.y, "fps": wfps, "source": "dronolok_sim", "frames": gt_wide}
	if not has_drone:
		gw["no_drones"] = true
	save_json("%s/%s_wide.mp4.gt.json" % [out_dir, base], gw)
	if fine_enabled():
		save_json("%s/%s_fine.mp4.gt.json" % [out_dir, base],
			{"w": fine_vp.size.x, "h": fine_vp.size.y, "fps": ffps, "source": "dronolok_sim", "frames": gt_fine})
	var tracks = []
	for tg in targets:
		var pts = []
		for k in int(round(secs * wfps)):
			var p = target_pos(tg, k / wfps)
			if p != null:
				pts.append([snappedf(k / wfps, 1e-4), snappedf(p.x, 0.01), snappedf(-p.z, 0.01), snappedf(p.y, 0.01)])
		tracks.append({"id": tg["id"], "kind": tg["kind"], "size": tg["size"], "enu": pts})
	save_json(out_dir + "/meta.json", {
		"name": base, "seconds": secs, "config": cfg,
		"wide": {"w": wide_vp.size.x, "h": wide_vp.size.y, "fps": wfps, "hfov": wide_cam.fov, "poses": poses_wide},
		"fine": {"enabled": fine_enabled(), "w": fine_vp.size.x, "h": fine_vp.size.y, "fps": ffps, "hfov": fine_cam.fov,
			"poses": poses_fine, "pose_fields": ["t", "az", "el", "vaz", "vel", "target"]},
		"rig_height": rig_pos.y, "targets": tracks})
