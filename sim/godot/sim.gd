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
#   [--sector forest|lowrise|highrise] [--az A] [--el E] [--pan DPS] [--sweep DEG] — куда смотрит широкая камера.
#   [--panorama 1] — вместо ролика 12 снимков по кругу (pano_NN.png) для склейки панорамы.
#   --view 1 — окно просмотра: стрелки поворачивают камеру, 1/2/3 — сектора, см. sim/README.md.
#   --live <порт> — источник кадров эмулятора двух камер Dronolok (app/src/scenelink.hpp): сцена
#   подключается к приложению и рисует кадры по запросу, цели летают без конца (новые вместо улетевших).

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
var boxes: Array = []          # [обратное преобразование, AABB в своих осях] домов и столбов
var box_buckets: Array = []    # 360 корзин по азимуту -> индексы boxes
var view_mode := false         # окно просмотра (--view)
var view_az := 0.0
var view_el := 15.0
var view_t := 0.0
var view_speed := 1.0
var view_pan := false
var view_hud: Label
const LIVE_PATH_S := 40.0       # --live: столько секунд летит случайная цель, потом новая
var live_mode := false
var live_peer: StreamPeerTCP
var live_buf := PackedByteArray()
var live_next_id := 1000

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
	view_mode = args.has("view")
	if not cfg.has("world"):
		cfg["world"] = {}
	wcfg = cfg.get("wide", {})
	fcfg = cfg.get("fine", {})
	cfg["wide"] = wcfg
	# куда смотрит широкая камера: сектор (его середина) или явный азимут; имя ролика с суффиксом
	if args.has("sector"):
		var sc = sector_by_name(str(args["sector"]))
		if sc.is_empty():
			push_error("нет сектора " + str(args["sector"]) + " в world.sectors")
			get_tree().quit(2)
			return
		wcfg["az"] = sector_mid(sc)
		cfg["name"] = str(cfg.get("name", "sim")) + "_" + str(args["sector"])
	if args.has("az"):
		wcfg["az"] = float(args["az"])
		cfg["name"] = str(cfg.get("name", "sim")) + "_az%d" % int(round(float(args["az"])))
	if args.has("el"):
		wcfg["el"] = float(args["el"])
	if args.has("pan"):
		wcfg["pan_dps"] = float(args["pan"])
	if args.has("sweep"):
		wcfg["sweep_deg"] = float(args["sweep"])
	if args.has("live") and not live_connect(int(args["live"])):
		get_tree().quit(3)
		return
	if view_mode:
		get_window().size = Vector2i(1280, 800)
		get_window().move_to_center()
		wcfg["w"] = 1280
		wcfg["h"] = 800
		scale = 1.0
	if scale != 1.0:
		for c in [wcfg, fcfg]:
			c["w"] = int(round(float(c.get("w", 640)) * scale))
			c["h"] = int(round(float(c.get("h", 480)) * scale))
	rng.seed = int(cfg.get("seed", 1))
	noise.seed = int(cfg.get("seed", 1))
	noise.frequency = 0.004
	Engine.max_fps = 60 if view_mode else 0
	if not view_mode and not live_mode:
		DirAccess.make_dir_recursive_absolute(out_dir + "/wide")
		if fine_enabled() and not args.has("panorama"):
			DirAccess.make_dir_recursive_absolute(out_dir + "/fine")

	setup_viewports()
	build_environment()
	build_ground()
	build_hills()
	build_trees()
	build_houses()
	build_lowrise()
	build_highrise()
	build_poles()
	bucket_boxes()
	build_targets()
	if view_mode:
		start_view()
		return
	if live_mode:
		await live_loop()
		get_tree().quit()
		return
	if args.has("panorama"):
		await run_panorama()
	else:
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
		if not sectors().is_empty() and rng.randf() > sector_trees(sector_at(a / DEG)):
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

const WALLS := [Color(0.85, 0.82, 0.74), Color(0.70, 0.55, 0.42), Color(0.62, 0.66, 0.70), Color(0.90, 0.88, 0.85), Color(0.55, 0.40, 0.30)]
const ROOFS := [Color(0.45, 0.18, 0.14), Color(0.30, 0.32, 0.35), Color(0.20, 0.35, 0.25), Color(0.55, 0.55, 0.55)]

var house_models: Array = []
var house_items: Array = []    # процедурные дома: [Transform3D основания, ширина, глубина, высота стен, высота крыши]


func load_house_models() -> void:
	if not house_models.is_empty():
		return
	for p in list_glb("houses"):
		var g = glb_mesh(p)
		if g.size() > 0:
			house_models.append(g)
	if house_models.is_empty():
		house_models.append([])   # признак «моделей нет», чтобы не искать снова


# Один дом в точке p (координаты Godot x, z), rot — поворот вокруг вертикали.
func place_house(p: Vector2, rot: float) -> void:
	if house_models[0].size() > 0:
		var g: Array = house_models[rng.randi() % house_models.size()]
		var h = rng.randf_range(6.0, 10.0)
		var mi = MeshInstance3D.new()
		mi.mesh = g[0]
		mi.transform = Transform3D(Basis(Vector3.UP, rot).scaled(Vector3.ONE * h), Vector3(p.x, 0, p.y)) * g[3]
		world.add_child(mi)
		add_box(mi.transform, g[0].get_aabb())
		return
	var bw = rng.randf_range(6.0, 12.0)
	var bd = rng.randf_range(6.0, 10.0)
	var bh = rng.randf_range(3.0, 7.0)
	var rh = rng.randf_range(2.0, 4.0)
	var xf = Transform3D(Basis(Vector3.UP, rot), Vector3(p.x, 0, p.y))
	house_items.append([xf, bw, bd, bh, rh, rng.randi() % WALLS.size(), rng.randi() % ROOFS.size()])
	add_box(xf, AABB(Vector3(-bw / 2, 0, -bd / 2), Vector3(bw, bh + rh, bd)))


# Процедурные дома одним MultiMesh на стены и одним на крыши (сотни домов — два вызова отрисовки).
func flush_houses() -> void:
	if house_items.is_empty():
		return
	var mw = StandardMaterial3D.new()
	mw.vertex_color_use_as_albedo = true
	mw.roughness = 1.0
	var mr = StandardMaterial3D.new()
	mr.vertex_color_use_as_albedo = true
	mr.roughness = 0.8
	var bm = BoxMesh.new()
	bm.material = mw
	var pr = PrismMesh.new()
	pr.material = mr
	var parts = [[bm, 0], [pr, 1]]
	for pp in parts:
		var mm = MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = true
		mm.mesh = pp[0]
		mm.instance_count = house_items.size()
		for i in house_items.size():
			var it: Array = house_items[i]
			var xf: Transform3D = it[0]
			var bw: float = it[1]
			var bd: float = it[2]
			var bh: float = it[3]
			var rh: float = it[4]
			if pp[1] == 0:
				mm.set_instance_transform(i, xf * Transform3D(Basis().scaled(Vector3(bw, bh, bd)), Vector3(0, bh * 0.5, 0)))
				mm.set_instance_color(i, WALLS[it[5]])
			else:
				mm.set_instance_transform(i, xf * Transform3D(Basis().scaled(Vector3(bw + 0.6, rh, bd + 0.6)), Vector3(0, bh + rh * 0.5, 0)))
				mm.set_instance_color(i, ROOFS[it[6]])
		var mmi = MultiMeshInstance3D.new()
		mmi.multimesh = mm
		world.add_child(mmi)
	house_items.clear()


func build_houses() -> void:
	var w: Dictionary = cfg.get("world", {})
	if w.get("villages", []).is_empty():
		return
	load_house_models()
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
			place_house(p, rng.randf() * TAU)
	flush_houses()


# ---------- сектора панорамы: лес / малоэтажная / многоэтажная застройка ----------
# world.sectors: [{"type": "forest"|"lowrise"|"highrise", "az_from": 0, "az_to": 120, ...}], азимуты в градусах.

func sectors() -> Array:
	return cfg.get("world", {}).get("sectors", [])


func in_sector(sc: Dictionary, az: float) -> bool:
	var a0 = float(sc.get("az_from", 0.0))
	var span = fposmod(float(sc.get("az_to", 360.0)) - a0, 360.0)
	if span == 0.0:
		span = 360.0
	return fposmod(az - a0, 360.0) < span


func sector_at(az: float) -> Dictionary:
	for sc in sectors():
		if in_sector(sc, az):
			return sc
	return {}


func sector_mid(sc: Dictionary) -> float:
	var a0 = float(sc.get("az_from", 0.0))
	var span = fposmod(float(sc.get("az_to", 360.0)) - a0, 360.0)
	if span == 0.0:
		span = 360.0
	return fposmod(a0 + span * 0.5, 360.0)


func sector_by_name(nm: String) -> Dictionary:
	for sc in sectors():
		if str(sc.get("name", sc.get("type", ""))) == nm or str(sc.get("type", "")) == nm:
			return sc
	return {}


# доля деревьев леса, которая остаётся в секторе (дворы и скверы в застройке)
func sector_trees(sc: Dictionary) -> float:
	var def = {"forest": 1.0, "lowrise": 0.12, "highrise": 0.05}
	return float(sc.get("trees", def.get(str(sc.get("type", "forest")), 1.0)))


# Случайная точка в секторе на расстоянии r0..r1 (равномерно по площади): [Vector2 (x, z), азимут°].
func sector_point(sc: Dictionary, r0: float, r1: float) -> Array:
	var a0 = float(sc.get("az_from", 0.0))
	var span = fposmod(float(sc.get("az_to", 360.0)) - a0, 360.0)
	if span == 0.0:
		span = 360.0
	var az = a0 + rng.randf() * span
	var r = sqrt(rng.randf_range(r0 * r0, r1 * r1))
	return [Vector2(sin(az * DEG) * r, -cos(az * DEG) * r), az]


# Сетка занятости для расстановки без пересечений: ключ — клетка cell×cell м.
func grid_free(grid: Dictionary, p: Vector2, rad: float, cell: float) -> bool:
	var c0 = Vector2i(floor((p.x - rad) / cell), floor((p.y - rad) / cell))
	var c1 = Vector2i(floor((p.x + rad) / cell), floor((p.y + rad) / cell))
	for gx in range(c0.x - 1, c1.x + 2):
		for gz in range(c0.y - 1, c1.y + 2):
			for q in grid.get(Vector2i(gx, gz), []):
				if p.distance_to(Vector2(q[0], q[1])) < rad + float(q[2]):
					return false
	return true


func grid_put(grid: Dictionary, p: Vector2, rad: float, cell: float) -> void:
	var k = Vector2i(floor(p.x / cell), floor(p.y / cell))
	if not grid.has(k):
		grid[k] = []
	grid[k].append([p.x, p.y, rad])


# Частный сектор: дома вдоль улиц (фасадом к установке), кварталы и пустыри по шуму.
func build_lowrise() -> void:
	var any = false
	for sc in sectors():
		if str(sc.get("type", "")) != "lowrise":
			continue
		any = true
		load_house_models()
		var r0 = maxf(float(sc.get("from", 120.0)), float(cfg["world"].get("clear_radius", 35.0)))
		var r1 = float(sc.get("to", 800.0))
		var n = int(sc.get("houses", 350))
		var grid = {}
		var placed = 0
		var tries = 0
		while placed < n and tries < n * 20:
			tries += 1
			var sp = sector_point(sc, r0, r1)
			var p: Vector2 = sp[0]
			if noise.get_noise_2d(p.x * 0.6 + 900.0, p.y * 0.6) < -0.25:
				continue   # пустыри и огороды
			if not grid_free(grid, p, 8.0, 24.0):
				continue
			grid_put(grid, p, 8.0, 24.0)
			var rot = -float(sp[1]) * DEG + (PI * 0.5 if rng.randf() < 0.3 else 0.0) + rng.randf_range(-0.12, 0.12)
			place_house(p, rot)
			placed += 1
	if any:
		flush_houses()


const BUILDING_SHADER := """
shader_type spatial;
render_mode cull_back;
uniform float floor_h = 2.9;
uniform float win_w = 3.0;
varying vec3 lv;
varying vec3 ln;
varying vec3 sc;
varying vec2 seed;
varying vec3 tint;
float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void vertex() {
	sc = vec3(length(MODEL_MATRIX[0].xyz), length(MODEL_MATRIX[1].xyz), length(MODEL_MATRIX[2].xyz));
	lv = VERTEX * sc + vec3(0.0, sc.y * 0.5, 0.0);   // метры, y от земли
	ln = NORMAL;
	seed = MODEL_MATRIX[3].xz;
	tint = COLOR.rgb;
}
void fragment() {
	vec3 wall = tint;
	float glass = 0.0;
	if (abs(ln.y) < 0.5) {
		float u = (abs(ln.x) > 0.5 ? lv.z : lv.x) + (abs(ln.x) > 0.5 ? sc.z : sc.x) * 0.5;
		vec2 cell = vec2(u / win_w, lv.y / floor_h);
		vec2 f = fract(cell);
		vec2 id = floor(cell);
		float win = step(0.2, f.x) * step(f.x, 0.8) * step(0.3, f.y) * step(f.y, 0.8) * step(1.0, id.y);
		// издалека окна сливаются: среднее вместо мельтешения пикселей
		float aa = clamp(max(fwidth(cell.x), fwidth(cell.y)) * 1.5 - 0.4, 0.0, 1.0);
		win = mix(win, 0.24, aa);
		float r = h21(id + seed + ln.xz * 17.0);
		vec3 g = mix(vec3(0.08, 0.10, 0.13), vec3(0.42, 0.48, 0.55), r * r);   // стекло: тёмное или с отражением неба
		wall *= 1.0 - 0.10 * step(f.y, 0.05) * (1.0 - aa);                      // швы панелей
		ALBEDO = mix(wall, g, win);
		glass = win;
	} else {
		ALBEDO = wall * 0.5;
	}
	ROUGHNESS = mix(0.9, 0.25, glass);
	SPECULAR = mix(0.3, 0.7, glass);
}
"""

# Многоэтажки: микрорайоны из панельных «пластин» (9–17 эт.) и башен (16–25 эт.) с окнами,
# на крышах машинные будки и мачты (тонкие вертикали у неба — источник ложных целей).
func build_highrise() -> void:
	var slabs = []     # [Transform3D основания, ширина, глубина, высота, цвет]
	var masts = []     # [x, z, y0, h]
	for sc in sectors():
		if str(sc.get("type", "")) != "highrise":
			continue
		var r0 = maxf(float(sc.get("from", 250.0)), float(cfg["world"].get("clear_radius", 35.0)))
		var r1 = float(sc.get("to", 1500.0))
		var n = int(sc.get("buildings", 90))
		var fl = sc.get("floors", [9, 25])
		var grid = {}
		var tries = 0
		var placed = 0
		var colors = [Color(0.82, 0.80, 0.76), Color(0.74, 0.74, 0.72), Color(0.86, 0.78, 0.66), Color(0.68, 0.72, 0.78), Color(0.78, 0.66, 0.58), Color(0.90, 0.90, 0.88)]
		while placed < n and tries < n * 30:
			tries += 1
			var sp = sector_point(sc, r0, r1)
			var p: Vector2 = sp[0]
			var tower = rng.randf() < float(sc.get("towers", 0.4))
			var bw = rng.randf_range(18.0, 28.0) if tower else rng.randf_range(40.0, 100.0)
			var bd = bw * rng.randf_range(0.8, 1.0) if tower else rng.randf_range(12.0, 15.0)
			var floors = rng.randi_range(maxi(int(fl[0]), 16) if tower else int(fl[0]), int(fl[1]) if tower else mini(int(fl[1]), 17))
			var bh = floors * 2.9 + 1.0
			var rad = 0.5 * sqrt(bw * bw + bd * bd) + 12.0
			if not grid_free(grid, p, rad, 60.0):
				continue
			grid_put(grid, p, rad, 60.0)
			# кварталы: оси домов по сетке района (шум даёт поворот на куски района)
			var rot = -float(sp[1]) * DEG + (PI * 0.5 if rng.randf() < 0.45 else 0.0) + noise.get_noise_2d(p.x * 0.05, p.y * 0.05) * 0.6
			var xf = Transform3D(Basis(Vector3.UP, rot), Vector3(p.x, 0, p.y))
			var c: Color = colors[rng.randi() % colors.size()]
			slabs.append([xf, bw, bd, bh, c])
			add_box(xf, AABB(Vector3(-bw / 2, 0, -bd / 2), Vector3(bw, bh, bd)))
			# машинное отделение лифта на крыше
			var nb = 1 if tower else maxi(1, int(bw / 30.0))
			for k in nb:
				var ox = (k - (nb - 1) * 0.5) * (bw / nb)
				slabs.append([xf * Transform3D(Basis(), Vector3(ox, bh, 0)), 5.0, 6.0, 3.2, c * 0.85])
			if rng.randf() < float(sc.get("masts", 0.35)):
				var mp = xf * Vector3(rng.randf_range(-bw * 0.4, bw * 0.4), 0, rng.randf_range(-bd * 0.3, bd * 0.3))
				masts.append([mp.x, mp.z, bh, rng.randf_range(3.0, 9.0)])
			placed += 1
	if slabs.is_empty():
		return
	var m = ShaderMaterial.new()
	m.shader = make_shader(BUILDING_SHADER)
	var bm = BoxMesh.new()
	bm.material = m
	var mm = MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.mesh = bm
	mm.instance_count = slabs.size()
	for i in slabs.size():
		var it: Array = slabs[i]
		var xf: Transform3D = it[0]
		mm.set_instance_transform(i, xf * Transform3D(Basis().scaled(Vector3(it[1], it[3], it[2])), Vector3(0, it[3] * 0.5, 0)))
		mm.set_instance_color(i, it[4])
	var mmi = MultiMeshInstance3D.new()
	mmi.multimesh = mm
	world.add_child(mmi)
	if masts.is_empty():
		return
	var cm = CylinderMesh.new()
	cm.top_radius = 0.05
	cm.bottom_radius = 0.1
	cm.height = 1.0
	cm.radial_segments = 6
	cm.material = mat_color(Color(0.30, 0.30, 0.32), 0.7)
	var m2 = MultiMesh.new()
	m2.transform_format = MultiMesh.TRANSFORM_3D
	m2.mesh = cm
	m2.instance_count = masts.size()
	for i in masts.size():
		var it: Array = masts[i]
		m2.set_instance_transform(i, Transform3D(Basis().scaled(Vector3(1, it[3], 1)), Vector3(it[0], it[2] + it[3] * 0.5, it[1])))
	var mi2 = MultiMeshInstance3D.new()
	mi2.multimesh = m2
	world.add_child(mi2)


# Препятствие для разметки: коробка ab в осях xf (дома повёрнуты — AABB в мировых осях раздувался бы).
func add_box(xf: Transform3D, ab: AABB) -> void:
	boxes.append([xf.affine_inverse(), ab, xf])


# Корзины по азимуту, как у деревьев: проверка заслонения смотрит только коробки в направлении цели.
func bucket_boxes() -> void:
	box_buckets.resize(360)
	for i in 360:
		box_buckets[i] = PackedInt32Array()
	for idx in boxes.size():
		var ab: AABB = boxes[idx][1]
		var xf: Transform3D = boxes[idx][2]
		var lo = 1e9
		var hi = -1e9
		var ref = 0.0
		var inside = false
		for k in 8:
			var q = xf * ab.get_endpoint(k)
			var d = Vector2(q.x - rig_pos.x, q.z - rig_pos.z)
			if d.length() < 1.0:
				inside = true
			var az = atan2(d.x, -d.y) / DEG
			if k == 0:
				ref = az
			var rel = wrap180(az - ref)
			lo = minf(lo, rel)
			hi = maxf(hi, rel)
		var full = inside or hi - lo > 179.0 or ab.has_point(xf.affine_inverse() * Vector3(rig_pos.x, ab.position.y + 0.01, rig_pos.z))
		var a_lo = int(floor(ref + lo - 0.6))
		var a_hi = int(floor(ref + hi + 0.6))
		if full:
			a_lo = 0
			a_hi = 359
		for a in range(a_lo, a_hi + 1):
			var arr: PackedInt32Array = box_buckets[posmod(a, 360)]
			arr.append(idx)
			box_buckets[posmod(a, 360)] = arr


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
		add_box(Transform3D(), AABB(Vector3(pos.x - r, 0, pos.z - r), Vector3(2 * r, h, 2 * r)))


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
var rt_rng := RandomNumberGenerator.new()   # случайные цели (и новые вместо улетевших в --live)
var rt_ctx := {}


func random_targets() -> Array:
	var rc: Dictionary = cfg.get("random_targets", {})
	var out = []
	if rc.is_empty():
		return out
	rt_rng.seed = int(rc.get("seed", int(cfg.get("seed", 1)) + 100))
	var types: Dictionary = rc.get("types", {"quad": 0.5, "delta": 0.25, "plane": 0.25})
	var tot = 0.0
	for k in types:
		tot += float(types[k])
	rt_ctx = {"types": types, "tot": tot, "range": rc.get("range", [150, 1500]),
		"half": float(wcfg.get("hfov", 120.0)) * 0.4, "az0": float(wcfg.get("az", 0.0)),
		"around": bool(rc.get("around", false)) or view_mode,   # по всему кругу, а не только в кадре
		"secs": LIVE_PATH_S if live_mode else float(cfg.get("seconds", 10.0))}
	var mult = 3 if view_mode and not bool(rc.get("around", false)) else 1
	var n_dr = int(rc.get("drones", 6)) * mult
	for i in n_dr + int(rc.get("birds", 3)) * mult:
		out.append(gen_random(i < n_dr, 0.0))
	return out


# Одна случайная цель: тип по долям, прямая через точку в поле зрения широкой (середина пути — в t0 + secs/2).
func gen_random(drone: bool, t0: float) -> Dictionary:
	var r = rt_rng
	var types: Dictionary = rt_ctx["types"]
	var rng_r = rt_ctx["range"]
	var t = {}
	var model = "bird"
	var size = r.randf_range(0.3, 1.2)
	var speed = r.randf_range(8, 14)
	if drone:
		var x = r.randf() * float(rt_ctx["tot"])
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
	var half = float(rt_ctx["half"])
	var az = (float(rt_ctx["az0"]) + r.randf_range(-half, half)) * DEG
	if rt_ctx["around"]:
		az = r.randf() * TAU
	var el = r.randf_range(3, 25) * DEG
	var c = Vector3(sin(az) * cos(el), cos(az) * cos(el), sin(el)) * dist   # восток, север, верх
	var hd = r.randf() * TAU
	var dirv = Vector3(cos(hd), sin(hd), r.randf_range(-0.05, 0.05))
	var secs = float(rt_ctx["secs"])
	var a = c - dirv * speed * secs * 0.5
	var b = c + dirv * speed * secs * 0.5
	t["path"] = {"type": "line", "from": [a.x, a.y, maxf(a.z, 15.0)], "to": [b.x, b.y, maxf(b.z, 15.0)]}
	if live_mode:
		t["start"] = t0
		t["end"] = t0 + secs
		t["random"] = true
	return t


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
	for i in box_buckets[az]:
		var inv: Transform3D = boxes[i][0]
		var ab: AABB = boxes[i][1]
		if ab.intersects_segment(inv * c, inv * p) != null:
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
	var wp = Vector2(view_az, view_el) if view_mode else wide_pose(t)
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


# ---------- панорама и окно просмотра ----------

# 12 снимков по 30° (hfov 30, угол места 10°) — sim/panorama.py склеивает их в полосу 360°.
func run_panorama() -> void:
	var n = 12
	wide_vp.size = Vector2i(480, 480)
	wide_cam.fov = 360.0 / n
	set_time(0.0)
	var info = []
	for i in n:
		var az = (i + 0.5) * 360.0 / n
		aim(wide_cam, az, 10.0)
		wide_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		await RenderingServer.frame_post_draw
		if i == 0:   # первый кадр после смены размера — ещё с прогревом шейдеров
			wide_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
			await RenderingServer.frame_post_draw
		var img = wide_vp.get_texture().get_image()
		img.convert(Image.FORMAT_RGB8)
		img.save_png("%s/pano_%02d.png" % [out_dir, i])
		var sc = sector_at(az)
		info.append({"az": az, "sector": str(sc.get("name", sc.get("type", "")))})
	save_json(out_dir + "/pano.json", {"name": str(cfg.get("name", "sim")), "hfov": 360.0 / n, "el": 10.0, "shots": info,
		"sectors": sectors()})
	print("панорама: %d снимков" % n)


func start_view() -> void:
	view_az = float(wcfg.get("az", 0.0))
	view_el = float(wcfg.get("el", 15.0))
	view_pan = float(wcfg.get("pan_dps", 0.0)) != 0.0
	wide_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr = TextureRect.new()
	tr.texture = wide_vp.get_texture()
	tr.set_anchors_preset(Control.PRESET_FULL_RECT)
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	add_child(tr)
	if fine_enabled():
		fine_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		var ft = TextureRect.new()
		ft.texture = fine_vp.get_texture()
		ft.stretch_mode = TextureRect.STRETCH_SCALE
		ft.size = Vector2(320, 240)
		ft.position = Vector2(get_window().size.x - 330, get_window().size.y - 250)
		add_child(ft)
	view_hud = Label.new()
	view_hud.position = Vector2(10, 8)
	view_hud.add_theme_color_override("font_color", Color(1, 1, 0.4))
	view_hud.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	view_hud.add_theme_constant_override("outline_size", 4)
	add_child(view_hud)
	set_time(0.0)


func _process(delta: float) -> void:
	if not view_mode or view_hud == null:
		return
	var k = 4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0
	var rate = 30.0 * k * wide_cam.fov / 120.0
	if Input.is_key_pressed(KEY_LEFT) or Input.is_key_pressed(KEY_A):
		view_az -= rate * delta
	if Input.is_key_pressed(KEY_RIGHT) or Input.is_key_pressed(KEY_D):
		view_az += rate * delta
	if Input.is_key_pressed(KEY_UP) or Input.is_key_pressed(KEY_W):
		view_el = minf(view_el + rate * delta, 89.0)
	if Input.is_key_pressed(KEY_DOWN) or Input.is_key_pressed(KEY_S):
		view_el = maxf(view_el - rate * delta, -10.0)
	if view_pan:
		var pd = float(wcfg.get("pan_dps", 0.0))
		view_az += (pd if pd != 0.0 else 10.0) * delta
	view_az = fposmod(view_az, 360.0)
	var secs = float(cfg.get("seconds", 10.0))
	view_t += delta * view_speed
	if view_t > secs:
		view_t = 0.0
		f_t = 0.0
	set_time(view_t)
	var sc = sector_at(view_az)
	var nm = str(sc.get("name", sc.get("type", "—")))
	var tid = targets[f_target]["id"] if f_target >= 0 else "—"
	view_hud.text = "az %.1f°  el %.1f°  fov %.0f°  сектор: %s  t %.1f/%.0f с%s  точная → цель %s\n" % [
		view_az, view_el, wide_cam.fov, nm, view_t, secs, " (пауза)" if view_speed == 0.0 else "", str(tid)] + \
		"←→↑↓/WASD поворот (Shift быстрее)  1-9 сектор  Tab следующий  колесо/+- зум  M обзор по кругу  Пробел пауза  R сначала  F12 снимок  Esc выход"


func _unhandled_input(ev: InputEvent) -> void:
	if not view_mode:
		return
	if ev is InputEventMouseButton and ev.pressed:
		if ev.button_index == MOUSE_BUTTON_WHEEL_UP:
			wide_cam.fov = maxf(wide_cam.fov / 1.15, 2.0)
		elif ev.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			wide_cam.fov = minf(wide_cam.fov * 1.15, 150.0)
		return
	if not (ev is InputEventKey and ev.pressed and not ev.echo):
		return
	var key = ev.keycode
	var sl = sectors()
	if key >= KEY_1 and key <= KEY_9 and key - KEY_1 < sl.size():
		view_az = sector_mid(sl[key - KEY_1])
	elif key == KEY_TAB and sl.size() > 0:
		var cur = -1
		for i in sl.size():
			if in_sector(sl[i], view_az):
				cur = i
		view_az = sector_mid(sl[(cur + 1) % sl.size()])
	elif key == KEY_EQUAL or key == KEY_KP_ADD:
		wide_cam.fov = maxf(wide_cam.fov / 1.25, 2.0)
	elif key == KEY_MINUS or key == KEY_KP_SUBTRACT:
		wide_cam.fov = minf(wide_cam.fov * 1.25, 150.0)
	elif key == KEY_M:
		view_pan = not view_pan
	elif key == KEY_SPACE:
		view_speed = 1.0 if view_speed == 0.0 else 0.0
	elif key == KEY_R:
		view_t = 0.0
		f_t = 0.0
	elif key == KEY_F12:
		var dir = out_dir if out_dir != "" else "."
		DirAccess.make_dir_recursive_absolute(dir)
		var fn = "%s/view_%s_az%03d.png" % [dir, str(cfg.get("name", "sim")), int(view_az)]
		wide_vp.get_texture().get_image().save_png(fn)
		print("снимок: ", fn)
	elif key == KEY_ESCAPE:
		get_tree().quit()


# ---------- источник кадров эмулятора (--live) ----------

# Подключиться к приложению и принять hello: камеры и случайные цели задаёт эмулятор.
func live_connect(port: int) -> bool:
	live_peer = StreamPeerTCP.new()
	if live_peer.connect_to_host("127.0.0.1", port) != OK:
		push_error("нет связи с приложением, порт %d" % port)
		return false
	var t0 = Time.get_ticks_msec()
	while live_peer.get_status() == StreamPeerTCP.STATUS_CONNECTING and Time.get_ticks_msec() - t0 < 10000:
		live_peer.poll()
		OS.delay_msec(5)
	if live_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		push_error("нет связи с приложением, порт %d" % port)
		return false
	live_peer.set_no_delay(true)
	var line = live_read_line()
	var h = JSON.parse_string(line if line != null else "")
	if typeof(h) != TYPE_DICTIONARY:
		push_error("приложение не прислало hello")
		return false
	live_mode = true
	var wv: Array = h.get("wide", [4608, 3456, 120, 14])
	var fv: Array = h.get("fine", [640, 480, 20, 60])
	wcfg["w"] = int(wv[0])
	wcfg["h"] = int(wv[1])
	wcfg["hfov"] = float(wv[2])
	wcfg["fps"] = float(wv[3])
	wcfg["az"] = float(h.get("az", wcfg.get("az", 0.0)))
	wcfg["el"] = float(h.get("el", wcfg.get("el", 15.0)))
	wcfg["pan_dps"] = 0.0
	fcfg["enabled"] = true
	fcfg["w"] = int(fv[0])
	fcfg["h"] = int(fv[1])
	fcfg["hfov"] = float(fv[2])
	fcfg["fps"] = float(fv[3])
	cfg["targets"] = []   # мир сцены — по её seed, цели — по номеру сцены эмулятора
	cfg["random_targets"] = {"drones": int(h.get("drones", 3)), "birds": int(h.get("birds", 1)),
		"seed": int(h.get("seed", 1)) + 100,
		"range": h.get("range", [250, 1200]), "types": h.get("types", {"quad": 0.5, "delta": 0.25, "plane": 0.25})}
	get_window().title = "Дронолок: 3D-сцена для эмулятора (не закрывать)"
	printerr("live: связь с приложением, порт %d" % port)
	return true


func live_read_line():
	while true:
		var i = live_buf.find(10)
		if i >= 0:
			var s = live_buf.slice(0, i).get_string_from_utf8()
			live_buf = live_buf.slice(i + 1)
			return s
		live_peer.poll()
		if live_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return null
		var n = live_peer.get_available_bytes()
		if n > 0:
			var r = live_peer.get_data(n)
			if r[0] == OK:
				live_buf.append_array(r[1])
		else:
			OS.delay_usec(200)
	return null


# Цель улетела из кадра широкой или её путь кончился — вместо неё новая (новый номер), как в эмуляторе.
func live_respawn(t: float) -> void:
	var w = wide_vp.size.x
	var h = wide_vp.size.y
	for tg in targets:
		var c: Dictionary = tg["cfg"]
		if not c.get("random", false):
			continue
		var p = target_pos(tg, t)
		var gone = p == null or p.y < 5.0
		if not gone:
			if wide_cam.is_position_behind(p):
				gone = true
			else:
				var px = wide_cam.unproject_position(p)
				gone = px.x < -0.05 * w or px.x > 1.05 * w or px.y < -0.05 * h or px.y > 1.05 * h
		if not gone:
			continue
		rt_ctx["az0"] = azel_of(-wide_cam.global_transform.basis.z).x
		var n = gen_random(tg["kind"] == "drone", t)
		n["model"] = c.get("model", n["model"])   # модель у узла та же: размер и форма прежние
		n["size"] = c.get("size", n["size"])
		tg["cfg"] = n
		tg["id"] = live_next_id
		live_next_id += 1


func live_loop() -> void:
	# прогрев шейдеров
	set_time(0.0)
	for _w in 2:
		wide_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		fine_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		await RenderingServer.frame_post_draw
	var shapes = {"bird": 0, "quad": 1, "delta": 2, "wing": 3, "plane": 3}
	printerr("live: сцена готова, целей %d" % targets.size())
	var served = 0
	while true:
		var line = live_read_line()
		if line == null:
			break
		var q = JSON.parse_string(line)
		if typeof(q) != TYPE_DICTIONARY:
			continue
		if q.has("quit"):
			break
		var t = float(q.get("t", 0.0))
		var cam = int(q.get("cam", 0))
		for m in time_mats:
			(m as ShaderMaterial).set_shader_parameter("sim_time", t)
		var vp: SubViewport = wide_vp if cam == 0 else fine_vp
		aim(wide_cam if cam == 0 else fine_cam, float(q.get("az", 0.0)), float(q.get("el", 0.0)))
		if cam == 0:
			live_respawn(t)
		update_targets(t)
		vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		await RenderingServer.frame_post_draw
		var img = vp.get_texture().get_image()
		img.convert(Image.FORMAT_L8)
		var data = img.get_data()
		var rows = []
		for tg in targets:
			var p = target_pos(tg, t)
			if p == null:
				continue
			var p2 = target_pos(tg, t + 0.05)
			var v = (p2 - p) / 0.05 if p2 != null else Vector3.ZERO
			var c: Dictionary = tg["cfg"]
			rows.append("%d %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d" % [tg["id"], 1 if tg["kind"] == "bird" else 0,
				shapes.get(str(c.get("model", "quad")), 1), float(tg["size"]), p.x, -p.z, p.y, v.x, -v.z, v.y,
				1 if occluded(rig_pos, p) else 0])
		var head = "FRAME %d %d %d %d\n" % [img.get_width(), img.get_height(), data.size(), rows.size()]
		if rows.size() > 0:
			head += "\n".join(rows) + "\n"
		if live_peer.put_data(head.to_utf8_buffer()) != OK or live_peer.put_data(data) != OK:
			printerr("live: связь оборвалась")
			break
		served += 1
		if served <= 2 or served % 500 == 0:
			printerr("live: кадров %d (t %.2f)" % [served, t])
	printerr("live: конец, кадров %d" % served)


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
		"rig_height": rig_pos.y, "targets": tracks,
		"sector": str(sector_at(float(wcfg.get("az", 0.0))).get("type", "")), "sectors": sectors()})
