"""
Broadcast-style renderer for a recorded lap (see record_lap.py).

Layout, in a 1920x1080 design space that scales to any output size:

    +------------------------------+----------------+
    | chase cam: track, kerbs,     | circuit name   |
    | speed-coloured racing line,  | track map      |
    | car; lap timer, speed gauge, | speed trace    |
    | throttle/brake, G-G diagram  | throttle/brake |
    +------------------------------+----------------+

Every frame is drawn `supersample` times larger and shrunk with smoothscale,
which anti-aliases all the polygons and lines pygame draws without it.

    renderer = LapRenderer(LapData("suzuka_lap.npz"), width=1920, height=1080)
    frame = renderer.render(t)          # pygame.Surface at (width, height)
"""
import math
import os

import numpy as np
import pygame

from f1_car import F1Car

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
DESIGN_W, DESIGN_H = 1920, 1080
CHASE_W = 1240  # design-space width of the chase-cam panel; the rest is the data panel

# ---- palette -----------------------------------------------------------------
GRASS = (22, 36, 27)
RUNOFF = (54, 55, 58)
ASPHALT = (38, 40, 46)
EDGE_LINE = (226, 228, 232)
KERB_RED = (206, 38, 46)
KERB_WHITE = (232, 232, 232)
PANEL_BG = (12, 14, 19)
PANEL_LINE = (38, 42, 52)
TEXT = (236, 238, 242)
TEXT_DIM = (132, 138, 152)
ACCENT = (255, 62, 84)
THROTTLE_GREEN = (64, 220, 128)
BRAKE_RED = (255, 72, 72)

# Speed colour scale (km/h -> RGB): slow corners are blue/violet, flat out is yellow
SPEED_STOPS = [(60, (80, 130, 255)), (140, (150, 95, 255)), (210, (240, 80, 170)),
               (270, (255, 140, 50)), (320, (255, 230, 70))]

SPLIT_LABELS = {"test": "NEVER SEEN IN TRAINING", "val": "VALIDATION TRACK", "train": "TRAINING TRACK"}


def speed_color(kph):
    """Map speeds (scalar or array, km/h) onto SPEED_STOPS. Returns an (..., 3) uint8 array."""
    kph = np.asarray(kph, dtype=float)
    xs = [s for s, _ in SPEED_STOPS]
    rgb = [np.interp(kph, xs, [c[i] for _, c in SPEED_STOPS]) for i in range(3)]
    return np.stack(rgb, axis=-1).astype(np.uint8)


def format_laptime(seconds):
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{secs:06.3f}"


def gaussian_smooth(values, sigma, wrap=False):
    """Smooth a 1-D array with a Gaussian kernel of `sigma` samples (edges padded, or wrapped)."""
    if sigma <= 0:
        return np.asarray(values, dtype=float)
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    mode = "wrap" if wrap else "edge"
    return np.convolve(np.pad(values, radius, mode=mode), k, mode="valid")


class LapData:
    """A recorded lap plus derived channels, with interpolation at any time t."""

    def __init__(self, path):
        d = np.load(path)
        get = lambda key, default=None: d[key] if key in d.files else default

        self.x, self.y, self.v = d["x"], d["y"], d["v"]
        self.theta = np.unwrap(d["theta"])
        self.throttle, self.steering, self.arc = d["throttle"], d["steering"], d["arc"]
        self.dt = float(d["dt"])
        self.n = len(self.x)
        self.duration = (self.n - 1) * self.dt
        self.lap_time = float(get("lap_time", self.duration))
        self.track_length = float(d["track_length"])

        self.centerline = np.asarray(d["centerline"], dtype=float)
        left, right = d["left_boundary"], d["right_boundary"]
        width = get("track_width")
        # Files from the old recorder have no width: recover it from the boundaries
        self.track_width = float(width) if width is not None else float(
            np.median(np.linalg.norm(left[:len(self.centerline)] - right[:len(self.centerline)], axis=1)))
        curvature = get("curvature")
        if curvature is None:
            from track_preprocessing import compute_curvature
            curvature = compute_curvature(self.centerline, signed=True)
        self.curvature = np.asarray(curvature, dtype=float)

        self.track_id = str(get("track_id", "track"))
        self.track_name = str(get("track_name", self.track_id))
        self.track_split = str(get("track_split", ""))
        self.model_name = str(get("model_name", ""))

        # Derived channels
        self.kph = self.v * 3.6
        self.progress = np.clip(self.arc / self.track_length, 0.0, 1.0)
        yaw_rate = np.gradient(self.theta, self.dt)
        self.g_lat = gaussian_smooth(self.v * yaw_rate, 1.5) / 9.81
        self.g_long = gaussian_smooth(np.gradient(self.v, self.dt), 1.5) / 9.81
        self.color = speed_color(self.kph)

    def index(self, t):
        """(i0, alpha): the sample before time t and the fraction of the way to the next."""
        f = float(np.clip(t / self.dt, 0.0, self.n - 1))
        i0 = min(int(f), self.n - 2)
        return i0, f - i0

    def lerp(self, arr, t):
        i0, a = self.index(t)
        return arr[i0] + a * (arr[i0 + 1] - arr[i0])

    def state(self, t):
        return {k: self.lerp(getattr(self, k), t)
                for k in ("x", "y", "theta", "v", "kph", "throttle", "steering", "arc", "progress",
                          "g_lat", "g_long")}


class Fonts:
    """Text rendering with Titillium Web from assets/fonts, via Pillow.

    pygame's own font modules fail to import on Python 3.14 (pygame 2.6), so
    text is rasterised with PIL and converted to pygame surfaces, with a cache
    because most labels repeat every frame.
    """

    FILES = {"regular": "TitilliumWeb-Regular.ttf", "semibold": "TitilliumWeb-SemiBold.ttf",
             "bold": "TitilliumWeb-Bold.ttf"}

    def __init__(self, scale):
        self.scale = scale
        self._fonts = {}
        self._cache = {}

    def _font(self, weight, size):
        from PIL import ImageFont
        key = (weight, size)
        if key not in self._fonts:
            px = max(6, int(round(size * self.scale)))
            path = os.path.join(FONT_DIR, self.FILES[weight])
            try:
                self._fonts[key] = ImageFont.truetype(path, px)
            except OSError:
                self._fonts[key] = ImageFont.load_default(px)
        return self._fonts[key]

    def render(self, msg, weight, size, color):
        key = (msg, weight, size, tuple(color))
        img = self._cache.get(key)
        if img is None:
            from PIL import Image, ImageDraw
            font = self._font(weight, size)
            ascent, descent = font.getmetrics()
            width = max(1, int(math.ceil(font.getlength(msg))))
            canvas = Image.new("RGBA", (width + 2, ascent + descent), (0, 0, 0, 0))
            ImageDraw.Draw(canvas).text((1, 0), msg, font=font, fill=(*color[:3], 255))
            img = pygame.image.frombytes(canvas.tobytes(), canvas.size, "RGBA")
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = img
        return img


class LapRenderer:
    def __init__(self, lap, width=1920, height=1080, supersample=2):
        if not pygame.get_init():
            pygame.init()
        self.lap = lap
        self.out_size = (width, height)
        self.ss = supersample
        self.s = supersample * height / DESIGN_H          # design units -> canvas pixels
        self.canvas = pygame.Surface((width * supersample, height * supersample))
        self.W, self.H = self.canvas.get_size()
        self.fonts = Fonts(self.s)
        self.chase_w = round(self.W * CHASE_W / DESIGN_W)

        self._prepare_track()
        self._prepare_camera()
        self.car = F1Car(ppm=8.0 * self.s, visual_scale=2.0, supersample=3)
        self._vignette = self._make_vignette()
        self._map = self._prepare_map()
        self._traces = self._prepare_traces()

    # ------------------------------------------------------------- helpers
    def u(self, v):
        """Design units -> canvas pixels."""
        return v * self.s

    def text(self, surf, msg, pos, weight="regular", size=24, color=TEXT, anchor="topleft"):
        img = self.fonts.render(msg, weight, size, color)
        rect = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        surf.blit(img, rect)
        return rect

    def panel(self, surf, rect, color=(8, 10, 14, 190), radius=14):
        r = pygame.Rect([round(c) for c in rect])
        box = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(box, color, box.get_rect(), border_radius=round(self.u(radius)))
        surf.blit(box, r.topleft)
        return r

    # ------------------------------------------------------- track geometry
    def _prepare_track(self):
        c = self.lap.centerline
        nxt = np.roll(c, -1, axis=0)
        prv = np.roll(c, 1, axis=0)
        tangent = nxt - prv
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # points left
        half = self.lap.track_width / 2
        self.edge_l, self.edge_r = c + normal * half, c - normal * half
        self.run_l, self.run_r = c + normal * (half + 9.0), c - normal * (half + 9.0)
        self.kerb_l, self.kerb_r = c + normal * (half + 1.4), c - normal * (half + 1.4)

        # Kerbs where the corner radius is under ~70 m, extended a little either side
        seg = np.linalg.norm(nxt - c, axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        curv = gaussian_smooth(np.abs(self.lap.curvature), 3, wrap=True)
        corner = curv > 1 / 70
        grow = np.convolve(np.pad(corner.astype(float), 6, mode="wrap"), np.ones(13), mode="valid") > 0
        self.kerb_on = grow
        self.kerb_color = np.where((arc // 3.5).astype(int) % 2 == 0, 0, 1)
        self.n_pts = len(c)
        self.start_pt, self.start_normal, self.start_tangent = c[0], normal[0], tangent[0]

    def _prepare_camera(self):
        lap = self.lap
        self.cam_heading = gaussian_smooth(lap.theta, 0.35 / lap.dt)
        v_smooth = gaussian_smooth(lap.v, 1.2 / lap.dt)
        self.cam_ppm = np.interp(v_smooth, [15, 90], [10.5, 6.6])  # zoom out at speed

    def _xf(self, pts, cam, heading, ppm, origin):
        """World points (N, 2) -> canvas points, with the car's heading pointing up."""
        rot = math.pi / 2 - heading
        c, s = math.cos(rot), math.sin(rot)
        rel = pts - cam
        rx = rel[..., 0] * c - rel[..., 1] * s
        ry = rel[..., 0] * s + rel[..., 1] * c
        return np.stack([origin[0] + ppm * rx, origin[1] - ppm * ry], axis=-1)

    # ---------------------------------------------------------------- chase
    def draw_chase(self, surf, t):
        lap = self.lap
        st = lap.state(t)
        heading = lap.lerp(self.cam_heading, t)
        ppm = lap.lerp(self.cam_ppm, t) * self.s
        cam = np.array([st["x"], st["y"]])
        origin = (self.chase_w / 2, self.H * 0.64)

        clip = pygame.Rect(0, 0, self.chase_w, self.H)
        surf.set_clip(clip)
        surf.fill(GRASS, clip)

        # Segments near the camera (both ends checked so long straights aren't culled)
        radius_m = math.hypot(self.chase_w, self.H) / ppm * 0.75 + 20
        near = np.linalg.norm(lap.centerline - cam, axis=1) < radius_m
        seg_on = near | np.roll(near, -1)
        idx = np.nonzero(seg_on)[0]
        j = (idx + 1) % self.n_pts

        xf = lambda p: self._xf(p, cam, heading, ppm, origin)
        run_l, run_r = xf(self.run_l), xf(self.run_r)
        kerb_l, kerb_r = xf(self.kerb_l), xf(self.kerb_r)
        edge_l, edge_r = xf(self.edge_l), xf(self.edge_r)

        for i, k in zip(idx, j):
            pygame.draw.polygon(surf, RUNOFF, (run_l[i], run_l[k], run_r[k], run_r[i]))
        for i, k in zip(idx, j):
            if self.kerb_on[i]:
                col = KERB_RED if self.kerb_color[i] == 0 else KERB_WHITE
                pygame.draw.polygon(surf, col, (edge_l[i], edge_l[k], kerb_l[k], kerb_l[i]))
                pygame.draw.polygon(surf, col, (edge_r[i], edge_r[k], kerb_r[k], kerb_r[i]))
        for i, k in zip(idx, j):
            pygame.draw.polygon(surf, ASPHALT, (edge_l[i], edge_l[k], edge_r[k], edge_r[i]))
        line_w = max(1, round(0.3 * ppm))
        for i, k in zip(idx, j):
            pygame.draw.line(surf, EDGE_LINE, edge_l[i], edge_l[k], line_w)
            pygame.draw.line(surf, EDGE_LINE, edge_r[i], edge_r[k], line_w)

        self._draw_start_line(surf, xf)
        self._draw_racing_line(surf, t, xf, ppm, cam, radius_m)

        car_pos = xf(np.array([[st["x"], st["y"]]]))[0]
        car_theta = st["theta"] - heading + math.pi / 2
        braking = st["throttle"] < -0.05
        self.car.draw(surf, car_pos, car_theta, st["steering"], braking=braking,
                      shadow_offset=(self.u(5), self.u(7)))

        surf.blit(self._vignette, (0, 0))
        surf.set_clip(None)

    def _draw_start_line(self, surf, xf):
        """Chequered start/finish line: two rows of squares across the track."""
        size = 1.2
        n = max(2, int(self.lap.track_width / size))
        for row in range(2):
            for col in range(n):
                if (row + col) % 2:
                    continue
                a = (self.start_pt + self.start_normal * (self.lap.track_width / 2 - col * size)
                     + self.start_tangent * (row * size))
                quad = np.array([a, a - self.start_normal * size,
                                 a - self.start_normal * size + self.start_tangent * size,
                                 a + self.start_tangent * size])
                pygame.draw.polygon(surf, (240, 240, 240), xf(quad))

    def _draw_racing_line(self, surf, t, xf, ppm, cam, radius_m):
        lap = self.lap
        i0, a = lap.index(t)
        pos = np.column_stack([lap.x, lap.y])
        head = pos[i0] + a * (pos[i0 + 1] - pos[i0])

        # Line still to come: faint dots, so the viewer sees the agent's chosen line
        ahead = pos[i0 + 1: i0 + 1 + int(6.0 / lap.dt)]
        if len(ahead):
            pts = xf(ahead)
            r = max(1, round(0.28 * ppm))
            for p in pts[::2]:
                pygame.draw.circle(surf, (150, 154, 168), p, r)

        # Line driven so far: thick, coloured by speed
        start = max(0, i0 - int(25.0 / lap.dt))
        trail = np.vstack([pos[start:i0 + 1], head])
        colors = lap.color[start:i0 + 2]
        keep = np.linalg.norm(trail - cam, axis=1) < radius_m
        pts = xf(trail)
        w = max(2, round(0.95 * ppm))
        for q in range(len(pts) - 1):
            if keep[q] or keep[q + 1]:
                pygame.draw.line(surf, colors[q + 1], pts[q], pts[q + 1], w)
                pygame.draw.circle(surf, colors[q + 1], pts[q + 1], w / 2)

    def _make_vignette(self):
        """Darkened edges and corners for the chase panel (a radial alpha gradient)."""
        w, h = self.chase_w, self.H
        small = pygame.Surface((64, 36), pygame.SRCALPHA)
        for y in range(36):
            for x in range(64):
                d = math.hypot((x - 31.5) / 32, (y - 20) / 22)
                small.set_at((x, y), (0, 0, 0, int(np.clip((d - 0.55) * 210, 0, 170))))
        return pygame.transform.smoothscale(small, (w, h))

    # ------------------------------------------------------------ chase HUD
    def draw_hud(self, surf, t):
        st = self.lap.state(t)
        u = self.u

        # Lap timer and progress
        box = self.panel(surf, (u(28), u(28), u(330), u(118)))
        self.text(surf, "LAP TIME", (box.x + u(22), box.y + u(14)), "semibold", 18, TEXT_DIM)
        shown = min(t, self.lap.lap_time)
        self.text(surf, format_laptime(shown), (box.x + u(20), box.y + u(30)), "bold", 50)
        bar = pygame.Rect(box.x + u(22), box.bottom - u(18), box.w - u(44), max(2, u(5)))
        pygame.draw.rect(surf, PANEL_LINE, bar, border_radius=bar.h)
        filled = bar.copy()
        filled.w = max(1, round(bar.w * st["progress"]))
        pygame.draw.rect(surf, ACCENT, filled, border_radius=bar.h)

        self._draw_speed_gauge(surf, st)
        self._draw_gg(surf, t)
        self._draw_speed_legend(surf)
        self._draw_title_card(surf, t)
        if t >= self.lap.duration:
            self._draw_finish_banner(surf)

    def _draw_speed_gauge(self, surf, st):
        u = self.u
        cx, cy, r = u(170), self.H - u(150), u(118)
        self.panel(surf, (cx - u(150), cy - u(142), u(410), u(270)), radius=18)
        # 250-degree dial, 0 km/h at bottom-left, filling clockwise
        start, span = math.radians(215), math.radians(250)
        frac = float(np.clip(st["kph"] / 350, 0, 1))
        width = u(14)

        def ring(a0, a1, color):
            angles = np.linspace(a0, a1, max(2, int(abs(a1 - a0) / 0.03) + 2))
            outer = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in angles]
            inner = [(cx + (r - width) * math.cos(a), cy - (r - width) * math.sin(a)) for a in angles[::-1]]
            pygame.draw.polygon(surf, color, outer + inner)

        ring(start, start - span, PANEL_LINE)
        steps = max(1, int(50 * frac))
        for q in range(steps):
            a0 = start - span * frac * q / steps
            a1 = start - span * frac * (q + 1) / steps - 0.005
            ring(a0, a1, tuple(speed_color(350 * frac * (q + 0.5) / steps)))
        self.text(surf, f"{st['kph']:.0f}", (cx, cy + u(4)), "bold", 78, anchor="center")
        self.text(surf, "KM/H", (cx, cy + u(52)), "semibold", 20, TEXT_DIM, anchor="center")

        # Throttle and brake bars
        for q, (label, value, color) in enumerate([("THR", max(st["throttle"], 0.0), THROTTLE_GREEN),
                                                   ("BRK", max(-st["throttle"], 0.0), BRAKE_RED)]):
            x = cx + u(162) + q * u(46)
            bar = pygame.Rect(round(x), round(cy - u(100)), round(u(24)), round(u(190)))
            pygame.draw.rect(surf, PANEL_LINE, bar, border_radius=round(u(5)))
            fill = bar.copy()
            fill.h = round(bar.h * min(value, 1.0))
            fill.bottom = bar.bottom
            if fill.h > 0:
                pygame.draw.rect(surf, color, fill, border_radius=round(u(5)))
            self.text(surf, label, (bar.centerx, bar.bottom + u(16)), "semibold", 16, TEXT_DIM, anchor="center")

    def _draw_gg(self, surf, t):
        """G-G diagram: lateral vs longitudinal g, with the last second as a fading trail."""
        lap, u = self.lap, self.u
        r = u(92)
        cx, cy = self.chase_w - u(140), self.H - u(150)
        self.panel(surf, (cx - u(118), cy - u(132), u(236), u(262)), radius=18)
        for g in (2, 4, 6):
            pygame.draw.circle(surf, PANEL_LINE, (cx, cy), r * g / 6.5, max(1, round(u(1.5))))
        pygame.draw.line(surf, PANEL_LINE, (cx - r, cy), (cx + r, cy), max(1, round(u(1))))
        pygame.draw.line(surf, PANEL_LINE, (cx, cy - r), (cx, cy + r), max(1, round(u(1))))

        def pt(g_lat, g_long):
            # Right turn pushes the driver left, so plot lateral g as felt (negated yaw direction)
            return cx - r * g_lat / 6.5, cy - r * g_long / 6.5

        i0, _ = lap.index(t)
        span = int(1.0 / lap.dt)
        for q in range(max(0, i0 - span), i0 + 1):
            fade = 1 - (i0 - q) / span
            col = tuple(int(c * (0.25 + 0.6 * fade)) for c in ACCENT)
            pygame.draw.circle(surf, col, pt(lap.g_lat[q], lap.g_long[q]), u(3.5))
        g_lat, g_long = lap.lerp(lap.g_lat, t), lap.lerp(lap.g_long, t)
        pygame.draw.circle(surf, (255, 255, 255), pt(g_lat, g_long), u(7))
        total = math.hypot(g_lat, g_long)
        self.text(surf, "G-FORCE", (cx - u(100), cy - u(122)), "semibold", 16, TEXT_DIM)
        self.text(surf, f"{total:.1f} G", (cx + u(100), cy - u(126)), "bold", 22, anchor="topright")

    def _draw_speed_legend(self, surf):
        u = self.u
        x, y, w, h = self.chase_w - u(300), u(40), u(250), u(10)
        self.panel(surf, (x - u(18), y - u(14), w + u(36), u(70)), radius=12)
        for q in range(round(w)):
            kph = SPEED_STOPS[0][0] + (SPEED_STOPS[-1][0] - SPEED_STOPS[0][0]) * q / w
            pygame.draw.line(surf, tuple(speed_color(kph)), (x + q, y), (x + q, y + h))
        self.text(surf, f"{SPEED_STOPS[0][0]}", (x, y + h + u(4)), "semibold", 16, TEXT_DIM)
        self.text(surf, "SPEED  KM/H", (x + w / 2, y + h + u(4)), "semibold", 16, TEXT_DIM, anchor="midtop")
        self.text(surf, f"{SPEED_STOPS[-1][0]}+", (x + w, y + h + u(4)), "semibold", 16, TEXT_DIM, anchor="topright")

    def _draw_title_card(self, surf, t):
        """Circuit name over the chase cam for the first seconds, then fades out."""
        alpha = float(np.clip((3.2 - t) / 0.8, 0, 1))
        if alpha <= 0:
            return
        u = self.u
        layer = pygame.Surface((self.chase_w, self.H), pygame.SRCALPHA)
        name = self.lap.track_name.upper()
        cx = self.chase_w / 2
        name_w = self.fonts.render(name, "bold", 42, TEXT).get_width()
        box = pygame.Rect(0, 0, max(u(760), name_w + u(90)), u(150))
        box.center = (round(cx), round(self.H * 0.3))
        pygame.draw.rect(layer, (8, 10, 14, 215), box, border_radius=round(u(16)))
        pygame.draw.rect(layer, ACCENT, (box.x, box.y, round(u(8)), box.h),
                         border_top_left_radius=round(u(16)), border_bottom_left_radius=round(u(16)))
        self.text(layer, name, (cx, box.y + u(48)), "bold", 42, anchor="center")
        tag = SPLIT_LABELS.get(self.lap.track_split, "")
        sub = f"{tag}  ·  AI DRIVER: SOFT ACTOR-CRITIC" if tag else "AI DRIVER: SOFT ACTOR-CRITIC"
        self.text(layer, sub, (cx, box.y + u(104)), "semibold", 22, ACCENT if tag else TEXT_DIM, anchor="center")
        layer.set_alpha(round(255 * alpha))
        surf.blit(layer, (0, 0))

    def _draw_finish_banner(self, surf):
        u = self.u
        box = pygame.Rect(0, 0, u(620), u(150))
        box.center = (round(self.chase_w / 2), round(self.H * 0.3))
        self.panel(surf, box, (8, 10, 14, 225), radius=16)
        pygame.draw.rect(surf, THROTTLE_GREEN, (box.x, box.y, round(u(8)), box.h),
                         border_top_left_radius=round(u(16)), border_bottom_left_radius=round(u(16)))
        self.text(surf, "LAP COMPLETE", (box.centerx, box.y + u(40)), "semibold", 26, THROTTLE_GREEN, anchor="center")
        self.text(surf, format_laptime(self.lap.lap_time), (box.centerx, box.y + u(96)), "bold", 64, anchor="center")

    # ----------------------------------------------------------- data panel
    def _panel_rect(self):
        return pygame.Rect(self.chase_w, 0, self.W - self.chase_w, self.H)

    def _prepare_map(self):
        """Static map surface plus the transform used to place the trail and car on it."""
        u = self.u
        panel = self._panel_rect()
        area = pygame.Rect(panel.x + u(40), u(150), panel.w - u(80), u(430))
        c = self.lap.centerline
        lo, hi = c.min(axis=0), c.max(axis=0)
        scale = min(area.w / (hi[0] - lo[0]), area.h / (hi[1] - lo[1]))
        mid = (lo + hi) / 2

        def to_map(p):
            p = np.asarray(p, dtype=float)
            return np.stack([area.centerx + (p[..., 0] - mid[0]) * scale,
                             area.centery - (p[..., 1] - mid[1]) * scale], axis=-1)

        surf = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        pts = to_map(np.vstack([c, c[:1]]))
        pygame.draw.lines(surf, (46, 50, 60), False, pts, max(2, round(u(16))))
        pygame.draw.lines(surf, (24, 26, 33), False, pts, max(1, round(u(9))))
        # Round joins: without them thick pygame lines leave notches at every vertex
        for p in pts:
            pygame.draw.circle(surf, (46, 50, 60), p, u(8))
        for p in pts:
            pygame.draw.circle(surf, (24, 26, 33), p, u(4.5))
        s0 = to_map(c[0])
        n0 = self.start_normal
        a, b = to_map(c[0] + n0 * 40), to_map(c[0] - n0 * 40)
        pygame.draw.line(surf, (240, 240, 240), a, b, max(2, round(u(4))))
        return {"surface": surf, "to_map": to_map, "trail": to_map(np.column_stack([self.lap.x, self.lap.y])),
                "start": s0}

    def _prepare_traces(self):
        u = self.u
        panel = self._panel_rect()
        x0, x1 = panel.x + u(70), panel.right - u(36)
        speed_box = pygame.Rect(x0, u(640), x1 - x0, u(190))
        pedal_box = pygame.Rect(x0, u(880), x1 - x0, u(120))
        xs = x0 + (x1 - x0) * self.lap.progress
        speed_y = speed_box.bottom - speed_box.h * np.clip(self.lap.kph / 350, 0, 1)
        pedal_y = pedal_box.centery - pedal_box.h / 2 * np.clip(self.lap.throttle, -1, 1)
        return {"speed_box": speed_box, "pedal_box": pedal_box, "xs": xs, "speed_y": speed_y,
                "pedal_y": pedal_y}

    def draw_panel(self, surf, t):
        u, lap = self.u, self.lap
        panel = self._panel_rect()
        surf.fill(PANEL_BG, panel)
        pygame.draw.line(surf, PANEL_LINE, panel.topleft, panel.bottomleft, max(1, round(u(2))))

        # Header
        x = panel.x + u(40)
        self.text(surf, lap.track_name.upper(), (x, u(34)), "bold", 30)
        length_km = lap.track_length / 1000
        meta = f"{lap.track_id.upper()}  ·  {length_km:.2f} KM"
        r = self.text(surf, meta, (x, u(78)), "semibold", 20, TEXT_DIM)
        tag = SPLIT_LABELS.get(lap.track_split)
        if tag:
            img = self.fonts.render(tag, "bold", 16, (255, 255, 255))
            pill = img.get_rect()
            pill.inflate_ip(u(22), u(8))
            pill.midleft = (r.right + u(18), r.centery)
            pygame.draw.rect(surf, ACCENT, pill, border_radius=pill.h // 2)
            surf.blit(img, img.get_rect(center=pill.center))

        # Map with the lap so far
        surf.blit(self._map["surface"], (0, 0))
        i0, a = lap.index(t)
        trail = self._map["trail"]
        w = max(2, round(u(5)))
        for q in range(i0):
            pygame.draw.line(surf, lap.color[q + 1], trail[q], trail[q + 1], w)
        here = trail[i0] + a * (trail[i0 + 1] - trail[i0])
        for rad, alpha in ((u(22), 40), (u(15), 80)):
            glow = pygame.Surface((round(2 * rad), round(2 * rad)), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*ACCENT, alpha), (rad, rad), rad)
            surf.blit(glow, (here[0] - rad, here[1] - rad))
        pygame.draw.circle(surf, (255, 255, 255), here, u(8))
        pygame.draw.circle(surf, ACCENT, here, u(5))

        self._draw_speed_trace(surf, i0, a)
        self._draw_pedal_trace(surf, i0, a)

        footer = f"MODEL {lap.model_name.upper()}  ·  STABLE-BASELINES3 SAC" if lap.model_name else "STABLE-BASELINES3 SAC"
        self.text(surf, footer, (x, self.H - u(74)), "semibold", 16, TEXT_DIM)
        self.text(surf, "github.com/AakashVijeta/racing-line-optimization", (x, self.H - u(48)),
                  "regular", 16, TEXT_DIM)

    def _cursor(self, surf, box, x):
        pygame.draw.line(surf, (255, 255, 255), (x, box.top), (x, box.bottom), max(1, round(self.u(2))))

    def _draw_speed_trace(self, surf, i0, a):
        u, tr, lap = self.u, self._traces, self.lap
        box = tr["speed_box"]
        self.text(surf, "SPEED", (box.x - u(30), box.y - u(40)), "semibold", 18, TEXT_DIM)
        for kph in (100, 200, 300):
            y = box.bottom - box.h * kph / 350
            pygame.draw.line(surf, PANEL_LINE, (box.x, y), (box.right, y), max(1, round(u(1))))
            self.text(surf, str(kph), (box.x - u(10), y), "regular", 14, TEXT_DIM, anchor="midright")
        n = i0 + 1
        xs, ys = tr["xs"][:n + 1].copy(), tr["speed_y"][:n + 1].copy()
        xs[-1] = tr["xs"][i0] + a * (tr["xs"][i0 + 1] - tr["xs"][i0])
        ys[-1] = tr["speed_y"][i0] + a * (tr["speed_y"][i0 + 1] - tr["speed_y"][i0])
        if len(xs) >= 2:
            fill = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
            poly = np.vstack([np.column_stack([xs, ys]), [[xs[-1], box.bottom], [xs[0], box.bottom]]])
            pygame.draw.polygon(fill, (255, 255, 255, 22), poly)
            surf.blit(fill, (0, 0))
            w = max(2, round(u(3)))
            for q in range(len(xs) - 1):
                pygame.draw.line(surf, lap.color[min(q + 1, lap.n - 1)], (xs[q], ys[q]), (xs[q + 1], ys[q + 1]), w)
        self._cursor(surf, box, xs[-1])

    def _draw_pedal_trace(self, surf, i0, a):
        u, tr, lap = self.u, self._traces, self.lap
        box = tr["pedal_box"]
        self.text(surf, "THROTTLE", (box.x - u(30), box.y - u(40)), "semibold", 18, THROTTLE_GREEN)
        self.text(surf, "BRAKE", (box.x + u(80), box.y - u(40)), "semibold", 18, BRAKE_RED)
        pygame.draw.line(surf, PANEL_LINE, (box.x, box.centery), (box.right, box.centery), max(1, round(u(1))))
        xs, ys = tr["xs"][:i0 + 2], tr["pedal_y"][:i0 + 2]
        fill = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        w = max(1, round(u(2)))
        for q in range(len(xs) - 1):
            col = THROTTLE_GREEN if lap.throttle[q + 1] >= 0 else BRAKE_RED
            pygame.draw.polygon(fill, (*col, 70), ((xs[q], box.centery), (xs[q], ys[q]), (xs[q + 1], ys[q + 1]),
                                                   (xs[q + 1], box.centery)))
        surf.blit(fill, (0, 0))
        for q in range(len(xs) - 1):
            col = THROTTLE_GREEN if lap.throttle[q + 1] >= 0 else BRAKE_RED
            pygame.draw.line(surf, col, (xs[q], ys[q]), (xs[q + 1], ys[q + 1]), w)
        self._cursor(surf, box, tr["xs"][i0] + a * (tr["xs"][i0 + 1] - tr["xs"][i0]))

    # -------------------------------------------------------------- render
    def render(self, t):
        """Draw the frame at time t (seconds from the start of the lap). Returns a Surface at out_size."""
        self.draw_chase(self.canvas, t)
        self.draw_hud(self.canvas, t)
        self.draw_panel(self.canvas, t)
        if self.canvas.get_size() == self.out_size:
            return self.canvas
        return pygame.transform.smoothscale(self.canvas, self.out_size)


def surface_to_array(surf):
    """pygame Surface -> (height, width, 3) uint8 array."""
    return np.ascontiguousarray(pygame.surfarray.pixels3d(surf).transpose(1, 0, 2))
