"""
Renderer for a recorded lap (see record_lap.py).

Layout, in a 1920x1080 design space that scales to any output size:

    +-------------+------------------------+----------------+
    | lap time    |                        | circuit name   |
    | and sectors | chase cam: grey track  |                |
    |             | with white edge blocks,|   track map    |
    | speed, on   | the racing line        |                |
    | its scale   | coloured by speed,     |   speed trace  |
    | pedals      | the car                | credit         |
    +-------------+------------------------+----------------+

Black throughout, and nothing is drawn over the road. One colour scale (red
when slow to green flat out) carries speed everywhere it appears.

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
LEFT_W, RIGHT_W = 360, 700  # design-space widths of the instrument and data columns; the chase cam takes the rest
PAD = 32                    # inner margin of the instrument column

# ---- palette -----------------------------------------------------------------
GROUND = (6, 6, 8)        # around the track: the same black as the two columns
ASPHALT = (44, 45, 49)
KERB = (236, 236, 236)    # white blocks along both edges, the whole way round
KERB_DEPTH = 2.6          # m, how far the blocks reach out from the track
KERB_BLOCK = 1.2          # m, length of one block (and of the gap after it)
LINE_AHEAD = (120, 122, 130)
PANEL_BG = (6, 6, 8)
PANEL_LINE = (36, 36, 40)
MAP_TRACK = (52, 53, 58)
TEXT = (240, 240, 240)
TEXT_DIM = (140, 140, 146)
THROTTLE_GREEN = (64, 220, 128)
BRAKE_RED = (255, 72, 72)

# Speed colour scale (km/h -> RGB): red when slow, through orange and yellow, to green
# flat out. Lightness rises at every stop too, so the order reads even for viewers
# who can't tell red from green.
SPEED_STOPS = [(60, (200, 30, 30)), (140, (235, 110, 30)), (210, (240, 180, 40)),
               (270, (170, 225, 60)), (320, (80, 240, 120))]

SPLIT_NOTES = {"test": "A circuit the agent never saw in training.",
               "val": "A validation circuit, not used for training.",
               "train": "One of the agent's training circuits."}


def speed_color(kph):
    """Map speeds (scalar or array, km/h) onto SPEED_STOPS. Returns an (..., 3) uint8 array."""
    kph = np.asarray(kph, dtype=float)
    xs = [s for s, _ in SPEED_STOPS]
    rgb = [np.interp(kph, xs, [c[i] for _, c in SPEED_STOPS]) for i in range(3)]
    return np.stack(rgb, axis=-1).astype(np.uint8)


def format_laptime(seconds):
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{secs:06.3f}"


def format_sector(seconds):
    """Sector time: plain seconds under a minute, lap-time format above."""
    return f"{seconds:.3f}" if seconds < 60 else format_laptime(seconds)


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

        def get(key, default=None):
            return d[key] if key in d.files else default

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
        # Three sectors of equal distance: the time each one ends
        times = np.arange(self.n) * self.dt
        arc = np.maximum.accumulate(self.arc)
        self.sector_ends = [float(np.interp(self.track_length * k / 3, arc, times)) for k in (1, 2)]
        self.sector_ends.append(self.lap_time)

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
            px = max(6, round(size * self.scale))
            path = os.path.join(FONT_DIR, self.FILES[weight])
            try:
                self._fonts[key] = ImageFont.truetype(path, px)
            except OSError:
                self._fonts[key] = ImageFont.load_default(px)
        return self._fonts[key]

    def advance(self, msg, weight, size):
        """Width of msg in canvas pixels, as the font would set it."""
        return self._font(weight, size).getlength(msg)

    def render(self, msg, weight, size, color):
        key = (msg, weight, size, tuple(color))
        img = self._cache.get(key)
        if img is None:
            from PIL import Image, ImageDraw
            font = self._font(weight, size)
            ascent, descent = font.getmetrics()
            width = max(1, math.ceil(font.getlength(msg)))
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
        left = round(self.W * LEFT_W / DESIGN_W)
        self.chase = pygame.Rect(left, 0, self.W - left - round(self.W * RIGHT_W / DESIGN_W), self.H)

        self._prepare_track()
        self._prepare_camera()
        self.car = F1Car(ppm=8.0 * self.s, visual_scale=2.0, supersample=3)
        self._map = self._prepare_map()
        self._trace = self._prepare_trace()

    # ------------------------------------------------------------- helpers
    def u(self, v):
        """Design units -> canvas pixels."""
        return v * self.s

    def text(self, surf, msg, pos, weight="regular", size=24, color=TEXT, anchor="topleft"):
        img = self.fonts.render(msg, weight, size, color)
        rect = img.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        surf.blit(img, rect)
        return rect

    def number(self, surf, msg, pos, weight="bold", size=24, color=TEXT, anchor="topleft"):
        """Text with every digit in a cell of the same width, so a changing readout stays still."""
        cell = max(self.fonts.advance(d, weight, size) for d in "0123456789")
        widths = [cell if ch.isdigit() else self.fonts.advance(ch, weight, size) for ch in msg]
        height = self.fonts.render("0", weight, size, color).get_height()
        box = pygame.Rect(0, 0, round(sum(widths)), height)
        setattr(box, anchor, (round(pos[0]), round(pos[1])))
        x = box.x
        for ch, w in zip(msg, widths):
            img = self.fonts.render(ch, weight, size, color)
            surf.blit(img, (round(x + (w - img.get_width()) / 2), box.y))
            x += w
        return box

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
        self.n_pts = len(c)

        # White blocks along both edges. The centerline is up to 5 m between points,
        # so the blocks are cut from a resampling at block length.
        loop = np.vstack([c, c[:1]])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
        s = np.arange(0.0, arc[-1], KERB_BLOCK)
        pts = np.column_stack([np.interp(s, arc, loop[:, 0]), np.interp(s, arc, loop[:, 1])])
        n = np.column_stack([np.interp(s, arc, np.r_[normal[:, 0], normal[0, 0]]),
                             np.interp(s, arc, np.r_[normal[:, 1], normal[0, 1]])])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        self.block_mid = pts
        self.block_in_l, self.block_out_l = pts + n * half, pts + n * (half + KERB_DEPTH)
        self.block_in_r, self.block_out_r = pts - n * half, pts - n * (half + KERB_DEPTH)
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
        origin = (self.chase.centerx, self.H * 0.64)

        surf.set_clip(self.chase)
        surf.fill(GROUND, self.chase)

        # Segments near the camera (both ends checked so long straights aren't culled)
        radius_m = math.hypot(self.chase.w, self.H) / ppm * 0.75 + 20
        near = np.linalg.norm(lap.centerline - cam, axis=1) < radius_m
        seg_on = near | np.roll(near, -1)
        idx = np.nonzero(seg_on)[0]
        j = (idx + 1) % self.n_pts

        def xf(p):
            return self._xf(p, cam, heading, ppm, origin)

        # Every other resampled slice is a block; the ones between are the gaps
        near_b = np.nonzero(np.linalg.norm(self.block_mid - cam, axis=1) < radius_m)[0]
        nb = len(self.block_mid)
        for inner, outer in ((self.block_in_l, self.block_out_l), (self.block_in_r, self.block_out_r)):
            inner, outer = xf(inner), xf(outer)
            for i in near_b[near_b % 2 == 0]:
                k = (i + 1) % nb
                pygame.draw.polygon(surf, KERB, (inner[i], inner[k], outer[k], outer[i]))
        edge_l, edge_r = xf(self.edge_l), xf(self.edge_r)
        for i, k in zip(idx, j):
            pygame.draw.polygon(surf, ASPHALT, (edge_l[i], edge_l[k], edge_r[k], edge_r[i]))

        self._draw_start_line(surf, xf)
        self._draw_racing_line(surf, t, xf, ppm, cam, radius_m)

        car_pos = xf(np.array([[st["x"], st["y"]]]))[0]
        car_theta = st["theta"] - heading + math.pi / 2
        braking = st["throttle"] < -0.05
        self.car.draw(surf, car_pos, car_theta, st["steering"], braking=braking,
                      shadow_offset=(self.u(5), self.u(7)))

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

        # Line still to come: a thin grey line, so the viewer sees the agent's chosen line
        ahead = np.vstack([head, pos[i0 + 1: i0 + 1 + int(6.0 / lap.dt)]])
        if len(ahead) >= 2:
            pygame.draw.lines(surf, LINE_AHEAD, False, xf(ahead), max(1, round(self.u(2))))

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

    # ---------------------------------------------------- instrument column
    def draw_instruments(self, surf, t):
        """Left column: lap time and sectors at the top, speed in the middle, pedals at the bottom."""
        col = pygame.Rect(0, 0, self.chase.x, self.H)
        surf.fill(PANEL_BG, col)
        pygame.draw.line(surf, PANEL_LINE, col.topright, col.bottomright, max(1, round(self.u(2))))
        st = self.lap.state(t)
        self._draw_lap_timer(surf, t, st)
        self._draw_speed(surf, st)
        self._draw_pedals(surf, st)

    def _draw_lap_timer(self, surf, t, st):
        u, lap = self.u, self.lap
        x, w = u(PAD), self.chase.x - 2 * u(PAD)
        shown = min(t, lap.lap_time)
        self.text(surf, "Lap complete" if t >= lap.duration else "Lap time", (x, u(34)), "regular", 22, TEXT_DIM)
        self.number(surf, format_laptime(shown), (x - u(3), u(60)), "bold", 72)

        # Lap progress, split into the three sectors
        bar = pygame.Rect(round(x), round(u(160)), round(w), max(2, round(u(5))))
        pygame.draw.rect(surf, PANEL_LINE, bar)
        filled = bar.copy()
        filled.w = max(1, round(bar.w * st["progress"]))
        pygame.draw.rect(surf, TEXT, filled)
        gap = max(2, round(u(4)))
        for k in (1, 2):
            pygame.draw.rect(surf, PANEL_BG, (bar.x + bar.w * k / 3 - gap / 2, bar.y, gap, bar.h))

        start = 0.0
        for k, end in enumerate(lap.sector_ends):
            cx = x + w * k / 3
            self.text(surf, f"Sector {k + 1}", (cx, u(180)), "regular", 18, TEXT_DIM)
            if shown >= end:
                self.number(surf, format_sector(end - start), (cx, u(204)), "semibold", 25)
            elif shown >= start:
                self.number(surf, format_sector(shown - start), (cx, u(204)), "semibold", 25, TEXT_DIM)
            start = end

    def _draw_speed(self, surf, st):
        """Speed as one large figure, over the colour scale that the racing line uses."""
        u = self.u
        x, w = u(PAD), self.chase.x - 2 * u(PAD)
        r = self.number(surf, f"{st['kph']:.0f}", (x - u(3), u(292)), "bold", 96)
        self.text(surf, "km/h", (r.right + u(12), r.bottom - u(26)), "regular", 24, TEXT_DIM, anchor="bottomleft")

        lo, hi = SPEED_STOPS[0][0], SPEED_STOPS[-1][0]
        y, h = u(446), u(10)
        for q in range(round(w)):
            pygame.draw.line(surf, tuple(speed_color(lo + (hi - lo) * q / w)), (x + q, y), (x + q, y + h))
        # Marker: where the current speed sits on the scale
        mx = x + w * float(np.clip((st["kph"] - lo) / (hi - lo), 0, 1))
        mw = max(2, round(u(4)))
        pygame.draw.rect(surf, PANEL_BG, (mx - mw / 2 - u(2), y - u(8), mw + u(4), h + u(16)))
        pygame.draw.rect(surf, TEXT, (mx - mw / 2, y - u(6), mw, h + u(12)))
        self.text(surf, str(lo), (x, y + h + u(10)), "regular", 18, TEXT_DIM)
        self.text(surf, f"{hi}+", (x + w, y + h + u(10)), "regular", 18, TEXT_DIM, anchor="topright")

    def _draw_pedals(self, surf, st):
        """One bar from a centre mark: braking grows to the left, throttle to the right."""
        u = self.u
        x, w = u(PAD), self.chase.x - 2 * u(PAD)
        y, h = u(892) - u(14), u(14)
        cx = x + w / 2
        self.text(surf, "Brake", (x, y - u(36)), "regular", 22, TEXT_DIM)
        self.text(surf, "Throttle", (x + w, y - u(36)), "regular", 22, TEXT_DIM, anchor="topright")
        pygame.draw.line(surf, PANEL_LINE, (x, y + h / 2), (x + w, y + h / 2), max(1, round(u(2))))
        throttle = float(np.clip(st["throttle"], -1, 1))
        if throttle > 0:
            pygame.draw.rect(surf, THROTTLE_GREEN, (cx, y, w / 2 * throttle, h))
        elif throttle < 0:
            pygame.draw.rect(surf, BRAKE_RED, (cx + w / 2 * throttle, y, -w / 2 * throttle, h))
        pygame.draw.rect(surf, TEXT, (cx - u(1.5), y - u(6), u(3), h + u(12)))

    # ------------------------------------------------------------ map column
    def _data_rect(self):
        """The inner area of the right column: circuit name, map, speed trace and credit."""
        x = self.chase.right + self.u(40)
        return pygame.Rect(round(x), 0, round(self.W - x - self.u(40)), self.H)

    def _prepare_map(self):
        """Static map surface plus the transform used to place the trail and car on it."""
        u = self.u
        col = self._data_rect()
        area = pygame.Rect(col.x + u(10), u(170), col.w - u(20), u(440))
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
        pygame.draw.lines(surf, MAP_TRACK, False, pts, max(2, round(u(13))))
        # Round joins: without them thick pygame lines leave notches at every vertex
        for p in pts:
            pygame.draw.circle(surf, MAP_TRACK, p, u(6.5))
        n0 = self.start_normal
        a, b = to_map(c[0] + n0 * 45), to_map(c[0] - n0 * 45)
        pygame.draw.line(surf, TEXT, a, b, max(2, round(u(4))))
        return {"surface": surf, "to_map": to_map, "trail": to_map(np.column_stack([self.lap.x, self.lap.y]))}

    def _prepare_trace(self):
        """Where each sample of the lap sits on the speed chart."""
        u = self.u
        col = self._data_rect()
        x0, x1 = col.x + u(58), col.right   # the gutter holds the y-axis tick labels
        box = pygame.Rect(round(x0), round(u(712)), round(x1 - x0), round(u(180)))
        xs = x0 + (x1 - x0) * self.lap.progress
        ys = box.bottom - box.h * np.clip(self.lap.kph / 350, 0, 1)
        return {"box": box, "xs": xs, "ys": ys}

    def _draw_speed_trace(self, surf, x, i0, a):
        u, tr, lap = self.u, self._trace, self.lap
        box = tr["box"]
        self.text(surf, "Speed, km/h", (x, box.y - u(42)), "regular", 22, TEXT_DIM)
        for kph in (100, 200, 300):
            y = box.bottom - box.h * kph / 350
            pygame.draw.line(surf, PANEL_LINE, (box.x, y), (box.right, y), max(1, round(u(1))))
            self.text(surf, str(kph), (box.x - u(12), y), "regular", 18, TEXT_DIM, anchor="midright")
        n = i0 + 1
        xs, ys = tr["xs"][:n + 1].copy(), tr["ys"][:n + 1].copy()
        xs[-1] = tr["xs"][i0] + a * (tr["xs"][i0 + 1] - tr["xs"][i0])
        ys[-1] = tr["ys"][i0] + a * (tr["ys"][i0 + 1] - tr["ys"][i0])
        pygame.draw.line(surf, TEXT, (xs[-1], box.top), (xs[-1], box.bottom), max(1, round(u(2))))
        w = max(2, round(u(3)))
        for q in range(len(xs) - 1):
            pygame.draw.line(surf, lap.color[min(q + 1, lap.n - 1)], (xs[q], ys[q]), (xs[q + 1], ys[q + 1]), w)
        # Head of the trace: a dot with a surface-coloured ring, so it reads where it crosses the line
        pygame.draw.circle(surf, PANEL_BG, (xs[-1], ys[-1]), u(9))
        pygame.draw.circle(surf, TEXT, (xs[-1], ys[-1]), u(6))
        # The x-axis is distance around the lap
        self.text(surf, "start", (box.x, box.bottom + u(8)), "regular", 16, TEXT_DIM)
        self.text(surf, "finish", (box.right, box.bottom + u(8)), "regular", 16, TEXT_DIM, anchor="topright")

    def _fit_size(self, msg, weight, size, max_w, min_size=18):
        """Largest font size up to `size` at which msg fits in max_w canvas pixels."""
        while size > min_size and self.fonts.render(msg, weight, size, TEXT).get_width() > max_w:
            size -= 1
        return size

    def draw_panel(self, surf, t):
        """Right column: circuit name, the map with the lap so far, the speed trace and the credit."""
        u, lap = self.u, self.lap
        surf.fill(PANEL_BG, (self.chase.right, 0, self.W - self.chase.right, self.H))
        pygame.draw.line(surf, PANEL_LINE, self.chase.topright, self.chase.bottomright, max(1, round(u(2))))
        col = self._data_rect()
        x = col.x

        self.text(surf, lap.track_name, (x, u(34)), "bold", self._fit_size(lap.track_name, "bold", 36, col.w))
        note = SPLIT_NOTES.get(lap.track_split, "")
        line = f"{lap.track_length / 1000:.2f} km. {note}".strip()
        self.text(surf, line, (x, u(88)), "regular", self._fit_size(line, "regular", 22, col.w), TEXT_DIM)

        # Map with the lap so far
        surf.blit(self._map["surface"], (0, 0))
        i0, a = lap.index(t)
        trail = self._map["trail"]
        w = max(2, round(u(5)))
        for q in range(i0):
            pygame.draw.line(surf, lap.color[q + 1], trail[q], trail[q + 1], w)
        here = trail[i0] + a * (trail[i0 + 1] - trail[i0])
        pygame.draw.circle(surf, PANEL_BG, here, u(12))
        pygame.draw.circle(surf, TEXT, here, u(8))

        self._draw_speed_trace(surf, x, i0, a)

        credit = f"Model {lap.model_name}, Soft Actor-Critic" if lap.model_name else "Soft Actor-Critic"
        self.text(surf, credit, (x, self.H - u(62)), "regular", 18, TEXT_DIM)
        self.text(surf, "github.com/AakashVijeta/racing-line-optimization", (x, self.H - u(38)),
                  "regular", 18, TEXT_DIM)

    # -------------------------------------------------------------- render
    def render(self, t):
        """Draw the frame at time t (seconds from the start of the lap). Returns a Surface at out_size."""
        self.draw_chase(self.canvas, t)
        self.draw_instruments(self.canvas, t)
        self.draw_panel(self.canvas, t)
        if self.canvas.get_size() == self.out_size:
            return self.canvas
        return pygame.transform.smoothscale(self.canvas, self.out_size)


def surface_to_array(surf):
    """pygame Surface -> (height, width, 3) uint8 array."""
    return np.ascontiguousarray(pygame.surfarray.pixels3d(surf).transpose(1, 0, 2))
