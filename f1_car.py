"""
f1_car.py - top-down Formula 1 car for the lap replay.

Car frame (metres): +u = forward, +v = left, origin = mid-wheelbase.
That matches the sim: heading theta is counter-clockwise, and positive
steering turns the car left (theta_dot = v/L * tan(delta)).

How it draws
------------
* Everything that never changes (body, wings, rear tyres, suspension) is drawn
  ONCE onto a high-resolution surface, shrunk with smoothscale (this is the
  anti-aliasing), and stored as a sprite.  Two sprites are kept: brake light off / on.
* Each frame the sprite is rotated with rotozoom (one blit).
* The two FRONT tyres are drawn per frame, because they turn with the steering.

Usage
-----
    car = F1Car(ppm=8.0)
    car.draw(surface, (x, y), theta, steering, braking=True, shadow_offset=(4, 6))
"""
import math

import pygame

try:
    from pygame import gfxdraw
except ImportError:  # very old / stripped pygame builds
    gfxdraw = None

# ---- livery (generic, no team branding) ------------------------------------
RED = (208, 30, 40)
RED_DARK = (140, 16, 26)
CARBON = (22, 22, 26)
CARBON_LIGHT = (58, 58, 66)
WHITE = (238, 238, 238)
TYRE = (16, 16, 18)
TYRE_TREAD = (34, 34, 38)
HELMET = (255, 205, 0)
BRAKE_ON = (255, 50, 50)
BRAKE_OFF = (96, 12, 16)

# ---- geometry, metres -------------------------------------------------------
FRONT_AXLE = 1.80
REAR_AXLE = -1.80
FRONT_TRACK = 1.62        # centre to centre of the two front tyres
REAR_TRACK = 1.56
FRONT_TYRE = (0.74, 0.34)  # (length along car, width across car)
REAR_TYRE = (0.78, 0.46)


def _mirror(half):
    """Points along the +v (left) side, front to rear -> closed symmetric outline."""
    return list(half) + [(u, -v) for (u, v) in reversed(half)]


def _rect(u0, u1, v0, v1):
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _rotate(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(u * c - v * s, u * s + v * c) for (u, v) in points]


class F1Car:
    def __init__(self, ppm=5.0, visual_scale=2.5, supersample=4, max_steer=0.4, origin="rear"):
        """
        ppm          : pixels per metre of the camera (ReplayViewer.ppm)
        visual_scale : draw the car this many times bigger than real life
        supersample  : sprite is drawn this many times larger, then shrunk (smooth edges)
        max_steer    : road-wheel angle in radians at steering = +/-1 (env uses 0.4)
        origin       : which point of the car sits at `center` in draw().
                       "rear" = rear axle (this is what the sim's bicycle model tracks:
                       x += v*cos(theta)*dt with theta_dot = v/L*tan(delta)),
                       "mid"  = middle of the wheelbase.
        """
        self.k = ppm * visual_scale          # screen pixels per car-metre
        self.max_steer = max_steer
        self.origin = origin
        self._ss = supersample
        self.sprite_off = self._build_sprite(braking=False)
        self.sprite_on = self._build_sprite(braking=True)
        self.shadow = self._build_shadow(self.sprite_off)

    # ------------------------------------------------------------------ sprite
    def _build_sprite(self, braking):
        ss, k = self._ss, self.k * self._ss
        half_len, half_wid = 3.25, 1.35                     # canvas margin in metres
        w, h = int(2 * half_len * k), int(2 * half_wid * k)
        big = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy = w / 2, h / 2

        def px(pts):
            return [(cx + u * k, cy - v * k) for (u, v) in pts]

        def poly(color, pts):
            pygame.draw.polygon(big, color, px(pts))

        def line(color, p0, p1, width_m):
            pygame.draw.line(big, color, px([p0])[0], px([p1])[0], max(1, int(width_m * k)))

        # floor: dark edge visible around the sidepods -> gives depth
        poly(CARBON, _mirror([(1.30, 0.34), (0.55, 0.64), (-1.20, 0.64), (-1.95, 0.32)]))

        # suspension arms (drawn under the tyres)
        for side in (1, -1):
            for (u0, u1) in ((1.55, 2.05),):
                line(CARBON_LIGHT, (u0, side * 0.20), (FRONT_AXLE, side * (FRONT_TRACK / 2 - 0.12)), 0.035)
                line(CARBON_LIGHT, (u1, side * 0.20), (FRONT_AXLE, side * (FRONT_TRACK / 2 - 0.12)), 0.035)
            line(CARBON_LIGHT, (-1.55, side * 0.30), (REAR_AXLE, side * (REAR_TRACK / 2 - 0.15)), 0.04)
            line(CARBON_LIGHT, (-2.05, side * 0.22), (REAR_AXLE, side * (REAR_TRACK / 2 - 0.15)), 0.04)

        # rear tyres (they do not steer, so they live in the sprite)
        for side in (1, -1):
            lu, lv = REAR_TYRE
            cu, cv = REAR_AXLE, side * REAR_TRACK / 2
            poly(TYRE, _rect(cu - lu / 2, cu + lu / 2, cv - lv / 2, cv + lv / 2))
            poly(TYRE_TREAD, _rect(cu - lu / 2 + 0.06, cu + lu / 2 - 0.06, cv - lv * 0.18, cv + lv * 0.18))

        # rear wing: pillar, main plane, red flap, endplates, rain / brake light
        poly(CARBON, _rect(-2.50, -2.20, -0.06, 0.06))
        poly(CARBON, _rect(-2.86, -2.46, -0.52, 0.52))
        poly(RED, _rect(-2.86, -2.72, -0.50, 0.50))
        for side in (1, -1):
            v0, v1 = sorted((side * 0.49, side * 0.55))
            poly(CARBON_LIGHT, _rect(-2.90, -2.42, v0, v1))
        poly(BRAKE_ON if braking else BRAKE_OFF, _rect(-2.93, -2.84, -0.09, 0.09))

        # front wing: main plane, red flap, endplates, nose pillars
        poly(CARBON, _rect(2.42, 2.80, -1.00, 1.00))
        poly(RED, _rect(2.66, 2.80, -0.96, 0.96))
        for side in (1, -1):
            v0, v1 = sorted((side * 0.97, side * 1.03))
            poly(CARBON_LIGHT, _rect(2.40, 2.84, v0, v1))

        # body: nose -> monocoque -> sidepods -> tapered "coke bottle" rear
        body_half = [
            (2.48, 0.07), (2.00, 0.10), (1.45, 0.17), (1.05, 0.27), (0.72, 0.40),
            (0.35, 0.52), (-0.30, 0.52), (-0.95, 0.42), (-1.55, 0.26), (-2.05, 0.16), (-2.38, 0.13),
        ]
        poly(RED, _mirror(body_half))

        # sidepod shading + inlets
        pod = [(0.30, 0.30), (-0.30, 0.30), (-0.85, 0.30), (-0.85, 0.36), (-0.25, 0.50), (0.30, 0.50)]
        for side in (1, -1):
            poly(RED_DARK, [(u, side * v) for (u, v) in pod])
            iv0, iv1 = sorted((side * 0.32, side * 0.44))
            poly(CARBON, _rect(0.62, 0.74, iv0, iv1))

        # engine cover / airbox stripe and thin white nose line
        poly(RED_DARK, _mirror([(0.30, 0.13), (-0.40, 0.13), (-1.20, 0.11), (-2.05, 0.09)]))
        poly(WHITE, _rect(1.05, 2.42, -0.035, 0.035))

        # cockpit, helmet, halo
        poly(CARBON, _mirror([(0.90, 0.10), (0.70, 0.20), (0.15, 0.25), (-0.15, 0.17)]))
        pygame.draw.circle(big, HELMET, px([(0.36, 0.0)])[0], int(0.155 * k))
        pygame.draw.circle(big, CARBON, px([(0.36, 0.0)])[0], int(0.155 * k), max(1, int(0.03 * k)))
        line(CARBON_LIGHT, (0.16, 0.26), (0.88, 0.05), 0.045)
        line(CARBON_LIGHT, (0.16, -0.26), (0.88, -0.05), 0.045)
        line(CARBON_LIGHT, (0.88, 0.05), (0.88, -0.05), 0.045)

        return pygame.transform.smoothscale(big, (w // ss, h // ss))

    @staticmethod
    def _build_shadow(sprite, alpha=110):
        """Soft dark silhouette of the sprite, for a drop shadow under the car."""
        w, h = sprite.get_size()
        shadow = sprite.copy()
        shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)   # keep only the alpha
        shadow.fill((0, 0, 0, alpha), special_flags=pygame.BLEND_RGBA_MIN)
        # Blur by shrinking and growing back
        small = pygame.transform.smoothscale(shadow, (max(1, w // 6), max(1, h // 6)))
        return pygame.transform.smoothscale(small, (w, h))

    # -------------------------------------------------------------------- draw
    @staticmethod
    def _fill_aa(surface, color, pts):
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        if gfxdraw is not None:
            gfxdraw.filled_polygon(surface, pts, color)
            gfxdraw.aapolygon(surface, pts, color)
        else:
            pygame.draw.polygon(surface, color, pts)

    def draw(self, surface, center, theta, steering=0.0, braking=False, shadow_offset=None):
        """
        surface  : target pygame Surface
        center   : (x, y) screen position of the car's origin (mid-wheelbase)
        theta    : heading in radians (counter-clockwise, as in the sim)
        steering : raw action in [-1, 1]; +1 = full left lock
        braking  : light the rear brake light
        shadow_offset : (dx, dy) screen offset of a drop shadow, or None for no shadow
        """
        sx, sy = center
        k = self.k
        c, s = math.cos(theta), math.sin(theta)
        if self.origin == "rear":
            # the drawn car is centred on mid-wheelbase, so move it forward by half the wheelbase
            sx += k * FRONT_AXLE * c
            sy -= k * FRONT_AXLE * s

        def to_screen(u, v):
            # car frame -> screen (screen y points down, so v is subtracted)
            return sx + k * (u * c - v * s), sy - k * (u * s + v * c)

        if shadow_offset is not None:
            rot = pygame.transform.rotozoom(self.shadow, math.degrees(theta), 1.0)
            surface.blit(rot, rot.get_rect(center=(round(sx + shadow_offset[0]), round(sy + shadow_offset[1]))))

        # front tyres: rotate each rectangle by the wheel angle about its own centre
        delta = steering * self.max_steer
        lu, lv = FRONT_TYRE
        for side in (1, -1):
            cu, cv = FRONT_AXLE, side * FRONT_TRACK / 2
            tyre = _rotate(_rect(-lu / 2, lu / 2, -lv / 2, lv / 2), delta)
            tread = _rotate(_rect(-lu / 2 + 0.06, lu / 2 - 0.06, -lv * 0.18, lv * 0.18), delta)
            self._fill_aa(surface, TYRE, [to_screen(cu + u, cv + v) for u, v in tyre])
            self._fill_aa(surface, TYRE_TREAD, [to_screen(cu + u, cv + v) for u, v in tread])

        # body sprite on top (rotozoom rotates counter-clockwise, same sense as theta)
        img = self.sprite_on if braking else self.sprite_off
        rot = pygame.transform.rotozoom(img, math.degrees(theta), 1.0)
        surface.blit(rot, rot.get_rect(center=(round(sx), round(sy))))