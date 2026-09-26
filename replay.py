import numpy as np
import pygame
import sys
import math
import argparse
from functools import lru_cache
from f1_car import F1Car
from PIL import Image, ImageDraw, ImageFont

class PILTextRenderer:
    def __init__(self, size=24):
        self.size = size
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except IOError:
            self.font = ImageFont.load_default()

    @lru_cache(maxsize=512)
    def render(self, text, color=(255, 255, 255)):
        # Cached: HUD labels repeat across frames and PIL rendering is slow
        img = Image.new('RGBA', (400, 60), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=self.font, fill=color)
        
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        raw_str = img.tobytes("raw", 'RGBA')
        surface = pygame.image.fromstring(raw_str, img.size, 'RGBA')
        return surface

class ReplayViewer:
    def __init__(self, npz_path="suzuka_lap.npz", width=1280, height=720, ppm=5.0):
        self.width = width
        self.height = height
        self.ppm = ppm
        self.car_sprite = F1Car(ppm=self.ppm)
        data = np.load(npz_path)
        self.x = data['x']
        self.y = data['y']
        self.theta = data['theta']
        self.v = data['v']
        self.throttle = data['throttle']
        self.steering = data['steering']
        self.arc = data['arc']
        self.left_boundary = data['left_boundary']
        self.right_boundary = data['right_boundary']
        self.centerline = data['centerline']
        self.dt = float(data['dt'])
        self.track_length = float(data['track_length'])
        
        segment_lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        self.centerline_arc = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        self.n_frames = len(self.x)
        self.max_v = np.max(self.v)
        self.trail_colors = [self.get_speed_color(s) for s in self.v]
        
        self.minimap_surface = None
        self._init_minimap()
        
        self.font_large = PILTextRenderer(24)
        self.font_small = PILTextRenderer(24)

    def get_speed_color(self, v):
        ratio = min(max(v / (self.max_v + 1e-3), 0.0), 1.0)
        if ratio < 0.5:
            r = 255
            g = int(255 * (ratio * 2))
            b = 0
        else:
            r = int(255 * (1.0 - (ratio - 0.5) * 2))
            g = 255
            b = 0
        return (r, g, b)

    def _init_minimap(self, mm_size=200, padding=10):
        self.mm_size = mm_size
        self.minimap_surface = pygame.Surface((mm_size, mm_size), pygame.SRCALPHA)
        self.minimap_surface.fill((0, 0, 0, 150))
        
        all_pts = np.vstack([self.left_boundary, self.right_boundary])
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        self.mm_scale = min((mm_size - 2*padding) / w, (mm_size - 2*padding) / h)
        self.mm_cx = (min_x + max_x) / 2
        self.mm_cy = (min_y + max_y) / 2
        
        def to_mm(px, py):
            sx = mm_size / 2 + (px - self.mm_cx) * self.mm_scale
            sy = mm_size / 2 - (py - self.mm_cy) * self.mm_scale
            return (int(sx), int(sy))
            
        pts_left = [to_mm(p[0], p[1]) for p in self.left_boundary]
        pts_right = [to_mm(p[0], p[1]) for p in self.right_boundary]
        
        pygame.draw.lines(self.minimap_surface, (100, 100, 100), True, pts_left, 1)
        pygame.draw.lines(self.minimap_surface, (100, 100, 100), True, pts_right, 1)

    def world_to_screen(self, px, py, cx, cy):
        if isinstance(px, np.ndarray):
            sx = self.width / 2 + (px - cx) * self.ppm
            sy = self.height / 2 - (py - cy) * self.ppm
            return np.column_stack((sx, sy)).astype(int).tolist()
        else:
            sx = self.width / 2 + (px - cx) * self.ppm
            sy = self.height / 2 - (py - cy) * self.ppm
            return sx, sy

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def state_at(self, t):
        idx_float = t / self.dt
        idx_0 = int(math.floor(idx_float))
        idx_1 = min(idx_0 + 1, self.n_frames - 1)
        alpha = idx_float - idx_0
        
        if idx_0 >= self.n_frames - 1:
            idx_0 = self.n_frames - 1
            idx_1 = self.n_frames - 1
            alpha = 0.0
            
        cx = self.x[idx_0] + alpha * (self.x[idx_1] - self.x[idx_0])
        cy = self.y[idx_0] + alpha * (self.y[idx_1] - self.y[idx_0])
        cv = self.v[idx_0] + alpha * (self.v[idx_1] - self.v[idx_0])
        cthrottle = self.throttle[idx_0] + alpha * (self.throttle[idx_1] - self.throttle[idx_0])
        csteering = self.steering[idx_0] + alpha * (self.steering[idx_1] - self.steering[idx_0])
        carc = self.arc[idx_0] + alpha * (self.arc[idx_1] - self.arc[idx_0])
        
        th_0 = self.theta[idx_0]
        th_1 = self.theta[idx_1]
        diff = self.wrap_angle(th_1 - th_0)
        ctheta = th_0 + alpha * diff
        
        return cx, cy, ctheta, cv, cthrottle, csteering, carc, idx_0

    def draw_minimap(self, surface, cx, cy):
        # Top right corner
        x_base = self.width - self.mm_size - 20
        y_base = 20
        surface.blit(self.minimap_surface, (x_base, y_base))
        
        sx = x_base + self.mm_size / 2 + (cx - self.mm_cx) * self.mm_scale
        sy = y_base + self.mm_size / 2 - (cy - self.mm_cy) * self.mm_scale
        pygame.draw.circle(surface, (255, 0, 0), (int(sx), int(sy)), 4)

    def draw_frame_by_state(self, surface, cx, cy, ctheta, cv, cthrottle, csteering, carc, trail_up_to_idx=None, lap_time=0.0):
        surface.fill((40, 40, 40))
        
        safe_arc = carc % self.track_length
        nearest_idx = np.searchsorted(self.centerline_arc, safe_arc) % len(self.centerline)
        
        track_window = 200
        n_pts = len(self.left_boundary)
        start_idx = (nearest_idx - track_window) % n_pts
        end_idx = (nearest_idx + track_window) % n_pts
        
        if start_idx < end_idx:
            indices = np.arange(start_idx, end_idx)
        else:
            indices = np.concatenate((np.arange(start_idx, n_pts), np.arange(0, end_idx)))
            
        left_pts = self.left_boundary[indices]
        right_pts = self.right_boundary[indices]
        
        left_screen = self.world_to_screen(left_pts[:, 0], left_pts[:, 1], cx, cy)
        right_screen = self.world_to_screen(right_pts[:, 0], right_pts[:, 1], cx, cy)
        
        for i in range(len(indices) - 1):
            poly = [left_screen[i], right_screen[i], right_screen[i+1], left_screen[i+1]]
            pygame.draw.polygon(surface, (80, 80, 80), poly)
            
        pygame.draw.lines(surface, (255, 255, 255), False, left_screen, 2)
        pygame.draw.lines(surface, (255, 255, 255), False, right_screen, 2)
        
        if trail_up_to_idx is not None and trail_up_to_idx > 0:
            for i in range(max(0, trail_up_to_idx - 300), trail_up_to_idx):
                p1 = self.world_to_screen(self.x[i], self.y[i], cx, cy)
                p2 = self.world_to_screen(self.x[i+1], self.y[i+1], cx, cy)
                pygame.draw.line(surface, self.trail_colors[i], p1, p2, 4)
                
            p1 = self.world_to_screen(self.x[trail_up_to_idx], self.y[trail_up_to_idx], cx, cy)
            p2 = self.world_to_screen(cx, cy, cx, cy)
            pygame.draw.line(surface, self.trail_colors[trail_up_to_idx], p1, p2, 4)

        self.car_sprite.draw(surface, (self.width/2, self.height/2), ctheta, csteering, braking=(cthrottle < -0.05))
        
        # Display HUD using high quality PIL fonts
        speed_kph = cv * 3.6
        surface.blit(self.font_large.render(f"SPEED: {speed_kph:.1f} KM/H"), (20, 20))
        surface.blit(self.font_large.render(f"TIME: {lap_time:.2f} S"), (20, 50))
        
        # Bars with explicit numbers
        bar_w = 200
        bar_h = 20
        x_base = 20
        y_base = self.height - 100
        
        # Throttle/Brake
        pygame.draw.rect(surface, (100, 100, 100), (x_base, y_base, bar_w, bar_h))
        if cthrottle >= 0:
            w = int(cthrottle * bar_w)
            pygame.draw.rect(surface, (0, 255, 0), (x_base, y_base, w, bar_h))
            val_text = f"THROTTLE: {cthrottle*100:.0f}%"
        else:
            w = int(abs(cthrottle) * bar_w)
            pygame.draw.rect(surface, (255, 0, 0), (x_base, y_base, w, bar_h))
            val_text = f"BRAKE: {abs(cthrottle)*100:.0f}%"
            
        lbl_surf = self.font_small.render(val_text)
        surface.blit(lbl_surf, (x_base, y_base - 30))
        
        # Steering
        y_base += 60
        pygame.draw.rect(surface, (100, 100, 100), (x_base, y_base, bar_w, bar_h))
        pygame.draw.line(surface, (255, 255, 255), (x_base + bar_w/2, y_base), (x_base + bar_w/2, y_base + bar_h), 2)
        
        sw = int(abs(csteering) * (bar_w / 2))
        if csteering >= 0:
            pygame.draw.rect(surface, (0, 165, 255), (x_base + bar_w/2 - sw, y_base, sw, bar_h))
            val_text = f"STEERING: {csteering*100:.0f}% LEFT"
        else:
            pygame.draw.rect(surface, (255, 165, 0), (x_base + bar_w/2, y_base, sw, bar_h))
            val_text = f"STEERING: {abs(csteering)*100:.0f}% RIGHT"
            
        lbl_surf = self.font_small.render(val_text)
        surface.blit(lbl_surf, (x_base, y_base - 30))
        
        # Progress Bar (Top center)
        prog_w = 600
        prog_h = 10
        px = self.width/2 - prog_w/2
        py = 20
        pygame.draw.rect(surface, (100, 100, 100), (px, py, prog_w, prog_h))
        progress = min(max(carc / self.track_length, 0.0), 1.0)
        pygame.draw.rect(surface, (255, 255, 255), (px, py, int(prog_w * progress), prog_h))

        self.draw_minimap(surface, cx, cy)

    def run(self):
        pygame.init()
        
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Lap Replay")
        clock = pygame.time.Clock()
        
        t = 0.0
        running = True
        paused = False
        
        total_time = (self.n_frames - 1) * self.dt
        
        while running:
            dt_real = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        t = min(t + 1.0, total_time)
                    elif event.key == pygame.K_LEFT:
                        t = max(t - 1.0, 0.0)
                        
            if not paused:
                t = min(t + dt_real, total_time)
                
            state = self.state_at(t)
            self.draw_frame_by_state(screen, *state[:7], trail_up_to_idx=state[7], lap_time=t)
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded lap (SPACE pause, LEFT/RIGHT seek)")
    parser.add_argument("npz", nargs="?", default="suzuka_lap.npz", help="file written by record_lap.py")
    args = parser.parse_args()
    viewer = ReplayViewer(npz_path=args.npz)
    viewer.run()
