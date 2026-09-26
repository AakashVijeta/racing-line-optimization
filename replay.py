"""
Interactive replay of a recorded lap, drawn with the same renderer as export_video.py.

    python replay.py                      # suzuka_lap.npz
    python replay.py spa_lap.npz --width 1280

Controls: SPACE pause, LEFT/RIGHT seek 1 s (hold SHIFT for 5 s), R restart, ESC quit.
"""
import argparse

import pygame

from lap_render import LapData, LapRenderer


def main():
    parser = argparse.ArgumentParser(description="Interactive lap replay")
    parser.add_argument("npz", nargs="?", default="suzuka_lap.npz", help="file written by record_lap.py")
    parser.add_argument("--width", type=int, default=1600, help="window width (height follows 16:9)")
    parser.add_argument("--supersample", type=int, default=1, help="2 looks smoother but renders slower")
    args = parser.parse_args()

    pygame.init()
    size = (args.width, round(args.width * 9 / 16))
    screen = pygame.display.set_mode(size)
    lap = LapData(args.npz)
    pygame.display.set_caption(f"Lap replay: {lap.track_name}")
    renderer = LapRenderer(lap, *size, supersample=args.supersample)

    clock = pygame.time.Clock()
    t, paused, end = 0.0, False, lap.duration + 3.0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                step = 5.0 if event.mod & pygame.KMOD_SHIFT else 1.0
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    t = min(end, t + step)
                elif event.key == pygame.K_LEFT:
                    t = max(0.0, t - step)
                elif event.key == pygame.K_r:
                    t = 0.0
        if not paused:
            t = min(end, t + dt)
        screen.blit(renderer.render(t), (0, 0))
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
