import pygame
import numpy as np
from car import Car
from plot_track import is_on_track, make_oval, compute_boundaries, close_loop

pygame.init()
SCREEN_W, SCREEN_H = 800, 500
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()


WORLD_SCALE = 2.0   #tune this so the whole track fits on screen
OFFSET_X, OFFSET_Y = SCREEN_W // 2, SCREEN_H // 2

def world_to_screen(x, y):
    screen_x = OFFSET_X + x * WORLD_SCALE
    screen_y = OFFSET_Y - y * WORLD_SCALE   
    return int(screen_x), int(screen_y)

# --- set up track and car ---
centerline = make_oval()
left, right = compute_boundaries(centerline, track_width=15)
left_closed = close_loop(left)
right_closed = close_loop(right)

car = Car(x=-100, y=-60, theta=0) 

running = True
while running:
    dt = clock.tick(60) / 1000.0  

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    max_steer = 0.4  # radians
    if keys[pygame.K_LEFT]:
        steering_angle = max_steer
    elif keys[pygame.K_RIGHT]:
        steering_angle = -max_steer
    else:
        steering_angle = 0.0

    max_throttle = 20.0
    if keys[pygame.K_UP]:
        throttle = max_throttle
    elif keys[pygame.K_DOWN]:
        throttle = -max_throttle
    else:
        throttle = 0.0

    car.step(steering_angle, throttle, dt)

    screen.fill((30, 30, 30))  # dark background

    left_array = []
    for point in range(len(left_closed)):
        screen_x, screen_y = world_to_screen(left_closed[point][0], left_closed[point][1])
        left_array.append((screen_x, screen_y))

    pygame.draw.lines(screen, (255, 255, 255), False, left_array, 2)
    right_array = []
    for point in range(len(right_closed)):
        screen_x, screen_y = world_to_screen(right_closed[point][0], right_closed[point][1])
        right_array.append((screen_x, screen_y))
    pygame.draw.lines(screen, (255, 255, 255), False, right_array, 2)

    car_x, car_y = world_to_screen(car.x, car.y)
    on_track = is_on_track(car.x, car.y, centerline, track_width=15)
    car_color = (0, 255, 0) if on_track else (255, 0, 0)
    pygame.draw.circle(screen, car_color, (car_x, car_y), 10)
    if not on_track:
        car.x, car.y, car.theta, car.v = -100, -60, 0, 0.0

    pygame.display.flip() 

pygame.quit()