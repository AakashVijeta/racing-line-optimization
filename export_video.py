import argparse
import numpy as np
import pygame
import imageio
from replay import ReplayViewer

def export():
    parser = argparse.ArgumentParser(description="Render a recorded lap to MP4 and a short GIF")
    parser.add_argument("npz", nargs="?", default="suzuka_lap.npz", help="file written by record_lap.py")
    parser.add_argument("--mp4", default="lap_full.mp4")
    parser.add_argument("--gif", default="lap_snippet.gif")
    parser.add_argument("--gif-start", type=float, default=20.0, help="GIF window start (s)")
    parser.add_argument("--gif-end", type=float, default=30.0, help="GIF window end (s)")
    args = parser.parse_args()

    pygame.init()
    viewer = ReplayViewer(npz_path=args.npz)
    
    # Render to an offscreen surface instead of setting up a display mode
    surface = pygame.Surface((viewer.width, viewer.height))
    
    fps_out = 60
    fps_gif = 20  # GIF captures every third frame
    
    total_time = (viewer.n_frames - 1) * viewer.dt
    total_out_frames = int(total_time * fps_out)
    
    mp4_writer = imageio.get_writer(args.mp4, fps=fps_out)
    gif_writer = imageio.get_writer(args.gif, fps=fps_gif)
    
    # Corner-heavy window for the GIF (e.g., 20s to 30s)
    gif_start_time = args.gif_start
    gif_end_time = args.gif_end
    
    print(f"Exporting {total_out_frames} frames to MP4 and GIF...")
    
    try:
        for f in range(total_out_frames):
            t = f / fps_out
            state = viewer.state_at(t)
            
            # state is (cx, cy, ctheta, cv, cthrottle, csteering, carc, idx_0)
            viewer.draw_frame_by_state(surface, *state[:7], trail_up_to_idx=state[7], lap_time=t)
            
            # pygame.surfarray.array3d returns shape (width, height, 3), imageio expects (height, width, 3)
            arr = pygame.surfarray.array3d(surface)
            arr = np.transpose(arr, (1, 0, 2))
            
            mp4_writer.append_data(arr)
            
            if gif_start_time <= t <= gif_end_time:
                if f % 3 == 0:
                    # Scale down the surface to 640x360
                    small_surf = pygame.transform.smoothscale(surface, (640, 360))
                    arr_small = pygame.surfarray.array3d(small_surf)
                    arr_small = np.transpose(arr_small, (1, 0, 2))
                    gif_writer.append_data(arr_small)
                
            if f % 100 == 0:
                print(f"Rendered frame {f}/{total_out_frames}")
    finally:
        mp4_writer.close()
        gif_writer.close()
        pygame.quit()
        print("Export complete!")

if __name__ == "__main__":
    export()
