"""
Render a recorded lap (record_lap.py) to a broadcast-style MP4, a looping GIF
and a poster PNG.

    python export_video.py                                # suzuka_lap.npz -> assets/suzuka_lap.{mp4,gif,png}
    python export_video.py spa_lap.npz --name spa_lap
    python export_video.py --gif-start 20 --gif-length 10 # choose the GIF window yourself

Frames are piped straight into ffmpeg (the binary bundled with imageio-ffmpeg).
The GIF is cut from the finished MP4 with a two-pass palette, which gives far
cleaner colour than a fixed-palette GIF encoder.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # render headless

from lap_render import LapData, LapRenderer, surface_to_array


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def busiest_window(lap, length):
    """Start time of the `length`-second window with the most cornering (mean |lateral g|)."""
    if lap.duration <= length + 4:
        return 0.0
    per_step = np.abs(lap.g_lat)
    n = int(length / lap.dt)
    sums = np.convolve(per_step, np.ones(n), mode="valid")
    first = int(3.0 / lap.dt)                      # skip the standing start and title card
    last = len(sums) - int(2.0 / lap.dt)
    return float(np.argmax(sums[first:last]) + first) * lap.dt


def render_mp4(renderer, path, fps, tail, crf):
    lap = renderer.lap
    w, h = renderer.out_size
    n_frames = int((lap.duration + tail) * fps) + 1
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(n_frames):
            proc.stdin.write(surface_to_array(renderer.render(f / fps)).tobytes())
            if f % (fps * 5) == 0:
                print(f"\r  {f / fps:5.1f} / {n_frames / fps:.1f} s", end="", flush=True)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        sys.exit("ffmpeg failed while encoding the MP4")
    print(f"\r  wrote {path} ({n_frames} frames, {os.path.getsize(path) / 1e6:.1f} MB)")


def render_gif(mp4_path, gif_path, start, length, fps, width):
    graph = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=max_colors=256:stats_mode=diff[p];"
             "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
           "-i", mp4_path, "-vf", graph, "-loop", "0", gif_path]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("ffmpeg failed while making the GIF")
    print(f"  wrote {gif_path} ({start:.1f}-{start + length:.1f} s, {os.path.getsize(gif_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Render a recorded lap to MP4, GIF and a poster PNG")
    parser.add_argument("npz", nargs="?", default="suzuka_lap.npz", help="file written by record_lap.py")
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument("--name", default=None, help="output file stem (default: the .npz name)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--supersample", type=int, default=2, help="render this many times larger, then shrink")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, bigger)")
    parser.add_argument("--tail", type=float, default=3.0, help="seconds to hold the finish screen")
    parser.add_argument("--gif-start", type=float, default=None,
                        help="GIF window start in seconds (default: the most corner-heavy window)")
    parser.add_argument("--gif-length", type=float, default=10.0)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--skip-mp4", action="store_true", help="reuse an existing MP4 (GIF/poster only)")
    args = parser.parse_args()

    lap = LapData(args.npz)
    stem = args.name or os.path.splitext(os.path.basename(args.npz))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    mp4 = os.path.join(args.out_dir, f"{stem}.mp4")
    gif = os.path.join(args.out_dir, f"{stem}.gif")
    poster = os.path.join(args.out_dir, f"{stem}.png")

    renderer = LapRenderer(lap, args.width, args.height, supersample=args.supersample)
    print(f"{lap.track_name}: lap {lap.lap_time:.3f} s, model {lap.model_name or 'unknown'}")

    # Poster: the finish screen, with the whole lap drawn on the map and traces
    from PIL import Image
    Image.fromarray(surface_to_array(renderer.render(lap.duration + 1.0))).save(poster)
    print(f"  wrote {poster}")

    if not args.skip_mp4:
        render_mp4(renderer, mp4, args.fps, args.tail, args.crf)
    elif not os.path.exists(mp4):
        sys.exit(f"--skip-mp4 given but {mp4} does not exist")

    start = args.gif_start if args.gif_start is not None else busiest_window(lap, args.gif_length)
    render_gif(mp4, gif, start, args.gif_length, args.gif_fps, args.gif_width)


if __name__ == "__main__":
    main()
