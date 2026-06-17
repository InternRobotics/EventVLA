import cv2
import numpy as np
import os
import shutil
import subprocess
import tempfile
import pickle
import pdb


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"

    temp_dir = tempfile.mkdtemp(prefix="robotwin_mem_video_")
    temp_video_path = os.path.join(temp_dir, os.path.basename(out_path) or "episode.mp4")
    try:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{W}x{H}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                temp_video_path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            ffmpeg.stdin.write(imgs.tobytes())
            ffmpeg.stdin.close()
        except BrokenPipeError:
            pass

        stderr = ffmpeg.stderr.read().decode(errors="replace") if ffmpeg.stderr else ""
        if ffmpeg.wait() != 0:
            raise IOError(
                "Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported."
                f"\nffmpeg stderr:\n{stderr.strip()}"
            )

        shutil.copyfile(temp_video_path, out_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )
