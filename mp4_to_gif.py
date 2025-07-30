import os
import subprocess
from PIL import Image
import numpy as np

def extract_frames_with_ffmpeg(mp4_path, output_folder, fps=10):
    os.makedirs(output_folder, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", mp4_path,
        "-vf", f"fps={fps}",
        os.path.join(output_folder, "frame_%04d.png")
    ]
    subprocess.run(cmd, check=True)
    print(f"🎞️ Extracted frames from {mp4_path}")

def make_gif_with_transparency(frames_folder, output_gif):
    frames = []
    for filename in sorted(os.listdir(frames_folder)):
        if filename.endswith(".png"):
            frame_path = os.path.join(frames_folder, filename)
            img = Image.open(frame_path).convert("RGBA")
            data = np.array(img)
            r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
            
            # Make white (or near-white) transparent
            white_mask = (r > 240) & (g > 240) & (b > 240)
            data[..., 3][white_mask] = 0

            new_img = Image.fromarray(data, mode="RGBA")
            frames.append(new_img)

    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=100,  # 100ms per frame = 10 FPS
            disposal=2
        )
        print(f"✅ Created GIF: {output_gif}")

def process_all_mp4s_in_folder(folder="."):
    for file in os.listdir(folder):
        if file.lower().endswith(".mp4"):
            mp4_path = os.path.join(folder, file)
            temp_dir = os.path.join(folder, "temp_frames")
            gif_name = os.path.splitext(file)[0] + ".gif"
            gif_path = os.path.join(folder, gif_name)

            extract_frames_with_ffmpeg(mp4_path, temp_dir)
            make_gif_with_transparency(temp_dir, gif_path)

            # Clean up
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

if __name__ == "__main__":
    process_all_mp4s_in_folder()
