import os
from PIL import Image, ImageSequence
import numpy as np

def remove_black_lines_from_gif(input_path, threshold=40):
    gif = Image.open(input_path)
    frames = []

    for frame in ImageSequence.Iterator(gif):
        rgba = frame.convert('RGBA')
        data = np.array(rgba)

        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        mask_black = (r < threshold) & (g < threshold) & (b < threshold) & (a > 0)
        data[mask_black] = [255, 255, 255, 0]

        cleaned_frame = Image.fromarray(data, 'RGBA')
        frames.append(cleaned_frame)

    # Save back to the same file, replacing it
    frames[0].save(
        input_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        transparency=0,
        disposal=2
    )

def clean_all_gifs_in_directory(directory="."):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".gif"):
            input_path = os.path.join(directory, filename)
            try:
                print(f"🛠️ Processing: {filename}")
                remove_black_lines_from_gif(input_path)
                print(f"✅ Cleaned and replaced {filename}\n")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}\n")

if __name__ == "__main__":
    clean_all_gifs_in_directory()