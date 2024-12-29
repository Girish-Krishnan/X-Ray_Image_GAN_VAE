import imageio
import os
from PIL import Image

# File name templates
normal_template = "NORMAL_epoch_{}.png"
pneumonia_template = "PNEUMONIA_epoch_{}.png"

# Specify the range of epochs
start_epoch = 1
end_epoch = 50

# Output GIF name
output_gif = "epoch_animation.gif"

# Function to combine two images side by side
def combine_images(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Ensure both images have the same height (optional: resize if necessary)
    if img1.size[1] != img2.size[1]:
        img2 = img2.resize((int(img2.size[0] * img1.size[1] / img2.size[1]), img1.size[1]))
    
    # Combine images horizontally
    combined_width = img1.size[0] + img2.size[0]
    combined_image = Image.new("RGB", (combined_width, img1.size[1]))
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.size[0], 0))
    
    return combined_image

# Collect paths and validate images
frames = []
for epoch in range(start_epoch, end_epoch + 1):
    normal_path = normal_template.format(epoch)
    pneumonia_path = pneumonia_template.format(epoch)
    if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
        combined_image = combine_images(normal_path, pneumonia_path)
        frames.append(combined_image)
    else:
        print(f"Warning: Missing files for epoch {epoch}.")

# Create GIF
if frames:
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=0.5,  # Adjust duration for speed (0.5 seconds per frame)
        loop=0,        # Loop forever
    )
    print(f"GIF saved as {output_gif}")
else:
    print("No valid frames to create GIF.")
