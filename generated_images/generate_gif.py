import imageio
import os

# File name template
file_template = "epoch_{}.png"

# Specify the range of epochs
start_epoch = 1
end_epoch = 50

# Collect image paths in the correct order
image_paths = [file_template.format(i) for i in range(start_epoch, end_epoch + 1)]

# Verify that all images exist
for path in image_paths:
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")

# Output GIF name
output_gif = "epoch_animation.gif"

# Create the GIF
with imageio.get_writer(output_gif, mode="I", duration=1.0) as writer:  # duration is in seconds
    for path in image_paths:
        if os.path.exists(path):  # Only add existing images
            image = imageio.imread(path)
            writer.append_data(image)

print(f"GIF saved as {output_gif}")
