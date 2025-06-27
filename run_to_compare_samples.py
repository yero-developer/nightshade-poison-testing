import os
from PIL import Image
import cv2
import numpy as np


def add_label_below(img, label, font_scale=0.4, padding=10):
    """Add a label under an OpenCV image"""
    h, w, _ = img.shape
    label_height = 20 + padding
    canvas = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
    canvas[:h] = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
    text_x = max((w - text_size[0]) // 2, 0)
    text_y = h + padding + 10

    cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

folder_path = os.getcwd() + "/samples_generated"

# Iterate over files in directory
for path, folders, files in os.walk(folder_path):
    full_images_list = []
    full_name_list = []
    for file in sorted(files):
        file_path = path + '/' + file

        pil_image = Image.open(file_path).resize((128, 128), Image.NEAREST)
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        full_images_list.append(img_cv)
        full_name_list.append(file.split('.')[0])

    ordered_images_list = []
    ordered_names_list = []

    for i in range(0, len(full_images_list), 2):
        ordered_images_list.append([full_images_list[i], full_images_list[i + 1]])
        ordered_names_list.append([full_name_list[i], full_name_list[i + 1]])
        i = i + 2

    if ordered_images_list == []:
        continue

    # === Config ===
    image_size = (128, 128)
    pairs_per_row = 2  # 2 pairs = 4 images per row

    # Group into rows of 4 images (2 pairs)
    rows = []
    row_images = []

    for i in range(len(ordered_images_list)):
        #print(i)
        # Get a pair (2 images)
        img1, name1 = ordered_images_list[i][0], ordered_names_list[i][0]
        img2, name2 = ordered_images_list[i][1], ordered_names_list[i][1]

        pair_list = [[img1, name1], [img2, name2]]

        for pair in pair_list:
            labeled = add_label_below(pair[0], pair[1])
            row_images.append(labeled)

        # After 2 pairs (4 images), create a row
        if len(row_images) == 4:
            row = np.hstack(row_images)
            rows.append(row)
            row_images = []

    # If last row has < 4 images, pad it
    if row_images:
        h, w = row_images[0].shape[:2]
        while len(row_images) < 4:
            row_images.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
        row = np.hstack(row_images)
        rows.append(row)

    # Stack all rows into final image
    final_grid = np.vstack(rows)

    folder_placement = f'samples_generated_compared/'
    os.makedirs(folder_placement, exist_ok=True)

    cv2.imwrite(f"{folder_placement}/{ordered_names_list[0][0].split('_')[0]}_comparison.png", final_grid)


