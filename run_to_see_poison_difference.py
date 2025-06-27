from pathlib import Path
import cv2
import os
import numpy as np

# Root directory with all folders
script_path = Path(__file__).resolve()
project_root = script_path.parent
root_dir = Path(project_root / 'poisoned_folder/')

# Get folder pairs
folder_names = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
paired_folders = {}


for name in folder_names:
    key = name.split('_')[0]  # 'dog' or 'chair'
    paired_folders.setdefault(key, {})
    if 'chosen' in name:
        paired_folders[key]['chosen'] = os.path.join(root_dir, name)
    elif 'p' in name:
        paired_folders[key]['p'] = os.path.join(root_dir, name)
print(paired_folders)

# Output folder
os.makedirs('poison_difference_compare', exist_ok=True)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8  # will look normal after scaling
font_color = (255, 255, 255)
font_thickness = 1


# Draw text on the collage
def draw_label(image, text, top_left):
    cv2.putText(
        image, text, top_left, font, font_scale, font_color,
        font_thickness, cv2.LINE_AA
    )
    return image

for label, folders in paired_folders.items():
    chosen_dir = folders.get('chosen')
    p_dir = folders.get('p')
    if not chosen_dir or not p_dir:
        continue

    # Match files based on substrings
    chosen_files = os.listdir(chosen_dir)
    p_files = os.listdir(p_dir)

    for c_file in chosen_files:
        for p_file in p_files:
            # Check if they share a substring (adjust logic if needed)
            shared_substr = os.path.splitext(c_file)[0].split('.')[0]
            if shared_substr in p_file:
                img_chosen = cv2.imread(os.path.join(chosen_dir, c_file))
                img_p = cv2.imread(os.path.join(p_dir, p_file))

                if img_chosen is None or img_p is None:
                    continue

                # Resize to match
                if img_chosen.shape[0] != 32:
                    img_chosen = cv2.resize(img_chosen, (720, 720), interpolation=cv2.INTER_NEAREST)
                    img_p = cv2.resize(img_p, (720, 720), interpolation=cv2.INTER_NEAREST)

                # Difference
                diff = cv2.absdiff(img_chosen, img_p)

                # Heatmap
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                heatmap = cv2.applyColorMap(cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)

                # Stack side-by-side
                collage = np.hstack((img_chosen, img_p, diff, heatmap))

                if img_chosen.shape[1] == 32:
                    scale_factor = 8
                    collage = cv2.resize(
                        collage,
                        (collage.shape[1] * scale_factor, collage.shape[0] * scale_factor),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    scale_factor = 1

                h = img_chosen.shape[0] * scale_factor
                w = img_chosen.shape[1] * scale_factor

                # Add bottom bar
                label_height = 25
                bar = np.zeros((label_height, collage.shape[1], 3), dtype=np.uint8)
                collage = np.vstack((collage, bar))

                cv2.putText(collage, c_file.split('.')[0], (5, h + 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(collage, 'poisoned', (w + 5, h + 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(collage, 'difference', (2 * w + 5, h + 20), font, font_scale, font_color, font_thickness,
                            cv2.LINE_AA)
                cv2.putText(collage, 'heatmap difference', (3 * w + 5, h + 20), font, font_scale, font_color,
                            font_thickness, cv2.LINE_AA)

                # Save
                save_path = os.path.join('poison_difference_compare', f'{shared_substr}_compare.png')

                cv2.imwrite(save_path, collage)

                print(f'Saved: {save_path}')
                break
