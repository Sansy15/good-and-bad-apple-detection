# images_to_video.py
import argparse
import glob
import os
import cv2

def images_to_video(image_folder, output_path, fps=5, resize_to=None):
    # gather image files (sorted)
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    images = []
    for p in patterns:
        images.extend(sorted(glob.glob(os.path.join(image_folder, p))))
    if not images:
        raise RuntimeError("No images found in folder: " + image_folder)

    # read first image to get size (or use resize_to)
    first = cv2.imread(images[0])
    if first is None:
        raise RuntimeError("Could not read the first image.")
    if resize_to:
        width, height = resize_to
    else:
        height, width = first.shape[:2]

    # ensure output dir exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for " + output_path)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print("Warning: skipping", img_path)
            continue
        if resize_to:
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    print("Video creation completed:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--resize", nargs=2, type=int, help="width height", default=None)
    args = parser.parse_args()
    resize_to = (args.resize[0], args.resize[1]) if args.resize else None
    images_to_video(args.image_folder, args.output, fps=args.fps, resize_to=resize_to)
