# extract_frames.py
import cv2
import os
import argparse

def extract_frames(video_path, out_dir, skip=1, max_frames=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()
    extract_frames(args.video, args.out_dir, skip=args.skip, max_frames=args.max_frames)
