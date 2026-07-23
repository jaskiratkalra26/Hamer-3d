"""
Example integration showing how to use HandTracker3D in external Python projects.
"""
import cv2
from hand_tracker import HandTracker3D

def main():
    print("Initializing HandTracker3D pipeline...")
    # 1. Initialize tracker once
    tracker = HandTracker3D(body_detector='regnety', scale_inference=0.5, det_stride=5)

    # 2. Read frame from video or camera stream
    video_path = "../test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video at {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    # 3. Process frame in memory (returns 3D mesh vertices, 3D translations, and handedness)
    results = tracker.process_frame(frame)

    print("\n==========================================")
    print(f"Detected {len(results['hands'])} hand(s):")
    for i, hand in enumerate(results['hands']):
        handedness = "Right Hand" if hand['is_right'] == 1 else "Left Hand"
        print(f"  Hand #{i+1} ({handedness}):")
        print(f"    - 3D Mesh Vertices shape: {hand['verts_3d'].shape}") # (778, 3) 3D points
        print(f"    - 3D Camera Translation:  {hand['cam_t']}")        # [tx, ty, tz]
    print("==========================================\n")

    # 4. Optional: Render 3D wireframe overlay frame
    overlay_frame = tracker.render_overlay(frame, results)
    cv2.imwrite("output_sample_frame.jpg", overlay_frame)
    print("Saved rendered overlay to output_sample_frame.jpg")

    cap.release()

if __name__ == '__main__':
    main()
