"""
Phase 1 Level 2 ADAS Perception & Collision Warning Engine
Combines YOLOv8 detection, ByteTrack Multi-Object Tracking, Monocular Distance Estimation,
Relative Velocity, and Time-To-Collision (TTC) for Forward Collision Warning (FCW) & AEB.
"""
import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv


class ADASPerceptionEngine:
    def __init__(self, weights_path, camera_height=1.65, focal_length_y=720.0):
        """
        camera_height: Height of dashcam above ground in meters (default 1.65m)
        focal_length_y: Vertical focal length of camera in pixels
        """
        self.model = YOLO(weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        
        self.camera_height = camera_height
        self.focal_length_y = focal_length_y

        # Track history for velocity & TTC calculation: track_id -> deque of (timestamp, distance_m)
        self.history = defaultdict(lambda: deque(maxlen=10))

    def estimate_distance(self, bbox, frame_height):
        """
        Estimate distance to object base in meters using monocular ground plane geometry:
        d = (f_y * h_cam) / (y_bottom - y_horizon)
        """
        x1, y1, x2, y2 = bbox
        y_bottom = y2
        y_horizon = frame_height * 0.52  # Approximate camera pitch horizon line
        
        dy = max(y_bottom - y_horizon, 1.0)
        distance = (self.focal_length_y * self.camera_height) / dy
        return float(np.clip(distance, 1.0, 150.0))

    def compute_ttc_and_velocity(self, track_id, current_dist, current_time):
        """
        Compute relative velocity (m/s and km/h) and Time-To-Collision (TTC in seconds)
        """
        history = self.history[track_id]
        history.append((current_time, current_dist))

        if len(history) < 3:
            return 0.0, float('inf'), "SAFE"

        # Fit linear velocity over recent history
        times = np.array([t for t, d in history])
        dists = np.array([d for t, d in history])
        dt = times[-1] - times[0]

        if dt < 0.05:
            return 0.0, float('inf'), "SAFE"

        # Negative slope means object is closing in (approaching ego vehicle)
        velocity_ms = -(dists[-1] - dists[0]) / dt
        velocity_kmh = velocity_ms * 3.6

        if velocity_ms > 0.5:  # Approaching target
            ttc = current_dist / velocity_ms
        else:
            ttc = float('inf')

        # Safety State Machine
        if ttc <= 1.8:
            state = "AEB_BRAKE"
        elif ttc <= 3.0:
            state = "FCW_WARN"
        else:
            state = "SAFE"

        return velocity_kmh, ttc, state

    def process_video(self, video_path, output_path, conf_thresh=0.30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"Processing Level 2 ADAS on: {video_path} ({w}x{h} @ {fps:.1f} FPS)")
        
        frame_idx = 0
        closest_distance_global = float('inf')
        global_alert_state = "SAFE"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_now = frame_idx / fps
            frame_idx += 1

            # Run YOLOv8 detection
            results = self.model(frame, conf=conf_thresh, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Update ByteTrack tracker
            detections = self.tracker.update_with_detections(detections)

            labels = []
            frame_closest_dist = float('inf')
            frame_critical_state = "SAFE"

            # Color masks per detection
            custom_colors = []

            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                class_id = detections.class_id[i]
                class_name = self.model.names[class_id]
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else i

                dist_m = self.estimate_distance(bbox, h)
                vel_kmh, ttc_s, state = self.compute_ttc_and_velocity(track_id, dist_m, t_now)

                if dist_m < frame_closest_dist:
                    frame_closest_dist = dist_m

                if state == "AEB_BRAKE":
                    frame_critical_state = "AEB_BRAKE"
                    color = (0, 0, 255)  # Red
                elif state == "FCW_WARN":
                    if frame_critical_state != "AEB_BRAKE":
                        frame_critical_state = "FCW_WARN"
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green

                # Build Telemetry Overlay Text
                if ttc_s < 30.0:
                    lbl = f"#{track_id} {class_name} | {dist_m:.1f}m | {vel_kmh:.0f}km/h | TTC:{ttc_s:.1f}s"
                else:
                    lbl = f"#{track_id} {class_name} | {dist_m:.1f}m"
                labels.append(lbl)

            # Draw Annotations
            annotated_frame = frame.copy()
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Draw ADAS Level 2 Heads-Up Display (HUD) Status Banner
            cv2.rectangle(annotated_frame, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(annotated_frame, "ADAS LEVEL 2 PERCEPTION ENGINE", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            closest_str = f"{frame_closest_dist:.1f}m" if frame_closest_dist < 100 else "N/A"
            cv2.putText(annotated_frame, f"Closest Obj: {closest_str}", (w - 420, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw Warning Banner if FCW / AEB active
            if frame_critical_state == "AEB_BRAKE":
                cv2.rectangle(annotated_frame, (w // 2 - 250, 55), (w // 2 + 250, 105), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "!!! EMERGENCY BRAKE (AEB) !!!", (w // 2 - 220, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            elif frame_critical_state == "FCW_WARN":
                cv2.rectangle(annotated_frame, (w // 2 - 250, 55), (w // 2 + 250, 105), (0, 255, 255), -1)
                cv2.putText(annotated_frame, "FORWARD COLLISION WARNING", (w // 2 - 220, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"Level 2 ADAS processing finished. Video saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument("--weights", default="runs/detect/train-2/weights/best.pt")
    ap.add_argument("--out", default="adas_l2_output.mp4")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    engine = ADASPerceptionEngine(weights_path=args.weights)
    engine.process_video(args.video, args.out, conf_thresh=args.conf)


if __name__ == "__main__":
    main()
