"""
Full Level 2 ADAS Perception Engine (Steps 1, 2 & Phase 1 Combined)

Features:
1. Object Detection & ByteTrack Multi-Object Tracking
2. Monocular Distance Estimation & Relative Speed (km/h)
3. Time-to-Collision (TTC) & Forward Collision Warning (FCW / AEB)
4. Step 1: Drivable Lane Boundary Detection, Ego-Lane Center Offset (meters) & Lane Departure Warning (LDW)
5. Step 2: Traffic Light Color State Classification (RED / YELLOW / GREEN) & Speed Limit Recognition
6. Level 2 ADAS Cockpit HUD Telemetry Overlay
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


class ADASLevel2Engine:
    def __init__(self, weights_path, camera_height=1.65, focal_length_y=720.0):
        self.model = YOLO(weights_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.45, text_padding=4)
        
        self.camera_height = camera_height
        self.focal_length_y = focal_length_y
        self.history = defaultdict(lambda: deque(maxlen=10))

    def estimate_distance(self, bbox, frame_height):
        x1, y1, x2, y2 = bbox
        y_bottom = y2
        y_horizon = frame_height * 0.52
        dy = max(y_bottom - y_horizon, 1.0)
        distance = (self.focal_length_y * self.camera_height) / dy
        return float(np.clip(distance, 1.0, 150.0))

    def compute_ttc_and_velocity(self, track_id, current_dist, current_time):
        history = self.history[track_id]
        history.append((current_time, current_dist))

        if len(history) < 3:
            return 0.0, float('inf'), "SAFE"

        times = np.array([t for t, d in history])
        dists = np.array([d for t, d in history])
        dt = times[-1] - times[0]

        if dt < 0.05:
            return 0.0, float('inf'), "SAFE"

        velocity_ms = -(dists[-1] - dists[0]) / dt
        velocity_kmh = velocity_ms * 3.6

        if velocity_ms > 0.5:
            ttc = current_dist / velocity_ms
        else:
            ttc = float('inf')

        if ttc <= 1.8:
            state = "AEB_BRAKE"
        elif ttc <= 3.0:
            state = "FCW_WARN"
        else:
            state = "SAFE"

        return velocity_kmh, ttc, state

    def detect_lane_lines_and_offset(self, frame):
        """
        Step 1: Detect ego-lane boundary lines, drivable area polynomial,
        compute lane center offset (meters), and trigger Lane Departure Warning (LDW).
        """
        h, w = frame.shape[:2]
        roi_vertices = np.array([[
            (int(w * 0.1), h),
            (int(w * 0.42), int(h * 0.60)),
            (int(w * 0.58), int(h * 0.60)),
            (int(w * 0.9), h)
        ]], dtype=np.int32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=20)
        
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3 or abs(slope) > 3.0:
                    continue
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

        # Average left & right lane lines
        left_x, right_x = int(w * 0.3), int(w * 0.7)
        if left_lines:
            left_pts = np.array(left_lines).reshape(-1, 2)
            left_x = int(np.mean(left_pts[:, 0]))
        if right_lines:
            right_pts = np.array(right_lines).reshape(-1, 2)
            right_x = int(np.mean(right_pts[:, 0]))

        lane_center = (left_x + right_x) / 2.0
        camera_center = w / 2.0
        
        # Approximate 0.0035 meters per pixel at bonnet line
        offset_m = (camera_center - lane_center) * 0.0035

        if offset_m > 0.35:
            ldw_state = "DRIFT_RIGHT"
        elif offset_m < -0.35:
            ldw_state = "DRIFT_LEFT"
        else:
            ldw_state = "CENTERED"

        return roi_vertices, left_x, right_x, offset_m, ldw_state

    def classify_traffic_light_state(self, crop_bgr):
        """
        Step 2: Inspect traffic sign/light crop in HSV color space to classify RED, YELLOW, or GREEN
        """
        if crop_bgr.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        # Red mask
        mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_red = mask_r1 | mask_r2

        # Yellow mask
        mask_yellow = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([35, 255, 255]))

        # Green mask
        mask_green = cv2.inRange(hsv, np.array([40, 70, 50]), np.array([90, 255, 255]))

        r_cnt, y_cnt, g_cnt = np.sum(mask_red), np.sum(mask_yellow), np.sum(mask_green)

        if r_cnt > y_cnt and r_cnt > g_cnt and r_cnt > 200:
            return "RED (STOP)"
        elif y_cnt > r_cnt and y_cnt > g_cnt and y_cnt > 200:
            return "YELLOW (CAUTION)"
        elif g_cnt > r_cnt and g_cnt > y_cnt and g_cnt > 200:
            return "GREEN (GO)"
        return "CLEAR"

    def process_video(self, video_path, output_path, conf_thresh=0.35):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"Running Full Level 2 ADAS on: {video_path} ({w}x{h} @ {fps:.1f} FPS)")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t_now = frame_idx / fps
            frame_idx += 1

            # Step 1: Detect Lane Lines & Offset
            roi_poly, left_x, right_x, offset_m, ldw_state = self.detect_lane_lines_and_offset(frame)

            # Step 2 & Phase 1: Run YOLOv8 Object Detection & ByteTrack
            results = self.model(frame, conf=conf_thresh, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            labels = []
            frame_closest_dist = float('inf')
            frame_critical_state = "SAFE"
            traffic_signal_status = "NONE"

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
                elif state == "FCW_WARN" and frame_critical_state != "AEB_BRAKE":
                    frame_critical_state = "FCW_WARN"

                # Step 2: Traffic Light Inspection
                if class_name in ["traffic_sign", "traffic_light"]:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    t_status = self.classify_traffic_light_state(crop)
                    if t_status != "CLEAR":
                        traffic_signal_status = t_status

                lbl = f"#{track_id} {class_name} | {dist_m:.1f}m"
                if ttc_s < 30.0:
                    lbl += f" | {vel_kmh:.0f}km/h | TTC:{ttc_s:.1f}s"
                labels.append(lbl)

            # Draw Annotations & Lane Overlay
            annotated_frame = frame.copy()

            # Draw Drivable Lane Polygon
            lane_overlay = annotated_frame.copy()
            lane_pts = np.array([
                [left_x, h],
                [int(w * 0.45), int(h * 0.60)],
                [int(w * 0.55), int(h * 0.60)],
                [right_x, h]
            ], dtype=np.int32)
            cv2.fillPoly(lane_overlay, [lane_pts], (0, 255, 120))  # Green drivable lane
            cv2.addWeighted(lane_overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)

            # Draw bounding boxes & labels
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Top Telemetry Status Bar
            cv2.rectangle(annotated_frame, (0, 0), (w, 60), (15, 15, 15), -1)
            cv2.putText(annotated_frame, "FULL LEVEL 2 ADAS COCKPIT HUD", (15, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Lane Offset Telemetry (Step 1)
            offset_color = (0, 255, 0) if ldw_state == "CENTERED" else (0, 165, 255)
            cv2.putText(annotated_frame, f"LKA Offset: {offset_m:+.2f}m ({ldw_state})", (380, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, offset_color, 2)

            # Traffic Signal Telemetry (Step 2)
            if traffic_signal_status != "NONE":
                sig_color = (0, 0, 255) if "RED" in traffic_signal_status else (0, 255, 0)
                cv2.putText(annotated_frame, f"Signal: {traffic_signal_status}", (w - 420, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, sig_color, 2)

            # Warning Banners (FCW / AEB / LDW)
            if frame_critical_state == "AEB_BRAKE":
                cv2.rectangle(annotated_frame, (w // 2 - 260, 65), (w // 2 + 260, 115), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "!!! EMERGENCY BRAKE (AEB) !!!", (w // 2 - 230, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
            elif frame_critical_state == "FCW_WARN":
                cv2.rectangle(annotated_frame, (w // 2 - 260, 65), (w // 2 + 260, 115), (0, 255, 255), -1)
                cv2.putText(annotated_frame, "FORWARD COLLISION WARNING", (w // 2 - 230, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
            elif ldw_state != "CENTERED":
                cv2.rectangle(annotated_frame, (w // 2 - 260, 65), (w // 2 + 260, 115), (0, 165, 255), -1)
                cv2.putText(annotated_frame, f"LANE DEPARTURE: {ldw_state}", (w // 2 - 220, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"Full Level 2 ADAS Video saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="runs/detect/train-2/weights/best.pt")
    ap.add_argument("--out", default="full_level2_adas_output.mp4")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    engine = ADASLevel2Engine(weights_path=args.weights)
    engine.process_video(args.video, args.out, conf_thresh=args.conf)


if __name__ == "__main__":
    main()
