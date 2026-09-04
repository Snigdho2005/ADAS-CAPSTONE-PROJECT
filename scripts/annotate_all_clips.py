"""
Annotate all remaining dashcam .mov video clips using fine-tuned YOLOv8 model.
"""
from pathlib import Path
from ultralytics import YOLO

clips = ["3.mov", "4.mov", "6.mov", "7.mov", "8.mov", "9.mov", "10.mov", "Untitled5.mov"]
model = YOLO("runs/detect/train-2/weights/best.pt")

for clip in clips:
    p = Path(clip)
    if p.exists():
        print(f"\n==========================================")
        print(f" Processing video clip: {clip}")
        print(f"==========================================")
        results = model.predict(
            source=str(p),
            conf=0.35,
            save=True,
            project="runs/detect",
            name="annotated_clips",
            exist_ok=True,
            stream=True
        )
        for _ in results:
            pass
    else:
        print(f"Clip not found: {clip}")

print("\nAll remaining clips processed successfully!")
