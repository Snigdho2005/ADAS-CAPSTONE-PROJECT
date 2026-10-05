"""
Batch execution of Full Level 2 ADAS Perception Engine on clips 2, 3, 4, 5 (Untitled5), 6, 10
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from adas_level2_full import ADASLevel2Engine

clips = ["2.mov", "3.mov", "4.mov", "Untitled5.mov", "6.mov", "10.mov"]
weights = "runs/detect/train-2/weights/best.pt"

engine = ADASLevel2Engine(weights_path=weights)
out_dir = Path("runs/detect/full_level2_clips")
out_dir.mkdir(parents=True, exist_ok=True)

for clip in clips:
    p = Path(clip)
    if p.exists():
        out_path = out_dir / f"full_l2_{p.stem}.mp4"
        print(f"\n==========================================")
        print(f" Full Level 2 ADAS Processing: {clip} -> {out_path}")
        print(f"==========================================")
        engine.process_video(str(p), str(out_path), conf_thresh=0.35)
    else:
        print(f"Clip not found: {clip}")

print("\nAll requested Full Level 2 ADAS clips processed successfully!")
