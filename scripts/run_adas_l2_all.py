"""
Run Phase 1 Level 2 ADAS Perception Engine across all remaining video clips.
"""
import sys
from pathlib import Path

# Add scripts directory to path if needed
sys.path.append(str(Path(__file__).parent))
from adas_level2_phase1 import ADASPerceptionEngine

clips = ["3.mov", "4.mov", "6.mov", "7.mov", "8.mov", "9.mov", "10.mov", "Untitled5.mov"]
weights = "runs/detect/train-2/weights/best.pt"

engine = ADASPerceptionEngine(weights_path=weights)
out_dir = Path("runs/detect/adas_level2_clips")
out_dir.mkdir(parents=True, exist_ok=True)

for clip in clips:
    p = Path(clip)
    if p.exists():
        out_path = out_dir / f"adas_l2_{p.stem}.mp4"
        print(f"\n==========================================")
        print(f" Level 2 ADAS Processing: {clip} -> {out_path}")
        print(f"==========================================")
        engine.process_video(str(p), str(out_path), conf_thresh=0.35)
    else:
        print(f"Clip not found: {clip}")

print("\nAll remaining Level 2 ADAS clips processed successfully!")
