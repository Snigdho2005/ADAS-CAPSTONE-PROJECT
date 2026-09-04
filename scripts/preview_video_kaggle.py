# Run this in a Kaggle notebook cell AFTER infer_phase1_visual.py finishes,
# to preview the annotated video directly in the notebook — no need to
# save a version and wait for the Output tab.
#
# Kaggle's IPython.display.Video can be finicky with mp4v-encoded files
# (the fourcc infer_phase1_visual.py uses), so this re-encodes to H.264
# first via ffmpeg, which Kaggle's environment ships with by default.

import subprocess
from pathlib import Path
from IPython.display import Video

SOURCE_VIDEO = "/kaggle/working/output_annotated.mp4"
PREVIEW_VIDEO = "/kaggle/working/output_preview_h264.mp4"

subprocess.run([
    "ffmpeg", "-y", "-i", SOURCE_VIDEO,
    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "23", PREVIEW_VIDEO,
], check=True, capture_output=True)

print(f"Re-encoded {Path(SOURCE_VIDEO).stat().st_size / 1e6:.1f} MB -> "
      f"{Path(PREVIEW_VIDEO).stat().st_size / 1e6:.1f} MB")

Video(PREVIEW_VIDEO, embed=True, width=800)
