"""Lightweight manifest/asset validation checks; never imports CLIP/PyTorch."""
import argparse
import csv
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from local_visual_embeddings_round4 import read_manifest, run, validate_asset, video_frames


def main():
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix="clip_manifest_check_") as directory:
        root = Path(directory)
        manifest = root / "assets.csv"
        base = {"split": "train", "row_index": 0, "path": "missing.jpg", "kind": "image"}
        def write(rows, fields=("split", "row_index", "path", "kind")):
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        write([base])
        row = read_manifest(manifest, {"train": 2, "test": 1})[0]
        assert validate_asset(row, time.time()+10)["error"] == "missing_local_file"
        (root / "corrupt.jpg").write_text("not an image")
        write([{**base, "path": "corrupt.jpg"}])
        row = read_manifest(manifest, {"train": 2, "test": 1})[0]
        assert not validate_asset(row, time.time()+10)["valid"]
        for rows, fields in (([{"split": "train"}], ("split",)),
                             ([{**base, "row_index": -1}], None),
                             ([{**base, "kind": "audio"}], None),
                             ([{**base, "path": "https://example.com/photo.jpg"}], None),
                             ([base, base], None)):
            write(rows, fields or ("split", "row_index", "path", "kind"))
            try:
                read_manifest(manifest, {"train": 2, "test": 1})
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed manifest accepted")
        write([base, {**base, "row_index": 1, "path": "corrupt.jpg"}])
        args = argparse.Namespace(manifest=str(manifest), output=str(root / "output"),
                                  deadline=str(time.time()+30), train_rows=2, test_rows=1,
                                  max_assets=512, validate_only=False, model_path=None)
        result = run(args)
        assert result["valid_assets"] == 0 and not result["inference_executed"] and not result["model_downloaded"]
        Image.new("RGB", (3, 4), "red").save(root / "valid.png")
        write([{**base, "path": "valid.png"}])
        args.validate_only = True
        result = run(args)
        assert result["valid_assets"] == 1 and not result["inference_executed"]
        video = root / "qa.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=64x48:d=2:r=5", "-threads", "1", "-pix_fmt", "yuv420p", str(video)],
                       check=True, timeout=15)
        metadata = validate_asset({**base, "path": str(video), "kind": "video"}, time.time()+30)
        assert metadata["valid"] and metadata["frame_timestamps"] == [.2, 1., 1.8]
        frames = video_frames(metadata, root, time.time()+30)
        assert len(frames) == 3 and frames == video_frames(metadata, root, time.time()+30)
        assert "torch" not in sys.modules and "clip" not in sys.modules
    print("10 manifest/asset/frame checks passed; no model inference or downloads.")


if __name__ == "__main__":
    main()
