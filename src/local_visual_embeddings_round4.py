"""Optional offline CLIP image/video features; no downloads, labels or training.

Embeddings are sparse by row_index. Missing items have coverage flags, never
fabricated zero vectors. Only official SHA-256 verified CLIP ViT-B/32 is loaded.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

MODEL_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
MODEL_URL = f"https://openaipublic.azureedge.net/clip/models/{MODEL_SHA256}/ViT-B-32.pt"
FRAME_FRACTIONS = (.1, .5, .9)


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Visual feature extraction deadline reached")


def sha256(path, deadline=float("inf")):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check_deadline(deadline)
            digest.update(block)
    return digest.hexdigest()


def parse_deadline(value):
    if value is None:
        return time.time() + 3600
    try:
        return float(value)
    except ValueError:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("ISO deadline must contain a timezone")
        return parsed.timestamp()


def read_manifest(path, row_counts):
    """Strict alignment/schema checks; allow several distinct assets per item."""
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "row_index", "path", "kind"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires split,row_index,path,kind")
        for line, row in enumerate(reader, 2):
            split, kind = row["split"].strip(), row["kind"].strip()
            if split not in row_counts or kind not in ("image", "video"):
                raise ValueError(f"Invalid split or kind at manifest line {line}")
            try:
                index = int(row["row_index"])
            except ValueError as error:
                raise ValueError(f"Invalid row_index at line {line}") from error
            if not 0 <= index < row_counts[split]:
                raise ValueError(f"row_index out of range at line {line}")
            raw_path = row["path"].strip()
            if not raw_path or "://" in raw_path:
                raise ValueError(f"Expected a local path at line {line}")
            asset_path = Path(raw_path)
            if not asset_path.is_absolute():
                asset_path = path.parent / asset_path
            asset_path = asset_path.resolve()
            key = (split, index, kind, str(asset_path))
            if key in seen:
                raise ValueError(f"Duplicate asset at line {line}")
            seen.add(key)
            rows.append({"split": split, "row_index": index, "kind": kind, "path": str(asset_path)})
    return rows


def validate_asset(record, deadline):
    check_deadline(deadline)
    path = Path(record["path"])
    result = {**record, "valid": False}
    if not path.is_file():
        return {**result, "error": "missing_local_file"}
    try:
        if record["kind"] == "image":
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                result.update(width=image.width, height=image.height, format=image.format)
        else:
            timeout = min(12., max(.1, deadline - time.time()))
            command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height:format=duration", "-of", "json", str(path)]
            metadata = json.loads(subprocess.run(command, capture_output=True, text=True,
                                                check=True, timeout=timeout).stdout)
            stream = metadata["streams"][0]
            duration = float(metadata["format"]["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid finite video duration")
            result.update(width=int(stream["width"]), height=int(stream["height"]), duration_seconds=duration,
                          frame_timestamps=[round(duration * fraction, 6) for fraction in FRAME_FRACTIONS])
        if result["width"] <= 0 or result["height"] <= 0:
            raise ValueError("Invalid image dimensions")
        result.update(valid=True, asset_sha256=sha256(path, deadline), bytes=path.stat().st_size)
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as error:
        result["error"] = type(error).__name__ + ": " + str(error)[:300]
    return result


def video_frames(record, output, deadline):
    from PIL import Image
    folder = output / "frames" / record["asset_sha256"]
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for number, timestamp in enumerate(record["frame_timestamps"]):
        check_deadline(deadline)
        path = folder / f"frame_{number}_{timestamp:.6f}.png"
        command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "2", "-ss", f"{timestamp:.6f}",
                   "-i", record["path"], "-frames:v", "1", "-vf",
                   "scale=512:512:force_original_aspect_ratio=decrease", "-threads", "2", "-y", str(path)]
        if not path.exists():
            subprocess.run(command, capture_output=True, check=True, timeout=min(30., max(.1, deadline-time.time())))
        with Image.open(path) as image:
            image.verify()
        frames.append({"path": str(path), "timestamp_seconds": timestamp, "sha256": sha256(path, deadline)})
    return frames


def save_coverage(output, row_counts, successes):
    with (output / "local_visual_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "row_index", "image_missing", "video_missing",
                                                    "image_assets_encoded", "video_assets_encoded", "video_frames_encoded"])
        writer.writeheader()
        for split, count in row_counts.items():
            for index in range(count):
                images = successes.get((split, index, "image"), [])
                videos = successes.get((split, index, "video"), [])
                writer.writerow({"split": split, "row_index": index, "image_missing": int(not images),
                                 "video_missing": int(not videos), "image_assets_encoded": len(images),
                                 "video_assets_encoded": len(videos), "video_frames_encoded": sum(x["frame_count"] for x in videos)})


def run(args):
    deadline = parse_deadline(args.deadline)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    row_counts = {"train": args.train_rows, "test": args.test_rows}
    records = read_manifest(args.manifest, row_counts)
    validated = []
    for number, record in enumerate(records):
        if number >= args.max_assets:
            validated.append({**record, "valid": False, "error": "asset_budget_exceeded"})
            continue
        try:
            validated.append(validate_asset(record, deadline))
        except TimeoutError:
            validated.extend({**pending, "valid": False, "error": "deadline_before_validation"} for pending in records[number:])
            break
    (output / "local_visual_validation.json").write_text(json.dumps(validated, indent=2))
    valid = [record for record in validated if record["valid"]]
    report = {"manifest_sha256": sha256(args.manifest), "model": "OpenAI CLIP ViT-B/32",
              "official_model_url": MODEL_URL, "expected_model_sha256": MODEL_SHA256,
              "manifest_assets": len(records), "valid_assets": len(valid), "encoded_assets": 0,
              "inference_executed": False, "model_downloaded": False, "model_trained": False,
              "row_counts": row_counts, "embedding_dimensions": 512, "cpu_threads": 2,
              "video_frame_fractions": list(FRAME_FRACTIONS), "embedding_missing_policy": "sparse rows; no fabricated vectors",
              "validate_only": args.validate_only}
    successes, encoded = {}, []
    if args.validate_only or not valid:
        report["status"] = "validated_without_inference" if args.validate_only else "no_valid_local_assets"
        save_coverage(output, row_counts, successes)
        (output / "local_visual_run.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return report
    model_path = Path(args.model_path).resolve() if args.model_path else None
    if model_path is None or not model_path.is_file():
        raise ValueError("Valid assets exist: provide --model-path to the already downloaded official ViT-B-32.pt")
    if sha256(model_path, deadline) != MODEL_SHA256:
        raise ValueError("CLIP weights SHA-256 does not match the official frozen ViT-B/32 checkpoint")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import numpy as np
    import torch
    import clip
    from PIL import Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model, preprocess = clip.load(str(model_path), device="cpu", jit=False)
    model.eval()
    cache = output / "embedding_cache"
    cache.mkdir(exist_ok=True)
    report.update(inference_executed=True, status="completed", model_sha256=MODEL_SHA256,
                  torch_version=torch.__version__, clip_module_path=str(Path(clip.__file__).resolve()),
                  clip_init_sha256=sha256(clip.__file__, deadline),
                  clip_loader_sha256=sha256(Path(clip.__file__).parent / "clip.py", deadline),
                  clip_architecture_sha256=sha256(Path(clip.__file__).parent / "model.py", deadline))
    try:
        for record in valid:
            check_deadline(deadline)
            frames = video_frames(record, output, deadline) if record["kind"] == "video" else [
                {"path": record["path"], "timestamp_seconds": None, "sha256": record["asset_sha256"]}]
            vectors = []
            for frame in frames:
                check_deadline(deadline)
                key = hashlib.sha256((MODEL_SHA256 + frame["sha256"] + "official_clip_preprocess_v1").encode()).hexdigest()
                cached = cache / f"{key}.npy"
                if cached.exists():
                    vector = np.load(cached, allow_pickle=False)
                else:
                    with Image.open(frame["path"]) as image:
                        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
                    with torch.inference_mode():
                        vector = model.encode_image(tensor).float().cpu().numpy()[0]
                    norm = float(np.linalg.norm(vector))
                    if vector.shape != (512,) or not np.isfinite(vector).all() or norm <= 0:
                        raise ValueError("Invalid CLIP vector")
                    vector = (vector / norm).astype(np.float32)
                    np.save(cached, vector, allow_pickle=False)
                if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1., atol=1e-4):
                    raise ValueError("Invalid cached normalized CLIP vector")
                vectors.append(vector)
            vector = np.mean(vectors, axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            key = (record["split"], record["row_index"], record["kind"])
            successes.setdefault(key, []).append({"embedding": vector, "frame_count": len(frames) if record["kind"] == "video" else 0})
            encoded.append({**record, "frames": frames})
            report["encoded_assets"] += 1
    except (TimeoutError, subprocess.TimeoutExpired):
        report["status"] = "deadline_stopped"
    except Exception as error:
        report.update(status="inference_error", error=type(error).__name__ + ": " + str(error)[:500])
        raise
    finally:
        save_coverage(output, row_counts, successes)
        for split in row_counts:
            for kind in ("image", "video"):
                keys = sorted(key for key in successes if key[0] == split and key[2] == kind)
                vectors = []
                for key in keys:
                    vector = np.mean([item["embedding"] for item in successes[key]], axis=0)
                    vector /= max(float(np.linalg.norm(vector)), 1e-12)
                    vectors.append(vector.astype(np.float32))
                np.savez_compressed(output / f"local_clip_{split}_{kind}.npz",
                                    row_index=np.asarray([key[1] for key in keys], dtype=np.int64),
                                    embeddings=np.stack(vectors) if vectors else np.empty((0, 512), dtype=np.float32))
        (output / "local_visual_encoded_assets.json").write_text(json.dumps(encoded, indent=2))
        report["covered_image_items"] = sum(key[2] == "image" for key in successes)
        report["covered_video_items"] = sum(key[2] == "video" for key in successes)
        (output / "local_visual_run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", help="Unix epoch or timezone-aware ISO-8601")
    parser.add_argument("--model-path")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--train-rows", type=int, default=70000)
    parser.add_argument("--test-rows", type=int, default=30000)
    parser.add_argument("--max-assets", type=int, default=512)
    args = parser.parse_args()
    if min(args.train_rows, args.test_rows) < 0 or args.max_assets < 0:
        parser.error("Row counts and max-assets must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
