"""Streaming media inventory and bounded, ordinary HTTP asset-access probes.

Never retrieves item metadata/labels, changes asset IDs, or follows video pages.
Only stored URLs and the HTTPS equivalent of an original HTTP URL are tested.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def nonempty(value):
    return value is not None and value != "" and value != [] and value != {}


def dimensions(value):
    try:
        width, height = map(float, str(value).lower().split("x"))
        return (width, height) if width > 0 and height > 0 else None
    except (ValueError, TypeError):
        return None


def media_features(row):
    pictures = [p for p in (row.get("pictures") or []) if isinstance(p, dict)]
    result = {"media_picture_n": len(pictures), "media_has_video": int(nonempty(row.get("video_id"))),
              "media_has_thumbnail": int(nonempty(row.get("thumbnail"))),
              "media_has_secure_thumbnail": int(nonempty(row.get("secure_thumbnail"))),
              "media_secure_url_fraction": sum(bool(p.get("secure_url")) for p in pictures) / max(1, len(pictures)),
              "media_quality_nonempty_fraction": sum(nonempty(p.get("quality")) for p in pictures) / max(1, len(pictures))}
    for source in ("size", "max_size"):
        pairs = [value for value in (dimensions(p.get(source)) for p in pictures) if value]
        result[f"media_{source}_available_fraction"] = len(pairs) / max(1, len(pictures))
        measures = {"width": [w for w, h in pairs], "height": [h for w, h in pairs],
                    "area": [w*h for w, h in pairs], "aspect": [w/h for w, h in pairs]}
        for name, values in measures.items():
            for summary, function in (("mean", statistics.mean), ("min", min), ("max", max)):
                result[f"media_{source}_{name}_{summary}"] = function(values) if values else float("nan")
            result[f"media_{source}_{name}_std"] = statistics.pstdev(values) if values else float("nan")
        for name, predicate in (("portrait", lambda w,h: w<h), ("landscape", lambda w,h: w>h),
                                ("square", lambda w,h: w==h), ("large", lambda w,h: min(w,h)>=500)):
            result[f"media_{source}_{name}_fraction"] = sum(predicate(w,h) for w,h in pairs)/len(pairs) if pairs else float("nan")
    ratios = [mw*mh/(w*h) for p in pictures
              if (small := dimensions(p.get("size"))) and (large := dimensions(p.get("max_size")))
              for w,h in [small] for mw,mh in [large]]
    result["media_resolution_expansion_mean"] = statistics.mean(ratios) if ratios else float("nan")
    result["media_resolution_expansion_max"] = max(ratios) if ratios else float("nan")
    return result


def inventory(path, feature_path=None):
    fields = collections.Counter()
    types = collections.defaultdict(collections.Counter)
    hosts = collections.Counter()
    picture_counts = collections.Counter()
    sample_images, sample_videos, video_samples = [], [], []
    rows = 0
    digest = hashlib.sha256()
    feature_stream = feature_path.open("w", newline="") if feature_path else None
    writer = None
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            digest.update(line)
            row = json.loads(line)
            rows += 1
            if feature_stream:
                features = {"row_index": index, **media_features(row)}
                if writer is None:
                    writer = csv.DictWriter(feature_stream, fieldnames=list(features))
                    writer.writeheader()
                writer.writerow(features)
            for key, value in row.items():
                if any(token in key.lower() for token in ("picture", "image", "thumbnail", "video")):
                    types[key][type(value).__name__] += 1
                    if nonempty(value):
                        fields[key] += 1
                    if "video" in key.lower() and nonempty(value) and len(video_samples) < 10:
                        video_samples.append({"row_index": index, "field": key, "value": value})
                    if "video" in key.lower() and isinstance(value, str) and value.startswith(("http://", "https://")):
                        if len(sample_videos) < 6 and value not in sample_videos:
                            sample_videos.append(value)
                    elif key == "video_id" and isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}(?:;youtube)?", value):
                        # Historical Mercado Libre documentation defines video_id as YOUTUBE_ID_HERE.
                        url = "https://www.youtube.com/embed/" + value.split(";")[0]
                        if len(sample_videos) < 6 and url not in sample_videos:
                            sample_videos.append(url)
            pictures = row.get("pictures") or []
            picture_counts[len(pictures) if isinstance(pictures, list) else -1] += 1
            candidates = [row.get("thumbnail"), row.get("secure_thumbnail")]
            if isinstance(pictures, list):
                for picture in pictures:
                    if isinstance(picture, dict):
                        candidates.extend([picture.get("url"), picture.get("secure_url")])
            for url in candidates:
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    hosts[urllib.parse.urlsplit(url).hostname] += 1
                    if len(sample_images) < 12 and url not in sample_images:
                        sample_images.append(url)
    if feature_stream:
        feature_stream.close()
    return {"rows": rows, "sha256": digest.hexdigest(), "nonempty_fields": dict(fields),
            "field_types": {key: dict(value) for key, value in types.items()},
            "picture_count_distribution": dict(picture_counts), "image_url_hosts": dict(hosts),
            "sample_images": sample_images, "sample_video_urls": sample_videos,
            "sample_video_values": video_samples}


def probe(url, kind, timeout=12):
    started = time.monotonic()
    result = {"url": url, "host": urllib.parse.urlsplit(url).hostname, "kind": kind}
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            sample = response.read(262144 if kind == "video" else 4096)
            result.update(status=response.status, content_type=response.headers.get("Content-Type", ""),
                          bytes_sampled=len(sample), final_url=response.url,
                          sample_sha256=hashlib.sha256(sample).hexdigest())
        mime = result["content_type"].split(";")[0].strip().lower()
        image_magic = sample.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")) or (sample[:4] == b"RIFF" and sample[8:12] == b"WEBP")
        result["accessible_media"] = bool(result["status"] == 200 and (image_magic if kind == "image" else mime.startswith("video/")))
        result["image_signature_valid"] = image_magic if kind == "image" else None
        if kind == "video" and mime == "text/html":
            html = sample.decode("utf-8", errors="replace")
            status = re.search(r'"playabilityStatus"\s*:\s*\{\s*"status"\s*:\s*"([A-Z_]+)"', html)
            result["player_playability_status"] = status.group(1) if status else None
            result["player_has_streaming_data"] = bool(re.search(r'"streamingData"\s*:', html))
            result["video_frames_retrieved"] = False
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        result.update(status=getattr(error, "code", None), accessible_media=False, error=str(error))
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probes", type=int, default=20)
    parser.add_argument("--video-followup", action="store_true", help="Include two public watch-page/thumbnail checks inside the 20-request total cap.")
    parser.add_argument("--cdn-migration", action="store_true", help="Include up to four documented current-CDN candidates of the same legacy asset, inside the total cap.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventories = {split: inventory(args.data_dir / f"{split}_data.jsonlines", args.output / f"media_features_{split}.csv") for split in ("train", "test")}
    (args.output / "media_inventory.json").write_text(json.dumps(inventories, indent=2, ensure_ascii=False))
    queue, seen = [], set()
    if args.cdn_migration:
        candidates = [url for url in inventories["train"]["sample_images"]
                      if url.endswith("-O.jpg") and ".mlstatic.com/" in url and "akamai" not in url][:2]
        for url in candidates:
            basename = url.rsplit("/", 1)[-1]
            for name in (basename, "D_NQ_NP_" + basename):
                variant = "https://http2.mlstatic.com/" + name
                if variant not in seen:
                    seen.add(variant)
                    queue.append((variant, "image", "train"))
    if args.video_followup:
        for split, info in inventories.items():
            if info["sample_video_urls"]:
                video_id = info["sample_video_urls"][0].rsplit("/", 1)[-1]
                for variant, kind in (("https://www.youtube.com/watch?v=" + video_id, "video"),
                                      ("https://i.ytimg.com/vi/" + video_id + "/default.jpg", "image")):
                    seen.add(variant)
                    queue.append((variant, kind, split))
    # Include both train and test, original pictures and original thumbnails.
    for kind in ("video", "image"):
      for split, info in inventories.items():
        for urls in ([info["sample_video_urls"][:2]] if kind == "video" else [info["sample_images"][:4]]):
            for url in urls:
                variants = [url]
                if url.startswith("http://"):
                    variants.append("https://" + url[7:])
                for variant in variants:
                    if variant not in seen:
                        seen.add(variant)
                        queue.append((variant, kind, split))
    records = []
    for url, kind, split in queue[:max(0, min(args.max_probes, 20))]:
        result = probe(url, kind)
        result["split"] = split
        records.append(result)
        print(json.dumps(result), flush=True)
    result = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "max_probes": 20,
              "actual_probes": len(records), "probes": records,
              "accessible_media": sum(record["accessible_media"] for record in records),
              "image_model_trained": False, "video_model_trained": False,
              "bulk_download_attempted": False,
              "policy": "Original stored media URLs and ordinary HTTPS equivalents only; no item/API label lookups, credentials or block evasion."}
    (args.output / "media_access_probes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"inventory_rows": {split: info["rows"] for split, info in inventories.items()},
                      "access": {key: result[key] for key in ("actual_probes", "accessible_media")}}), flush=True)


if __name__ == "__main__":
    main()
