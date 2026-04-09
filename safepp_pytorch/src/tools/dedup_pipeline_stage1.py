#!/usr/bin/env python3
"""
AIGC image deduplication pipeline.

Goals:
1) Scan a manifest YAML in the same style as the user's current split script.
2) Remove true duplicates for training (exact bytes / decoded pixels / near duplicates).
3) Build split-safe components that also bind paired content (FakeClue real/fake) and
   optionally lock DRAGON same-stem samples across generators into one split unit.

Outputs under --out_dir:
- scan_index.csv: all scanned samples with metadata and hashes
- dedup_pairs.csv: duplicate or split-lock edges with reasons
- components.csv: per-sample dedup_component_id / split_component_id / keep flags
- clean_index.csv: canonical samples after dedup, ready for split generation
- report.yaml: summary statistics

Designed to keep dependencies modest: PIL, numpy, pandas, scipy, pyyaml, tqdm.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.fftpack import dct

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

IMAGE_EXTS = ["jpg", "jpeg", "png", "webp", "bmp"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def normalize_split_hint(x: object) -> str:
    if x is None:
        return "seen"
    text = str(x).strip().lower()
    return text or "seen"


def normalize_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_stem(stem: str) -> str:
    s = normalize_text(stem)
    s = re.sub(r"(_(real|fake|gt|mask|copy|dup|duplicate))+$", "", s)
    s = re.sub(r"(_v\d+)$", "", s)
    return s or "unknown"


def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class Edge:
    src_idx: int
    dst_idx: int
    reason: str
    score: str = ""
    stage: str = "dedup"  # dedup | split_lock | suspicious


def iter_images(root: Path, recursive: bool, exts: Iterable[str]) -> List[Path]:
    valid_exts = {"." + e.lower().lstrip(".") for e in exts}
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files)


def resolve_group_token(path: Path, level: int) -> str:
    level = max(1, int(level))
    parents = list(path.parents)
    if not parents:
        return path.stem or "unknown"
    idx = min(level - 1, len(parents) - 1)
    token = parents[idx].name.strip()
    if token:
        return token
    for parent in parents[idx + 1 :]:
        if parent.name.strip():
            return parent.name.strip()
    return path.stem or "unknown"


def hash_bucket(text: str, num_buckets: int) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"bucket_{int(digest, 16) % max(1, num_buckets):03d}"


def infer_fakeclue_pair_group(spec: Dict, path: Path) -> str:
    dataset = normalize_text(str(spec.get("dataset", "")))
    if dataset != "fakeclue":
        return ""
    name = normalize_text(str(spec.get("name", "")))
    if name.startswith("fakeclue_"):
        body = name[len("fakeclue_") :]
        if body.endswith("_real"):
            body = body[: -len("_real")]
        elif body.endswith("_fake"):
            body = body[: -len("_fake")]
        if body:
            return f"fakeclue:{body}"
    parts = [normalize_text(x) for x in path.parts]
    if "fakeclue" in parts:
        try:
            idx = parts.index("fakeclue")
            subset = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
            if subset:
                return f"fakeclue:{subset}"
        except Exception:
            pass
    return "fakeclue:unknown"


def infer_dragon_prompt_group(spec: Dict, path: Path, enabled: bool) -> str:
    if not enabled:
        return ""
    dataset = normalize_text(str(spec.get("dataset", "")))
    if dataset != "dragon":
        return ""
    stem = normalize_stem(path.stem)
    if not stem:
        return ""
    return f"dragon:{stem}"


def make_row(path: Path, root: Path, spec: Dict, default_group_level: int, enable_dragon_stem_lock: bool) -> Dict:
    label = int(spec["label"])
    source = str(spec.get("name", path.parent.name or "unknown"))
    group_level = int(spec.get("group_level", default_group_level))
    group_token = resolve_group_token(path, level=group_level)
    relpath = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    pair_group = infer_fakeclue_pair_group(spec, path)
    prompt_group = infer_dragon_prompt_group(spec, path, enabled=enable_dragon_stem_lock)
    file_stat = path.stat()
    return {
        "path": str(path.resolve()),
        "relpath": relpath,
        "filename": path.name,
        "stem": path.stem,
        "content_id": normalize_stem(path.stem),
        "label": label,
        "source": source,
        "dataset": str(spec.get("dataset", "unknown")),
        "domain": str(spec.get("domain", "unknown")),
        "generator": str(spec.get("generator", "real" if label == 0 else "unknown")),
        "split_hint": normalize_split_hint(spec.get("split_hint", "seen")),
        "sample_weight": float(spec.get("sample_weight", 1.0)),
        "is_hard_negative": int(spec.get("is_hard_negative", 0)),
        "group_level": group_level,
        "group_token": group_token,
        "group_id": f"{source}:{group_token}",
        "source_root": str(root.resolve()),
        "pair_group": pair_group,
        "prompt_group": prompt_group,
        "file_size_bytes": int(file_stat.st_size),
        "mtime": float(file_stat.st_mtime),
    }


def scan_manifest(
    manifest: Dict,
    strict: bool,
    default_group_level: int,
    enable_dragon_stem_lock: bool,
) -> pd.DataFrame:
    recursive_default = bool(manifest.get("recursive", True))
    exts_default = manifest.get("exts", IMAGE_EXTS)
    sources = manifest.get("sources", [])
    if not sources:
        raise ValueError("Manifest must contain a non-empty `sources` list.")

    rows: List[Dict] = []
    missing: List[str] = []
    empty: List[str] = []

    for spec in sources:
        for key in ["path", "label"]:
            if key not in spec:
                raise ValueError(f"Missing `{key}` in source entry: {spec}")
        root = Path(spec["path"])
        recursive = bool(spec.get("recursive", recursive_default))
        exts = spec.get("exts", exts_default)
        if not root.exists():
            missing.append(str(root))
            if strict:
                continue
            print(f"[WARN] missing path: {root}")
            continue
        files = iter_images(root, recursive=recursive, exts=exts)
        if not files:
            empty.append(str(root))
            if strict:
                continue
            print(f"[WARN] no images found: {root}")
            continue
        rows.extend(
            make_row(path, root=root, spec=spec, default_group_level=default_group_level, enable_dragon_stem_lock=enable_dragon_stem_lock)
            for path in files
        )
        print(f"[OK] {spec.get('name', root.name)} -> {len(files)} files")

    if strict and (missing or empty):
        raise RuntimeError(f"Strict mode failed. Missing={missing}, empty={empty}")
    if not rows:
        raise RuntimeError("No image rows were collected from the manifest.")

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["path"], keep="first").reset_index(drop=True)


def maybe_rebucket_flat_sources(df: pd.DataFrame, threshold: int, num_buckets: int) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    by_source = out.groupby("source")["group_token"].nunique().to_dict()
    rebucket_sources = {src for src, n in by_source.items() if int(n) <= int(threshold)}
    if rebucket_sources:
        mask = out["source"].isin(rebucket_sources)
        out.loc[mask, "group_token"] = out.loc[mask, "path"].astype(str).map(lambda p: hash_bucket(p, num_buckets))
        out.loc[mask, "group_id"] = out.loc[mask, "source"].astype(str) + ":" + out.loc[mask, "group_token"].astype(str)
    return out


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pack_bits(bits: np.ndarray) -> int:
    flat = bits.reshape(-1).astype(np.uint8)
    value = 0
    for b in flat:
        value = (value << 1) | int(b)
    return int(value)


def _ahash_from_gray(gray: np.ndarray) -> int:
    small = np.array(Image.fromarray(gray).resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    return _pack_bits(small > float(small.mean()))


def _dhash_from_gray(gray: np.ndarray) -> int:
    small = np.array(Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    diff = small[:, 1:] > small[:, :-1]
    return _pack_bits(diff)


def _phash_from_gray(gray: np.ndarray) -> int:
    small = np.array(Image.fromarray(gray).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    coeff = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeff[:8, :8].copy()
    vals = low.reshape(-1)
    if vals.shape[0] > 1:
        med = float(np.median(vals[1:]))
    else:
        med = float(vals[0])
    bits = low > med
    bits[0, 0] = 0
    return _pack_bits(bits)


def compute_hash_record(path: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "path": path,
        "read_ok": 0,
        "width": -1,
        "height": -1,
        "raw_sha1": "",
        "pixel_sha1": "",
        "ahash64": "",
        "dhash64": "",
        "phash64": "",
        "thumb8_hex": "",
        "error": "",
    }
    try:
        raw_sha1 = _sha1_file(path)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            rgb = np.asarray(im)
            height, width = rgb.shape[:2]
            pixel_h = hashlib.sha1()
            pixel_h.update(f"{width}x{height}".encode("utf-8"))
            pixel_h.update(rgb.tobytes())
            gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
            out.update(
                {
                    "read_ok": 1,
                    "width": int(width),
                    "height": int(height),
                    "raw_sha1": raw_sha1,
                    "pixel_sha1": pixel_h.hexdigest(),
                    "ahash64": f"{_ahash_from_gray(gray):016x}",
                    "dhash64": f"{_dhash_from_gray(gray):016x}",
                    "phash64": f"{_phash_from_gray(gray):016x}",
                    "thumb8_hex": Image.fromarray(gray).resize((8, 8), Image.Resampling.BILINEAR).tobytes().hex(),
                }
            )
    except (UnidentifiedImageError, OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def compute_hashes(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    paths = df["path"].astype(str).tolist()
    records: List[Dict[str, object]] = []

    if workers <= 1:
        iterator: Iterator[str] = iter(paths)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(paths), desc="hashes")
        for path in iterator:
            records.append(compute_hash_record(path))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(compute_hash_record, path) for path in paths]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="hashes")
            for fut in iterator:
                records.append(fut.result())
    hashes_df = pd.DataFrame(records)
    return df.merge(hashes_df, on="path", how="left")


# def hamming_hex(a: str, b: str) -> int:
#     if not a or not b:
#         return 1 << 30
#     return (int(a, 16) ^ int(b, 16)).bit_count()

def hamming_hex(a: str, b: str) -> int:
    if not a or not b:
        return 1 << 30
    x = int(a, 16) ^ int(b, 16)
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def thumb_mae_hex(a: str, b: str) -> float:
    if not a or not b:
        return float("inf")
    arr_a = np.frombuffer(bytes.fromhex(a), dtype=np.uint8).astype(np.float32)
    arr_b = np.frombuffer(bytes.fromhex(b), dtype=np.uint8).astype(np.float32)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return float("inf")
    return float(np.mean(np.abs(arr_a - arr_b)))


def split_priority(split_hint: str) -> int:
    # Keep unseen over seen if duplicates cross split boundaries.
    text = normalize_split_hint(split_hint)
    return 0 if text == "unseen" else 1


def _canonical_sort_key(row: pd.Series) -> Tuple:
    pixel_count = int(row.get("width", -1)) * int(row.get("height", -1))
    return (
        split_priority(row.get("split_hint", "seen")),
        -pixel_count,
        -int(row.get("file_size_bytes", 0)),
        str(row.get("path", "")),
    )


def chain_group_edges(indices: Sequence[int], reason: str, stage: str, score: str = "") -> List[Edge]:
    idxs = list(indices)
    if len(idxs) <= 1:
        return []
    anchor = idxs[0]
    return [Edge(src_idx=anchor, dst_idx=other, reason=reason, stage=stage, score=score) for other in idxs[1:]]


def build_exact_edges(df: pd.DataFrame) -> List[Edge]:
    edges: List[Edge] = []
    for col, reason in [("raw_sha1", "exact_bytes"), ("pixel_sha1", "exact_pixels")]:
        good = df[df[col].astype(str) != ""]
        for _, idxs in good.groupby(col).groups.items():
            if len(idxs) > 1:
                edges.extend(chain_group_edges(sorted(list(idxs)), reason=reason, stage="dedup"))
    return edges


def aspect_close(row_a: pd.Series, row_b: pd.Series, tol: float) -> bool:
    wa, ha = max(1, int(row_a["width"])), max(1, int(row_a["height"]))
    wb, hb = max(1, int(row_b["width"])), max(1, int(row_b["height"]))
    ra = wa / ha
    rb = wb / hb
    return abs(math.log(ra) - math.log(rb)) <= tol


def iter_lsh_candidates(hex_values: Sequence[str], band_bits: int, max_bucket_size: int) -> Iterator[Tuple[int, int]]:
    num_bands = max(1, 64 // max(1, band_bits))
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, value in enumerate(hex_values):
        if not value:
            continue
        intval = int(value, 16)
        for band in range(num_bands):
            band_val = (intval >> (band * band_bits)) & ((1 << band_bits) - 1)
            buckets[(band, int(band_val))].append(idx)

    seen_pairs = set()
    for members in buckets.values():
        if len(members) <= 1:
            continue
        if len(members) > max_bucket_size:
            continue
        members = sorted(members)
        for i in range(len(members)):
            a = members[i]
            for j in range(i + 1, len(members)):
                b = members[j]
                key = (a, b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                yield key


def build_near_edges(
    df: pd.DataFrame,
    phash_threshold: int,
    dhash_threshold: int,
    band_bits: int,
    max_bucket_size: int,
    restrict_to_same_label: bool,
    cross_label_report_threshold: int,
    thumb_mae_threshold: float,
    aspect_log_tol: float,
) -> List[Edge]:
    edges: List[Edge] = []
    work = df.reset_index(drop=False).rename(columns={"index": "row_idx"})
    phashes = work["phash64"].fillna("").astype(str).tolist()
    dhashes = work["dhash64"].fillna("").astype(str).tolist()

    for local_a, local_b in iter_lsh_candidates(phashes, band_bits=band_bits, max_bucket_size=max_bucket_size):
        row_a = work.iloc[local_a]
        row_b = work.iloc[local_b]
        if int(row_a["read_ok"]) != 1 or int(row_b["read_ok"]) != 1:
            continue
        if not aspect_close(row_a, row_b, tol=aspect_log_tol):
            continue
        same_label = int(row_a["label"]) == int(row_b["label"])
        ph = hamming_hex(str(row_a["phash64"]), str(row_b["phash64"]))
        dh = hamming_hex(str(row_a["dhash64"]), str(row_b["dhash64"]))
        mae = thumb_mae_hex(str(row_a.get("thumb8_hex", "")), str(row_b.get("thumb8_hex", "")))
        if same_label:
            if mae <= thumb_mae_threshold and (ph <= phash_threshold or dh <= dhash_threshold):
                edges.append(
                    Edge(
                        src_idx=int(row_a["row_idx"]),
                        dst_idx=int(row_b["row_idx"]),
                        reason="near_hash",
                        score=f"ph={ph},dh={dh},mae={mae:.2f}",
                        stage="dedup",
                    )
                )
        else:
            if not restrict_to_same_label and mae <= thumb_mae_threshold and (ph <= phash_threshold or dh <= dhash_threshold):
                edges.append(
                    Edge(
                        src_idx=int(row_a["row_idx"]),
                        dst_idx=int(row_b["row_idx"]),
                        reason="near_hash_cross_label",
                        score=f"ph={ph},dh={dh},mae={mae:.2f}",
                        stage="dedup",
                    )
                )
            elif mae <= thumb_mae_threshold and (ph <= cross_label_report_threshold or dh <= max(1, cross_label_report_threshold - 2)):
                edges.append(
                    Edge(
                        src_idx=int(row_a["row_idx"]),
                        dst_idx=int(row_b["row_idx"]),
                        reason="suspicious_cross_label_near",
                        score=f"ph={ph},dh={dh},mae={mae:.2f}",
                        stage="suspicious",
                    )
                )
    return edges


def build_fakeclue_pair_edges(df: pd.DataFrame) -> List[Edge]:
    edges: List[Edge] = []
    sub = df[(df["pair_group"].astype(str) != "") & (df["content_id"].astype(str) != "")]
    if len(sub) == 0:
        return edges
    for _, block in sub.groupby(["pair_group", "content_id"], sort=False):
        if len(block) <= 1:
            continue
        idxs = block.index.tolist()
        labels = set(block["label"].astype(int).tolist())
        if labels == {0, 1}:
            edges.extend(chain_group_edges(idxs, reason="fakeclue_pair_lock", stage="split_lock"))
    return edges


def build_dragon_prompt_edges(df: pd.DataFrame, max_lock_size: int) -> List[Edge]:
    edges: List[Edge] = []
    sub = df[(df["prompt_group"].astype(str) != "") & (df["dataset"].astype(str).str.lower() == "dragon")]
    if len(sub) == 0:
        return edges
    for _, block in sub.groupby("prompt_group", sort=False):
        if len(block) <= 1:
            continue
        if len(block) > max_lock_size:
            continue
        num_generators = block["generator"].astype(str).nunique()
        if num_generators <= 1:
            continue
        idxs = block.index.tolist()
        edges.extend(chain_group_edges(idxs, reason="dragon_stem_lock", stage="split_lock"))
    return edges


def apply_edges(df: pd.DataFrame, edges: List[Edge]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    n = len(df)
    dedup_uf = UnionFind(n)
    split_uf = UnionFind(n)

    reason_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()

    for edge in edges:
        stage_counter[edge.stage] += 1
        reason_counter[edge.reason] += 1
        if edge.stage == "dedup":
            dedup_uf.union(edge.src_idx, edge.dst_idx)
            split_uf.union(edge.src_idx, edge.dst_idx)
        elif edge.stage == "split_lock":
            split_uf.union(edge.src_idx, edge.dst_idx)
        else:
            # suspicious edges are only logged, not unioned.
            pass

    out = df.copy()
    out["dedup_root"] = [dedup_uf.find(i) for i in range(n)]
    out["split_root"] = [split_uf.find(i) for i in range(n)]

    dedup_id_map: Dict[int, str] = {}
    split_id_map: Dict[int, str] = {}
    for root in sorted(set(out["dedup_root"].tolist())):
        dedup_id_map[root] = f"dedup_{len(dedup_id_map):08d}"
    for root in sorted(set(out["split_root"].tolist())):
        split_id_map[root] = f"split_{len(split_id_map):08d}"

    out["dedup_component_id"] = out["dedup_root"].map(dedup_id_map)
    out["split_component_id"] = out["split_root"].map(split_id_map)

    # Component statistics and canonical selection.
    dedup_meta: Dict[str, Dict[str, object]] = {}
    for comp_id, block in out.groupby("dedup_component_id", sort=False):
        labels = sorted(set(block["label"].astype(int).tolist()))
        splits = sorted(set(block["split_hint"].astype(str).tolist()))
        block_sorted = block.sort_values(
            by=["split_hint", "width", "height", "file_size_bytes", "path"],
            ascending=[True, False, False, False, True],
            key=lambda s: s if s.name != "split_hint" else s.map(split_priority),
        )
        canonical_path = str(block_sorted.iloc[0]["path"])
        dedup_meta[comp_id] = {
            "size": int(len(block)),
            "labels": labels,
            "splits": splits,
            "canonical_path": canonical_path,
            "has_label_conflict": int(len(labels) > 1),
            "cross_split_duplicate": int(len(splits) > 1),
        }

    out["dedup_component_size"] = out["dedup_component_id"].map(lambda x: dedup_meta[str(x)]["size"])
    out["has_label_conflict"] = out["dedup_component_id"].map(lambda x: dedup_meta[str(x)]["has_label_conflict"])
    out["cross_split_duplicate"] = out["dedup_component_id"].map(lambda x: dedup_meta[str(x)]["cross_split_duplicate"])
    out["canonical_path"] = out["dedup_component_id"].map(lambda x: dedup_meta[str(x)]["canonical_path"])
    out["is_canonical"] = (out["path"].astype(str) == out["canonical_path"].astype(str)).astype(int)

    out["dedup_drop_reason"] = ""
    out.loc[out["read_ok"].fillna(0).astype(int) != 1, "dedup_drop_reason"] = "read_error"
    out.loc[out["has_label_conflict"].astype(int) == 1, "dedup_drop_reason"] = "label_conflict"
    mask_dup_noncanonical = (
        (out["dedup_component_size"].astype(int) > 1)
        & (out["is_canonical"].astype(int) == 0)
        & (out["has_label_conflict"].astype(int) == 0)
        & (out["read_ok"].fillna(0).astype(int) == 1)
    )
    out.loc[mask_dup_noncanonical, "dedup_drop_reason"] = "duplicate_noncanonical"
    out["keep_after_dedup"] = (out["dedup_drop_reason"].astype(str) == "").astype(int)

    split_sizes = out.groupby("split_component_id").size().to_dict()
    out["split_component_size"] = out["split_component_id"].map(lambda x: int(split_sizes[str(x)]))

    edge_rows = []
    for edge in edges:
        row_a = out.iloc[edge.src_idx]
        row_b = out.iloc[edge.dst_idx]
        edge_rows.append(
            {
                "src_path": row_a["path"],
                "dst_path": row_b["path"],
                "src_source": row_a["source"],
                "dst_source": row_b["source"],
                "src_label": int(row_a["label"]),
                "dst_label": int(row_b["label"]),
                "reason": edge.reason,
                "score": edge.score,
                "stage": edge.stage,
                "src_dedup_component_id": row_a["dedup_component_id"],
                "dst_dedup_component_id": row_b["dedup_component_id"],
                "src_split_component_id": row_a["split_component_id"],
                "dst_split_component_id": row_b["split_component_id"],
            }
        )
    edges_df = pd.DataFrame(edge_rows)

    stats = {
        "edges_total": int(len(edges)),
        "dedup_edges": int(stage_counter.get("dedup", 0)),
        "split_lock_edges": int(stage_counter.get("split_lock", 0)),
        "suspicious_edges": int(stage_counter.get("suspicious", 0)),
    }
    for reason, count in reason_counter.items():
        stats[f"reason::{reason}"] = int(count)
    return out, edges_df, stats


def summarize_before_after(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, object]:
    def count_by(df: pd.DataFrame, col: str) -> Dict[str, int]:
        if col not in df.columns or len(df) == 0:
            return {}
        return {str(k): int(v) for k, v in df[col].astype(str).value_counts().items()}

    return {
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "removed_rows": int(len(raw_df) - len(clean_df)),
        "raw_by_label": {int(k): int(v) for k, v in raw_df["label"].astype(int).value_counts().sort_index().items()},
        "clean_by_label": {int(k): int(v) for k, v in clean_df["label"].astype(int).value_counts().sort_index().items()},
        "raw_by_dataset": count_by(raw_df, "dataset"),
        "clean_by_dataset": count_by(clean_df, "dataset"),
        "raw_by_source": count_by(raw_df, "source"),
        "clean_by_source": count_by(clean_df, "source"),
        "raw_by_generator": count_by(raw_df[raw_df["label"].astype(int) == 1], "generator"),
        "clean_by_generator": count_by(clean_df[clean_df["label"].astype(int) == 1], "generator"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dedup pipeline for the AIGC training corpus.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    
    parser.add_argument("--default_group_level", type=int, default=1)
    parser.add_argument("--flat_dir_bucket_threshold", type=int, default=1)
    parser.add_argument("--hash_buckets", type=int, default=128)

    parser.add_argument("--phash_threshold", type=int, default=8)
    parser.add_argument("--dhash_threshold", type=int, default=6)
    parser.add_argument("--cross_label_report_threshold", type=int, default=4)
    parser.add_argument("--thumb_mae_threshold", type=float, default=6.0)
    parser.add_argument("--lsh_band_bits", type=int, default=16)
    parser.add_argument("--lsh_max_bucket_size", type=int, default=256)
    parser.add_argument("--aspect_log_tol", type=float, default=0.12)
    parser.add_argument("--allow_cross_label_near_union", action="store_true")

    parser.add_argument("--disable_fakeclue_pair_lock", action="store_true")
    parser.add_argument("--enable_dragon_stem_lock", action="store_true")
    parser.add_argument("--dragon_max_lock_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    manifest = load_yaml(args.manifest)
    raw_df = scan_manifest(
        manifest,
        strict=args.strict,
        default_group_level=args.default_group_level,
        enable_dragon_stem_lock=args.enable_dragon_stem_lock,
    )
    raw_df = maybe_rebucket_flat_sources(raw_df, threshold=args.flat_dir_bucket_threshold, num_buckets=args.hash_buckets)

    hashed_df = compute_hashes(raw_df, workers=args.workers)
    hashed_df = hashed_df.sort_values("path").reset_index(drop=True)

    scan_csv = out_dir / "scan_index.csv"
    hashed_df.to_csv(scan_csv, index=False)

    exact_edges = build_exact_edges(hashed_df)
    near_edges = build_near_edges(
        hashed_df,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
        band_bits=args.lsh_band_bits,
        max_bucket_size=args.lsh_max_bucket_size,
        restrict_to_same_label=not args.allow_cross_label_near_union,
        cross_label_report_threshold=args.cross_label_report_threshold,
        thumb_mae_threshold=args.thumb_mae_threshold,
        aspect_log_tol=args.aspect_log_tol,
    )
    pair_edges = [] if args.disable_fakeclue_pair_lock else build_fakeclue_pair_edges(hashed_df)
    dragon_edges = build_dragon_prompt_edges(hashed_df, max_lock_size=args.dragon_max_lock_size)

    all_edges = exact_edges + near_edges + pair_edges + dragon_edges
    components_df, edges_df, edge_stats = apply_edges(hashed_df, all_edges)

    clean_df = components_df[components_df["keep_after_dedup"].astype(int) == 1].copy().reset_index(drop=True)
    clean_df = clean_df.sort_values(["label", "dataset", "source", "path"]).reset_index(drop=True)

    scan_csv = out_dir / "scan_index.csv"
    pairs_csv = out_dir / "dedup_pairs.csv"
    components_csv = out_dir / "components.csv"
    clean_csv = out_dir / "clean_index.csv"
    report_yaml = out_dir / "report.yaml"

    hashed_df.to_csv(scan_csv, index=False)
    edges_df.to_csv(pairs_csv, index=False)
    components_df.to_csv(components_csv, index=False)
    clean_df.to_csv(clean_csv, index=False)

    dedup_drop_counts = {str(k): int(v) for k, v in components_df["dedup_drop_reason"].astype(str).replace("", "keep").value_counts().items()}
    suspicious_df = edges_df[edges_df["stage"].astype(str) == "suspicious"] if len(edges_df) else edges_df

    report = {
        "manifest": str(args.manifest),
        "outputs": {
            "scan_index_csv": str(scan_csv),
            "dedup_pairs_csv": str(pairs_csv),
            "components_csv": str(components_csv),
            "clean_index_csv": str(clean_csv),
        },
        "settings": {
            "workers": int(args.workers),
            "strict": bool(args.strict),
            "phash_threshold": int(args.phash_threshold),
            "dhash_threshold": int(args.dhash_threshold),
            "cross_label_report_threshold": int(args.cross_label_report_threshold),
            "thumb_mae_threshold": float(args.thumb_mae_threshold),
            "lsh_band_bits": int(args.lsh_band_bits),
            "lsh_max_bucket_size": int(args.lsh_max_bucket_size),
            "aspect_log_tol": float(args.aspect_log_tol),
            "allow_cross_label_near_union": bool(args.allow_cross_label_near_union),
            "fakeclue_pair_lock": not bool(args.disable_fakeclue_pair_lock),
            "dragon_stem_lock": bool(args.enable_dragon_stem_lock),
            "dragon_max_lock_size": int(args.dragon_max_lock_size),
        },
        "summary": summarize_before_after(components_df, clean_df),
        "dedup_drop_counts": dedup_drop_counts,
        "edge_stats": edge_stats,
        "read_errors": int((components_df["read_ok"].fillna(0).astype(int) != 1).sum()),
        "label_conflict_components": int(
            components_df[components_df["has_label_conflict"].astype(int) == 1]["dedup_component_id"].nunique()
        ),
        "cross_split_duplicate_components": int(
            components_df[components_df["cross_split_duplicate"].astype(int) == 1]["dedup_component_id"].nunique()
        ),
        "suspicious_cross_label_pairs": int(len(suspicious_df)) if len(edges_df) else 0,
    }
    save_yaml(report, str(report_yaml))

    print(f"[DONE] scan_index -> {scan_csv}")
    print(f"[DONE] dedup_pairs -> {pairs_csv}")
    print(f"[DONE] components -> {components_csv}")
    print(f"[DONE] clean_index -> {clean_csv}")
    print(f"[DONE] report -> {report_yaml}")
    print(f"[INFO] raw={len(components_df)} clean={len(clean_df)} removed={len(components_df) - len(clean_df)}")


if __name__ == "__main__":
    main()
