#!/usr/bin/env python3
"""Convert 1218 entity_matching labels to PromptEM format.

Input:
  <dataset_root>/datalake_plus/*.csv
  <dataset_root>/label_plus/entity_matching/{train,validate,test}.csv

Output:
  <output_dir>/{left.txt,right.txt,train.csv,valid.csv,test.csv,manifest.json}
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

SplitRecord = Tuple[str, int, str, int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert *_1218 EM labels into PromptEM data directory")
    p.add_argument("--dataset-root", required=True, help="Path like .../wikidbs_1218")
    p.add_argument("--output-dir", required=True, help="Path like PromptEM_baseline/data/wikidbs_1218")
    p.add_argument("--max-cell-chars", type=int, default=200)
    p.add_argument("--skip-empty", action="store_true", help="Skip empty cells during serialization")
    return p.parse_args()


def normalize_text(v: str) -> str:
    return " ".join(str(v).replace("\n", " ").replace("\r", " ").split())


def serialize_row(table_name: str, row: Dict[str, str], max_cell_chars: int, skip_empty: bool) -> str:
    parts = [f"TABLE {table_name}"]
    for col, val in row.items():
        col = normalize_text(col)
        val = normalize_text(val)
        if max_cell_chars > 0:
            val = val[:max_cell_chars]
        if skip_empty and val == "":
            continue
        parts.append(f"COL {col} VAL {val}")
    return " ".join(parts).strip()


def read_split(path: Path) -> List[SplitRecord]:
    rows: List[SplitRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        required = ["ltable_name", "l_id", "rtable_name", "r_id", "label"]
        for c in required:
            if c not in rd.fieldnames:
                raise ValueError(f"Missing column '{c}' in {path}")
        for rec in rd:
            rows.append(
                (
                    rec["ltable_name"],
                    int(rec["l_id"]),
                    rec["rtable_name"],
                    int(rec["r_id"]),
                    int(rec["label"]),
                )
            )
    return rows


def load_required_rows(
    datalake_dir: Path,
    table_to_ids: Dict[str, Set[int]],
    max_cell_chars: int,
    skip_empty: bool,
) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for table_name in sorted(table_to_ids.keys()):
        needed_ids = table_to_ids[table_name]
        table_path = datalake_dir / table_name
        if not table_path.exists():
            raise FileNotFoundError(f"Missing table file: {table_path}")
        found_ids: Set[int] = set()
        with table_path.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            for idx, row in enumerate(rd):
                if idx in needed_ids:
                    out[(table_name, idx)] = serialize_row(
                        table_name=table_name,
                        row=row,
                        max_cell_chars=max_cell_chars,
                        skip_empty=skip_empty,
                    )
                    found_ids.add(idx)
                    if found_ids == needed_ids:
                        break
        missing = sorted(list(needed_ids - found_ids))
        if missing:
            raise IndexError(f"Row ids not found in {table_name}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_pairs(path: Path, pairs: List[Tuple[int, int, int]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for l_idx, r_idx, y in pairs:
            f.write(f"{l_idx},{r_idx},{y}\n")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    datalake_dir = dataset_root / "datalake_plus"
    label_dir = dataset_root / "label_plus" / "entity_matching"

    train_rows = read_split(label_dir / "train.csv")
    valid_rows = read_split(label_dir / "validate.csv")
    test_rows = read_split(label_dir / "test.csv")

    left_needed: Dict[str, Set[int]] = defaultdict(set)
    right_needed: Dict[str, Set[int]] = defaultdict(set)

    for rows in (train_rows, valid_rows, test_rows):
        for l_table, l_id, r_table, r_id, _ in rows:
            left_needed[l_table].add(l_id)
            right_needed[r_table].add(r_id)

    left_text = load_required_rows(
        datalake_dir=datalake_dir,
        table_to_ids=left_needed,
        max_cell_chars=args.max_cell_chars,
        skip_empty=args.skip_empty,
    )
    right_text = load_required_rows(
        datalake_dir=datalake_dir,
        table_to_ids=right_needed,
        max_cell_chars=args.max_cell_chars,
        skip_empty=args.skip_empty,
    )

    left_index: Dict[Tuple[str, int], int] = {}
    right_index: Dict[Tuple[str, int], int] = {}
    left_lines: List[str] = []
    right_lines: List[str] = []

    def get_left_idx(k: Tuple[str, int]) -> int:
        if k not in left_index:
            left_index[k] = len(left_lines)
            left_lines.append(left_text[k])
        return left_index[k]

    def get_right_idx(k: Tuple[str, int]) -> int:
        if k not in right_index:
            right_index[k] = len(right_lines)
            right_lines.append(right_text[k])
        return right_index[k]

    def map_pairs(rows: List[SplitRecord]) -> List[Tuple[int, int, int]]:
        out_pairs: List[Tuple[int, int, int]] = []
        for l_table, l_id, r_table, r_id, y in rows:
            l_idx = get_left_idx((l_table, l_id))
            r_idx = get_right_idx((r_table, r_id))
            out_pairs.append((l_idx, r_idx, y))
        return out_pairs

    train_pairs = map_pairs(train_rows)
    valid_pairs = map_pairs(valid_rows)
    test_pairs = map_pairs(test_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(output_dir / "left.txt", left_lines)
    write_lines(output_dir / "right.txt", right_lines)
    write_pairs(output_dir / "train.csv", train_pairs)
    write_pairs(output_dir / "valid.csv", valid_pairs)
    write_pairs(output_dir / "test.csv", test_pairs)

    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "counts": {
            "left_entities": len(left_lines),
            "right_entities": len(right_lines),
            "train_pairs": len(train_pairs),
            "valid_pairs": len(valid_pairs),
            "test_pairs": len(test_pairs),
        },
        "settings": {
            "max_cell_chars": args.max_cell_chars,
            "skip_empty": bool(args.skip_empty),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
