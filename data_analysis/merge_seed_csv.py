import argparse
import glob
import os
import re
from typing import List

import pandas as pd


DEFAULT_DIR = r"D:\codes\working\pos\Draft\csv\redo20260112\cls"
DEFAULT_OUT_SUBDIR = "merged"


def _detect_key_columns(dfs: List[pd.DataFrame]) -> List[str]:
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    if "epoch" in common:
        return ["epoch"]
    return []


def _infer_merge_fields(dfs: List[pd.DataFrame], key_cols: List[str]) -> List[str]:
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    merge_fields = sorted(c for c in common if c not in set(key_cols))
    return merge_fields


def _validate_keys_aligned(dfs: List[pd.DataFrame], key_cols: List[str]) -> None:
    if not key_cols:
        return
    base = dfs[0].set_index(key_cols).index
    for idx, df in enumerate(dfs[1:], start=1):
        if not base.equals(df.set_index(key_cols).index):
            raise ValueError(
                f"Key columns {key_cols} are not aligned across files "
                f"(mismatch at file #{idx + 1})."
            )


def _merge_csvs(csv_files: List[str], merge_fields: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in csv_files]
    key_cols = _detect_key_columns(dfs)
    _validate_keys_aligned(dfs, key_cols)

    if not merge_fields:
        merge_fields = _infer_merge_fields(dfs, key_cols)

    for field in merge_fields:
        if any(field not in df.columns for df in dfs):
            raise ValueError(f"Merge field '{field}' is not present in all files.")

    if key_cols:
        base_index = dfs[0].set_index(key_cols).index
    else:
        base_index = dfs[0].index

    merged = pd.DataFrame(index=base_index)
    for field in merge_fields:
        series = [df.set_index(key_cols)[field] if key_cols else df[field] for df in dfs]
        stacked = pd.concat(series, axis=1)
        merged[field] = stacked.mean(axis=1)
        merged[f"{field}_std"] = stacked.std(axis=1, ddof=0)

    if key_cols:
        merged = merged.reset_index()

    return merged


def _collect_seeds(csv_files: List[str]) -> List[str]:
    seeds = []
    for path in csv_files:
        base = os.path.basename(path)
        match = re.search(r"s(\d+)", base)
        if match:
            seeds.append(match.group(1))
    return sorted(set(seeds), key=int) if seeds else []


def _make_output_path(output_dir: str, pattern: str, seeds: List[str]) -> str:
    base = os.path.basename(pattern).replace("*", "").replace("?", "").removesuffix("_").removesuffix("_s")
    safe, ext = os.path.splitext(base)
    if seeds:
        seed_tag = f"s{'_'.join(seeds)}"
        out_name = f"{safe}_{seed_tag}{ext or '.csv'}"
    else:
        out_name = f"{safe}{ext or '.csv'}"
    return os.path.join(output_dir, out_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge seed CSVs by averaging common metrics."
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="CSV filename glob pattern, e.g. base_*_s*.csv",
    )
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated list of fields to merge. Defaults to common fields excluding epoch/step.",
    )
    parser.add_argument(
        "--out-subdir",
        default=DEFAULT_OUT_SUBDIR,
        help="Subdirectory under --dir to save the merged CSV.",
    )
    args = parser.parse_args()

    csv_dir = args.dir
    pattern = args.pattern
    csv_files = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched: {os.path.join(csv_dir, pattern)}")

    merge_fields = []
    if args.fields.strip():
        merge_fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    merged_df = _merge_csvs(csv_files, merge_fields)

    output_dir = os.path.join(csv_dir, args.out_subdir)
    os.makedirs(output_dir, exist_ok=True)
    seeds = _collect_seeds(csv_files)
    output_path = _make_output_path(output_dir, pattern, seeds)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")
    print(f"Files merged: {len(csv_files)}")
    print(f"Columns: {', '.join(merged_df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
