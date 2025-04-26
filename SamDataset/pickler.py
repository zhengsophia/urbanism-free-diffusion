import os
import pickle
import argparse
from pathlib import Path

def collect_sa_ids(directory: Path):
    """
    Scan `directory` for filenames matching sa_*.jpg,
    and return a list of the base filenames (without .jpg).
    """
    ids = []
    for img_path in directory.glob("sa_*.jpg"):
        # img_path.name is e.g. "sa_000123.jpg"
        ids.append(img_path.stem)  # "sa_000123"
    return sorted(ids)

def save_to_pickle(ids, out_path: Path):
    """
    Save the list `ids` to a pickle file at `out_path`.
    """
    with open(out_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved {len(ids)} ids to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Collect sa_*.jpg filenames and pickle their IDs"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the folder containing sa_*.jpg files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("sa_ids.pkl"),
        help="Output pickle file (default: sa_ids.pkl)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory.")
        return

    ids = collect_sa_ids(args.directory)
    if not ids:
        print(f"No sa_*.jpg files found in {args.directory}")
    else:
        save_to_pickle(ids, args.output)

if __name__ == "__main__":
    main()
