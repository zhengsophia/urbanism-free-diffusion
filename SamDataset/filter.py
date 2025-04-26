import argparse
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

# ─── DEFINE YOUR TOPICS IN ORDER (most specific first) ────────────────────────
# For example, vehicles are more specific than roads, so list them first.
TOPIC_SEQUENCE: List[Tuple[str, List[str]]] = [
    ("vehicle",  ["vehicle", "car", "truck", "bus", "plane", "bicycle", "motorcycle", "scooter", "moped", "train", "tram", "pedestrian"]),
    ("road",     ["road", "highway", "street", "sidewalk", "runway", "driveway", "crosswalk", "bridge", "tunnels"]),
    ("building", ["building", "skyscraper", "house", "apartment", "condo", "mansion", "airport"]),
    ("city",     ["village", "city", "metropolis", "cities", "downtown", "town", "suburb", "cosmopolis"]),
    ("urban",    ["urban", "architecture", "cityscape", "municipal"]),
    # …add more categories here…
]
# ────────────────────────────────────────────────────────────────────────────────

def load_ids(pickle_path: Path) -> List[str]:
    """Load list of sa_* IDs (strings) from a pickle."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def chunk_list(lst: List[str], n_chunks: int) -> List[List[str]]:
    """Split `lst` into n_chunks sublists, as evenly as possible."""
    k, m = divmod(len(lst), n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = k + (1 if i < m else 0)
        chunks.append(lst[start:start+size])
        start += size
    return chunks

def process_chunk(sa_ids: List[str], cap_dir: Path) -> Dict[str, List[str]]:
    """
    For each sa_id in this chunk:
      - load its caption text
      - assign to the first matching topic from TOPIC_SEQUENCE
      - if no topic matches, assign to "others"
    Returns a dict: { topic_name: [sa_ids...], ..., "others": [...] }
    """
    # prepare empty buckets
    buckets: Dict[str, List[str]] = {name: [] for name, _ in TOPIC_SEQUENCE}
    buckets["filtered"] = []

    for sa_id in sa_ids:
        cap_file = cap_dir / f"{sa_id}.txt"
        if not cap_file.exists():
            print(f"Warning: caption not found: {cap_file}")
            continue

        text = cap_file.read_text(encoding='utf-8').lower()
        placed = False

        # check topics in order, stop at first match
        for topic, keywords in TOPIC_SEQUENCE:
            if any(kw in text for kw in keywords):
                buckets[topic].append(sa_id)
                placed = True
                break

        if not placed:
            buckets["filtered"].append(sa_id)

    return buckets

def main():
    parser = argparse.ArgumentParser(
        description="Ordered, parallel caption-based categorization for SA-1B IDs"
    )
    parser.add_argument(
        "--ids-pkl",
        type=Path,
        default=Path("ids.pkl"),
        help="Pickle file with the list of sa_* IDs (default: ids.pkl)",
    )
    parser.add_argument(
        "--cap-dir",
        type=Path,
        default=Path("./captions"),
        help="Directory containing sa_*.txt caption files",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=16,
        help="Number of worker processes",
    )
    args = parser.parse_args()

    sa_ids = load_ids(args.ids_pkl)
    if not sa_ids:
        print("No IDs loaded—nothing to do.")
        return

    chunks = chunk_list(sa_ids, args.num_procs)

    # prepare global buckets
    all_buckets: Dict[str, List[str]] = {name: [] for name, _ in TOPIC_SEQUENCE}
    all_buckets["filtered"] = []

    with ProcessPoolExecutor(max_workers=args.num_procs) as exe:
        futures = [exe.submit(process_chunk, chunk, args.cap_dir)
                   for chunk in chunks]
        for fut in as_completed(futures):
            result = fut.result()
            for topic, ids in result.items():
                all_buckets[topic].extend(ids)
    total = len(sa_ids)
    print(f"Total IDs: {total}")
    for topic, ids in all_buckets.items():
        print(f"  {topic:10s}: {len(ids):5d} IDs")
        out_pkl = Path(f"{topic}_ids.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(ids, f)
    print("Done. Pickled one file per topic plus 'others_ids.pkl'.")
if __name__ == "__main__":
    main()
