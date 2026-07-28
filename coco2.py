"""
Download real reference images from Unsplash for the SADis geometry baseline prompts.

Usage:
    # WSL / Linux / macOS
    export UNSPLASH_ACCESS_KEY="YOUR_ACCESS_KEY"
    python coco2.py

    # Windows PowerShell
    $env:UNSPLASH_ACCESS_KEY="YOUR_ACCESS_KEY"
    python coco2.py

The script writes one image per key into assets/geometry_unsplash_real by default
and records the selected Unsplash photo metadata in CSV/JSON files.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus, urlencode

import requests
from PIL import Image


UNSPLASH_API_ROOT = "https://api.unsplash.com"
DEFAULT_OUTPUT_DIR = Path("assets/geometry_unsplash_real")


PROMPTS: Dict[str, str] = {
    "animal01": "A fox in the snow.",
    "animal02": "A howling fox.",
    "animal03": "A portrait of fox.",
    "animal04": "A fox sitting on a ground",
    "animal05": "A fox sitting on a ground",
    "animal06": "A fox sitting on a ground",
    "animal07": "A fox sitting on a ground",
    "animal08": "A rabbit on a ground",
    "animal09": "A tiger on a ground",
    "animal10": "A rabbit on a ground",
    "face01": "A woman portrait",
    "face02": "A woman portrait",
    "face03": "A woman portrait",
    "face04": "A woman portrait",
    "face05": "A woman portrait",
    "face06": "A woman portrait",
    "face07": "A woman portrait",
    "face08": "A woman portrait",
    "face09": "A woman portrait",
    "face10": "A woman portrait",
    "face11": "A woman portrait",
    "house01": "A Tudor-style house with bushes.",
    "house02": "A house by a lake with a sailboat, a tree, and flowers.",
    "house03": "A house in a field with a hill.",
    "house04": "A house.",
    "house05": "A house.",
    "house06": "A house.",
    "house07": "A house.",
    "house08": "A house.",
    "person01": "A woman sitting on a chair",
    "person02": "A woman sitting on a chair",
    "person03": "A man sitting on a chair",
    "person04": "A man sitting on a chair",
    "person05": "A man sitting on a chair",
    "seen01": "The building at a corner",
    "seen02": "A tall building with two birds",
    "seen03": "A path in a forest",
    "seen04": "A path and a house in a forest",
    "seen05": "The building",
    "seen06": "The building",
    "seen07": "The building",
}


# Shorter, search-oriented queries. The prompt text is still saved as metadata.
SEARCH_QUERIES: Dict[str, str] = {
    "animal01": "fox in snow wildlife",
    "animal02": "howling fox wildlife",
    "animal03": "fox portrait wildlife",
    "animal04": "fox sitting ground wildlife",
    "animal05": "fox sitting ground wildlife",
    "animal06": "fox sitting ground wildlife",
    "animal07": "fox sitting ground wildlife",
    "animal08": "rabbit on ground wildlife",
    "animal09": "tiger on ground wildlife",
    "animal10": "rabbit on ground wildlife",
    "face01": "woman portrait photo",
    "face02": "woman portrait photo",
    "face03": "woman portrait photo",
    "face04": "woman portrait photo",
    "face05": "woman portrait photo",
    "face06": "woman portrait photo",
    "face07": "woman portrait photo",
    "face08": "woman portrait photo",
    "face09": "woman portrait photo",
    "face10": "woman portrait photo",
    "face11": "woman portrait photo",
    "house01": "Tudor house bushes",
    "house02": "house by lake sailboat flowers",
    "house03": "house in field hill",
    "house04": "house exterior",
    "house05": "house exterior",
    "house06": "house exterior",
    "house07": "house exterior",
    "house08": "house exterior",
    "person01": "woman sitting on chair",
    "person02": "woman sitting on chair",
    "person03": "man sitting on chair",
    "person04": "man sitting on chair",
    "person05": "man sitting on chair",
    "seen01": "corner building street",
    "seen02": "tall building birds",
    "seen03": "forest path",
    "seen04": "forest path house",
    "seen05": "building exterior",
    "seen06": "building exterior",
    "seen07": "building exterior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one real Unsplash image for each SADis geometry baseline key."
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("UNSPLASH_ACCESS_KEY"),
        help="Unsplash API access key. Defaults to the UNSPLASH_ACCESS_KEY environment variable.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Offset into each query result list. Useful when rerunning for different examples.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between Unsplash requests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_access_key(access_key: str) -> None:
    if not access_key:
        raise RuntimeError(
            "Missing Unsplash API access key. In WSL/bash, run: "
            "export UNSPLASH_ACCESS_KEY='YOUR_ACCESS_KEY'. "
            "In PowerShell, run: $env:UNSPLASH_ACCESS_KEY='YOUR_ACCESS_KEY'. "
            "You can also pass --access-key YOUR_ACCESS_KEY. "
            "Create one from https://unsplash.com/developers"
        )


def request_json(url: str, access_key: str, params: Dict) -> Dict:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Client-ID {access_key}",
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_photos(query: str, access_key: str, per_page: int, page: int) -> List[Dict]:
    payload = request_json(
        f"{UNSPLASH_API_ROOT}/search/photos",
        access_key,
        {
            "query": query,
            "per_page": per_page,
            "page": page,
            "orientation": "squarish",
            "content_filter": "high",
        },
    )
    return payload.get("results", [])


def trigger_download_event(photo: Dict, access_key: str) -> None:
    download_location = photo.get("links", {}).get("download_location")
    if not download_location:
        return
    request_json(download_location, access_key, {})


def image_url(photo: Dict, width: int, height: int) -> str:
    raw_url = photo.get("urls", {}).get("raw")
    if not raw_url:
        raise ValueError(f"Photo {photo.get('id')} does not include urls.raw")
    separator = "&" if "?" in raw_url else "?"
    return raw_url + separator + urlencode(
        {
            "auto": "format",
            "fit": "crop",
            "w": width,
            "h": height,
            "q": 90,
        }
    )


def unsplash_search_url(query: str) -> str:
    return f"https://unsplash.com/s/photos/{quote_plus(query)}"


def download_image(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    # Validate that the saved file is actually an image.
    with Image.open(destination) as image:
        image.verify()


def choose_photo(results: List[Dict], used_photo_ids: set, occurrence_index: int, start_offset: int) -> Dict:
    if not results:
        raise RuntimeError("Unsplash returned no search results")

    preferred_index = occurrence_index + start_offset
    ordered_indices = list(range(preferred_index, len(results))) + list(range(0, preferred_index))
    for index in ordered_indices:
        photo = results[index % len(results)]
        photo_id = photo.get("id")
        if photo_id not in used_photo_ids:
            return photo
    return results[preferred_index % len(results)]


def make_metadata_row(key: str, prompt: str, query: str, photo: Dict, path: Path) -> Dict:
    user = photo.get("user", {})
    links = photo.get("links", {})
    return {
        "key": key,
        "prompt": prompt,
        "query": query,
        "photo_id": photo.get("id", ""),
        "author": user.get("name", ""),
        "author_username": user.get("username", ""),
        "unsplash_url": links.get("html", ""),
        "download_path": str(path),
    }


def save_metadata(rows: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "unsplash_real_images_metadata.csv"
    json_path = output_dir / "unsplash_real_images_metadata.json"

    fieldnames = [
        "key",
        "prompt",
        "query",
        "photo_id",
        "author",
        "author_username",
        "unsplash_url",
        "download_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.dry_run and not args.access_key:
        print("[INFO] Dry run without Unsplash key: showing planned searches only.")
        print("[INFO] To preview selected Unsplash photo IDs/URLs, set UNSPLASH_ACCESS_KEY or pass --access-key.")
        for key, prompt in PROMPTS.items():
            query = SEARCH_QUERIES.get(key, prompt)
            output_path = args.output_dir / f"{key}.jpg"
            print(f"[PLAN] {key}: query={query!r} | url={unsplash_search_url(query)} -> {output_path}")
        return

    require_access_key(args.access_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_photo_ids = set()
    query_occurrences = defaultdict(int)
    metadata_rows = []

    for key, prompt in PROMPTS.items():
        query = SEARCH_QUERIES.get(key, prompt)
        occurrence_index = query_occurrences[query]
        query_occurrences[query] += 1
        output_path = args.output_dir / f"{key}.jpg"

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {key}: {output_path} exists")
            metadata_rows.append(
                {
                    "key": key,
                    "prompt": prompt,
                    "query": query,
                    "photo_id": "",
                    "author": "",
                    "author_username": "",
                    "unsplash_url": "",
                    "download_path": str(output_path),
                }
            )
            continue

        print(f"[SEARCH] {key}: {query}")
        results = search_photos(query, args.access_key, args.per_page, args.page)
        photo = choose_photo(results, used_photo_ids, occurrence_index, args.start_offset)
        used_photo_ids.add(photo.get("id"))
        url = image_url(photo, args.width, args.height)

        if args.dry_run:
            print(f"[DRY] {key}: {photo.get('id')} -> {url}")
        else:
            trigger_download_event(photo, args.access_key)
            download_image(url, output_path)
            print(f"[DONE] {key}: saved {output_path}")

        metadata_rows.append(make_metadata_row(key, prompt, query, photo, output_path))
        time.sleep(args.sleep)

    save_metadata(metadata_rows, args.output_dir)
    print(f"[OK] Metadata saved under {args.output_dir}")


if __name__ == "__main__":
    main()
