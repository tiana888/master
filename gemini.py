import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DEFAULT_TEXTURE_DIR = SCRIPT_DIR / "assets" / "texture"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "gemini" / "texture_descriptions"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_SLEEP_SECONDS = 2.0

TEXTURE_PROMPT = """Role: You are a texture and formal analysis expert. Task: Describe the provided image's surface patterns and lines in less than 20 words, focusing exclusively on textures.

Strict Constraint: DO NOT mention any object names or semantic content (e.g., no "fox," "fur," "grass"). FORMAT: Return only a list of descriptive texture terms separated by commas, similar to a texture swatch. NO PREAMBLE: Do not say "The textures are..." or "Here is the list."
"""


def create_genai_client():
    try:
        from google import genai
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Missing Gemini SDK. Install it with `python -m pip install -U google-genai`, "
            "then set GEMINI_API_KEY before running."
        ) from exc

    return genai.Client()


def list_image_files(texture_dir: Path, recursive: bool = False) -> List[Path]:
    if not texture_dir.is_dir():
        raise NotADirectoryError(f"Texture folder not found: {texture_dir}")

    iterator: Iterable[Path] = texture_dir.rglob("*") if recursive else texture_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def safe_stem(path: Path, limit: int = 120) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in path.stem.strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned or "texture")[:limit]


def collect_response_text(response) -> str:
    response_text = getattr(response, "text", None)
    if response_text:
        return str(response_text).strip()

    text_parts: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))

    return "\n".join(text_parts).strip()


def count_words(text: str) -> int:
    return len([word for word in text.replace(",", " ").split() if word.strip()])


def analyze_texture_image(client, image_path: Path, model: str, dry_run: bool = False) -> str:
    if dry_run:
        return "[DRY-RUN]"

    response = client.models.generate_content(
        model=model,
        contents=[
            TEXTURE_PROMPT,
            load_rgb_image(image_path),
        ],
    )
    return collect_response_text(response)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "image_path",
        "output_path",
        "model",
        "status",
        "word_count",
        "constraint_ok",
        "gemini_output",
        "error",
        "created_at",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_record(
    image_path: Path,
    output_path: Path,
    model: str,
    status: str,
    gemini_output: str = "",
    error: str = "",
) -> dict:
    word_count = count_words(gemini_output) if gemini_output else 0
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "output_path": str(output_path),
        "model": model,
        "status": status,
        "word_count": word_count,
        "constraint_ok": bool(gemini_output) and word_count <= 20,
        "gemini_output": gemini_output,
        "error": error,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send every image in assets/texture to Gemini and save texture-only descriptions."
    )
    parser.add_argument(
        "--texture-dir",
        type=Path,
        default=DEFAULT_TEXTURE_DIR,
        help="Folder containing texture images. Defaults to assets/texture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where Gemini text outputs will be saved.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texture_dir = args.texture_dir.resolve()
    output_dir = args.output_dir.resolve()
    per_image_dir = output_dir / "per_image"
    jsonl_path = output_dir / "gemini_texture_outputs.jsonl"
    csv_path = output_dir / "gemini_texture_outputs.csv"
    summary_path = output_dir / "summary.json"

    image_paths = list_image_files(texture_dir, recursive=args.recursive)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No texture images found in {texture_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            client = create_genai_client()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    rows: List[dict] = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"[INFO] Texture folder: {texture_dir}")
    print(f"[INFO] Output folder:  {output_dir}")
    print(f"[INFO] Model:          {args.model}")
    print(f"[INFO] Images:         {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):
        output_path = per_image_dir / f"{safe_stem(image_path)}.txt"
        print(f"[RUN] {index}/{len(image_paths)} {image_path.name}")

        if output_path.exists() and not args.overwrite:
            gemini_output = output_path.read_text(encoding="utf-8").strip()
            record = build_record(
                image_path=image_path,
                output_path=output_path,
                model=args.model,
                status="skipped_existing",
                gemini_output=gemini_output,
            )
            rows.append(record)
            skipped_count += 1
            print(f"[SKIP] Existing output: {output_path}")
            continue

        try:
            gemini_output = analyze_texture_image(
                client=client,
                image_path=image_path,
                model=args.model,
                dry_run=args.dry_run,
            )
            write_text(output_path, gemini_output)
            record = build_record(
                image_path=image_path,
                output_path=output_path,
                model=args.model,
                status="ok",
                gemini_output=gemini_output,
            )
            rows.append(record)
            success_count += 1
            print(f"[SAVE] {output_path}")
            print(f"[TEXT] {gemini_output}")
        except Exception as exc:
            record = build_record(
                image_path=image_path,
                output_path=output_path,
                model=args.model,
                status="error",
                error=str(exc),
            )
            rows.append(record)
            failed_count += 1
            print(f"[ERROR] {image_path.name}: {exc}")

        if index < len(image_paths) and args.sleep > 0:
            time.sleep(args.sleep)

    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "texture_dir": str(texture_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "prompt": TEXTURE_PROMPT,
        "num_images": len(image_paths),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] CSV:     {csv_path}")
    print(f"[SAVE] JSONL:   {jsonl_path}")
    print(f"[SAVE] Summary: {summary_path}")
    print(f"[DONE] ok={success_count}, skipped={skipped_count}, failed={failed_count}")


if __name__ == "__main__":
    main()
