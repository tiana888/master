import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "Gemini_Generated_Image_xvxozdxvxozdxvxo.png"
DEFAULT_GEOMETRY = SCRIPT_DIR / "assets" / "geometry" / "animal06.jpeg"
DEFAULT_TEXTURE = SCRIPT_DIR / "assets" / "texture" / "8.png"
DEFAULT_COLOR = SCRIPT_DIR / "assets" / "color" / "color005.jpeg"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "single_image_metrics"
DEFAULT_BASELINES_DIR = SCRIPT_DIR / "results" / "dog_tuning_experiments0508" / "baselines"
DEFAULT_PROMPT = "A single cat on the ground"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def resolve_device(explicit_device: Optional[str]) -> str:
    if explicit_device:
        return explicit_device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def extract_canny_contour(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.bilateralFilter(gray, 5, 35, 35)
    median_value = np.median(blurred)
    lower = int(max(0, (1.0 - 0.25) * median_value))
    upper = int(min(255, (1.0 + 0.25) * median_value))
    if upper <= lower:
        upper = min(255, lower + 1)
    edges = cv2.Canny(blurred, lower, upper)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def color_histogram_distance(image_a: Image.Image, image_b: Image.Image, bins: int = 32) -> float:
    arr_a = np.array(image_a.convert("RGB"))
    arr_b = np.array(image_b.convert("RGB"))
    hist_a = cv2.calcHist([arr_a], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([arr_b], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist_a = cv2.normalize(hist_a, hist_a, norm_type=cv2.NORM_L1).flatten().astype(np.float32)
    hist_b = cv2.normalize(hist_b, hist_b, norm_type=cv2.NORM_L1).flatten().astype(np.float32)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def compute_ms_ssim_similarity(img_a: Image.Image, img_b: Image.Image, device: str) -> float:
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ModuleNotFoundError:
        return math.nan

    target_size = img_b.size

    def to_tensor(img: Image.Image):
        if img.size != target_size:
            img = img.resize(target_size, Image.BICUBIC)
        tensor = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        score = ms_ssim(to_tensor(img_a), to_tensor(img_b), data_range=1.0, size_average=True)
    return float(score.item())


def edge_overlap_metrics(output_sketch: Image.Image, geometry_sketch: Image.Image) -> Dict[str, float]:
    arr_out = np.array(output_sketch.convert("L"))
    arr_geo = np.array(geometry_sketch.convert("L"))
    mask_out = arr_out > 127
    mask_geo = arr_geo > 127
    overlap = float(np.sum(mask_out & mask_geo))
    output_edges = float(np.sum(mask_out))
    geometry_edges = float(np.sum(mask_geo))
    precision = overlap / (output_edges + 1e-8)
    recall = overlap / (geometry_edges + 1e-8)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
    return {
        "edge_precision_geometry": precision,
        "edge_recall_geometry": recall,
        "edge_f1_geometry": f1,
        "output_edge_pixels": output_edges,
        "geometry_edge_pixels": geometry_edges,
    }


def safe_baseline_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", name)


def find_auto_geometry_baseline(geometry_path: Path, baselines_dir: Path) -> Optional[Path]:
    if not baselines_dir.is_dir():
        return None
    matches = sorted(baselines_dir.glob(f"{geometry_path.stem}_geometry_*.png"))
    return matches[0] if matches else None


def find_auto_output_baseline(prompt: str, baselines_dir: Path) -> Optional[Path]:
    if not baselines_dir.is_dir():
        return None
    stem = safe_baseline_name(prompt)
    exact = baselines_dir / f"output_{stem}.png"
    if exact.is_file():
        return exact
    matches = sorted(baselines_dir.glob(f"output_{stem[:40]}*.png"))
    return matches[0] if matches else None


def feature_tensor(features):
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if hasattr(features, "image_embeds"):
        return features.image_embeds
    if hasattr(features, "text_embeds"):
        return features.text_embeds
    return features


def clip_image_text_similarity(model, processor, device: str, image: Image.Image, text: str) -> float:
    import torch

    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        image_features = feature_tensor(model.get_image_features(pixel_values=image_inputs["pixel_values"]))
        text_features = feature_tensor(
            model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float((image_features[0] * text_features[0]).sum().item())


def clip_image_image_similarity(model, processor, device: str, image_a: Image.Image, image_b: Image.Image) -> float:
    import torch

    with torch.no_grad():
        inputs_a = processor(images=image_a, return_tensors="pt").to(device)
        inputs_b = processor(images=image_b, return_tensors="pt").to(device)
        features_a = feature_tensor(model.get_image_features(pixel_values=inputs_a["pixel_values"]))
        features_b = feature_tensor(model.get_image_features(pixel_values=inputs_b["pixel_values"]))
        features_a = features_a / features_a.norm(dim=-1, keepdim=True)
        features_b = features_b / features_b.norm(dim=-1, keepdim=True)
        return float((features_a[0] * features_b[0]).sum().item())


def clip_image_embedding_raw(model, processor, device: str, image: Image.Image):
    import torch

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        features = feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        return features[0].detach().float().cpu()


def l2_distance_from_vectors(vec_a, vec_b) -> float:
    import torch

    return float(torch.norm(vec_a - vec_b, p=2).item())


def tensor_l2_norm(vec) -> float:
    import torch

    return float(torch.norm(vec, p=2).item())


def cosine_similarity_from_vectors(vec_a, vec_b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())


def compute_geometry_residual_metrics(
    model,
    processor,
    device: str,
    geometry_image: Image.Image,
    geometry_baseline_image: Image.Image,
    output_image: Image.Image,
    output_baseline_image: Image.Image,
) -> Dict[str, float]:
    g_canny = extract_canny_contour(geometry_image)
    gb_canny = extract_canny_contour(geometry_baseline_image)
    o_canny = extract_canny_contour(output_image)
    ob_canny = extract_canny_contour(output_baseline_image)

    g_emb = clip_image_embedding_raw(model, processor, device, g_canny)
    gb_emb = clip_image_embedding_raw(model, processor, device, gb_canny)
    o_emb = clip_image_embedding_raw(model, processor, device, o_canny)
    ob_emb = clip_image_embedding_raw(model, processor, device, ob_canny)

    g_res = g_emb - gb_emb
    o_res = o_emb - ob_emb

    return {
        "geometry_residual_l2_distance": l2_distance_from_vectors(g_res, o_res),
        "geometry_residual_cosine_similarity": cosine_similarity_from_vectors(g_res, o_res),
        "geometry_residual_norm": tensor_l2_norm(g_res),
        "output_residual_norm": tensor_l2_norm(o_res),
    }


def compute_geometry_improvement_ratio(
    model,
    processor,
    device: str,
    geometry_image: Image.Image,
    output_image: Image.Image,
    output_baseline_image: Image.Image,
) -> Dict[str, float]:
    geometry_canny = extract_canny_contour(geometry_image)
    output_canny = extract_canny_contour(output_image)
    output_baseline_canny = extract_canny_contour(output_baseline_image)

    geometry_embedding = clip_image_embedding_raw(model, processor, device, geometry_canny)
    output_embedding = clip_image_embedding_raw(model, processor, device, output_canny)
    output_baseline_embedding = clip_image_embedding_raw(model, processor, device, output_baseline_canny)

    output_to_geometry_l2 = l2_distance_from_vectors(output_embedding, geometry_embedding)
    output_baseline_to_geometry_l2 = l2_distance_from_vectors(output_baseline_embedding, geometry_embedding)
    denom = max(output_baseline_to_geometry_l2, 1e-8)
    geometry_improvement_ratio = (output_baseline_to_geometry_l2 - output_to_geometry_l2) / denom

    return {
        "geometry_improvement_ratio": geometry_improvement_ratio,
        "output_to_geometry_l2": output_to_geometry_l2,
        "output_baseline_to_geometry_l2": output_baseline_to_geometry_l2,
    }


def load_clip_model(clip_model: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(clip_model)
    return model, processor


def compute_metrics(args: argparse.Namespace) -> Dict[str, object]:
    image_path = resolve_path(args.image)
    geometry_path = resolve_path(args.geometry)
    texture_path = resolve_path(args.texture)
    color_path = resolve_path(args.color)
    baselines_dir = resolve_path(args.baselines_dir)
    geometry_baseline_path = resolve_path(args.geometry_baseline) if args.geometry_baseline else None
    output_baseline_path = resolve_path(args.output_baseline) if args.output_baseline else None
    if not args.no_auto_baseline:
        geometry_baseline_path = geometry_baseline_path or find_auto_geometry_baseline(geometry_path, baselines_dir)
        output_baseline_path = output_baseline_path or find_auto_output_baseline(args.prompt, baselines_dir)
    device = resolve_device(args.device)

    output_image = load_rgb(image_path)
    geometry_image = load_rgb(geometry_path)
    texture_image = load_rgb(texture_path)
    color_image = load_rgb(color_path)
    geometry_baseline_image = load_rgb(geometry_baseline_path) if geometry_baseline_path else None
    output_baseline_image = load_rgb(output_baseline_path) if output_baseline_path else None

    output_for_geometry = output_image.resize(geometry_image.size, Image.LANCZOS)
    output_sketch = extract_canny_contour(output_for_geometry)
    geometry_sketch = extract_canny_contour(geometry_image)

    metrics: Dict[str, object] = {
        "name": image_path.stem,
        "prompt": args.prompt,
        "clip_model": args.clip_model,
        "metrics_device": device,
        "output_path": str(image_path),
        "geometry": str(geometry_path),
        "texture": str(texture_path),
        "color": str(color_path),
        "geometry_baseline": str(geometry_baseline_path) if geometry_baseline_path else "",
        "output_baseline": str(output_baseline_path) if output_baseline_path else "",
        "color_histogram_distance": color_histogram_distance(output_image, color_image, bins=args.hist_bins),
        "ms_ssim_geometry": compute_ms_ssim_similarity(output_sketch, geometry_sketch, device),
        "geometry_residual_l2_distance": math.nan,
        "geometry_residual_cosine_similarity": math.nan,
        "geometry_residual_norm": math.nan,
        "output_residual_norm": math.nan,
        "geometry_improvement_ratio": math.nan,
        "output_to_geometry_l2": math.nan,
        "output_baseline_to_geometry_l2": math.nan,
        "baseline_error": "",
    }
    metrics.update(edge_overlap_metrics(output_sketch, geometry_sketch))

    if args.skip_clip:
        metrics.update(
            {
                "clip_text": math.nan,
                "clip_geometry_sketch": math.nan,
                "clip_texture_gray": math.nan,
                "clip_error": "skipped",
            }
        )
    else:
        try:
            model, processor = load_clip_model(args.clip_model, device)
            metrics["clip_text"] = clip_image_text_similarity(model, processor, device, output_image, args.prompt)
            metrics["clip_geometry_sketch"] = clip_image_image_similarity(
                model,
                processor,
                device,
                output_sketch,
                geometry_sketch,
            )
            metrics["clip_texture_gray"] = clip_image_image_similarity(
                model,
                processor,
                device,
                output_image.convert("L").convert("RGB"),
                texture_image.convert("L").convert("RGB"),
            )
            if geometry_baseline_image is not None and output_baseline_image is not None:
                metrics.update(
                    compute_geometry_residual_metrics(
                        model,
                        processor,
                        device,
                        geometry_image,
                        geometry_baseline_image,
                        output_image,
                        output_baseline_image,
                    )
                )
            else:
                missing = []
                if geometry_baseline_image is None:
                    missing.append("geometry_baseline")
                if output_baseline_image is None:
                    missing.append("output_baseline")
                metrics["baseline_error"] = "missing " + ", ".join(missing)
            if output_baseline_image is not None:
                metrics.update(
                    compute_geometry_improvement_ratio(
                        model,
                        processor,
                        device,
                        geometry_image,
                        output_image,
                        output_baseline_image,
                    )
                )
            metrics["clip_error"] = ""
        except Exception as exc:
            metrics.update(
                {
                    "clip_text": math.nan,
                    "clip_geometry_sketch": math.nan,
                    "clip_texture_gray": math.nan,
                    "clip_error": repr(exc),
                }
            )

    if args.save_debug:
        debug_dir = resolve_path(args.output_dir) / image_path.stem / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        output_sketch.save(debug_dir / "output_canny.png")
        geometry_sketch.save(debug_dir / "geometry_canny.png")
        metrics["debug_dir"] = str(debug_dir)

    return metrics


def save_reports(metrics: Dict[str, object], output_dir: Path) -> None:
    run_dir = resolve_path(output_dir) / str(metrics["name"])
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "metrics.json"
    csv_path = run_dir / "metrics.csv"

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[SAVE] JSON: {json_path}")
    print(f"[SAVE] CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SADis metrics for one generated image against geometry/color/texture references."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--texture", type=Path, default=DEFAULT_TEXTURE)
    parser.add_argument("--color", type=Path, default=DEFAULT_COLOR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--geometry-baseline", type=Path, default=None)
    parser.add_argument("--output-baseline", type=Path, default=None)
    parser.add_argument("--no-auto-baseline", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hist-bins", type=int, default=32)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(args)
    save_reports(metrics, args.output_dir)


if __name__ == "__main__":
    main()
