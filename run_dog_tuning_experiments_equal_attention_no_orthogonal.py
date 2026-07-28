import argparse
import csv
import gc
import json
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lpips
from pytorch_msssim import ms_ssim
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import CLIPModel, CLIPProcessor

from pipeline_stable_diffusion_xl_equal_attention_no_orthogonal import StableDiffusionXLPipeline
from ip_adapter_equal_attention_no_orthogonal import IPAdapterPlusXL
from util.torch_compat import ensure_supported_cuda_runtime


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TARGET_BLOCKS = ["down_blocks", "mid_block", "up_blocks"]

BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_ENCODER_PATH = "models/image_encoder"
IP_CKPT = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, letters, logo, lowres, low quality, worst quality, deformed, glitch, "
    "low contrast, noisy, blurry, copy reference image, same composition as reference, "
    "identical layout, copy texture reference, same objects as texture reference, "
    "same composition as texture reference, two children, multiple children, extra child, "
    "extra person, duplicate person, twins, crowd, church, cathedral, chapel, mosque, "
    "dull, desaturated, greyish, monochromatic, low saturation, flat lighting"
)

GEOMETRY_LUMA_TARGET_MEAN = 175.0
GEOMETRY_LUMA_MEAN_STRENGTH = 0.60
GEOMETRY_LUMA_CONTRAST = 0.65
GEOMETRY_CHROMA_STRENGTH = 0.25
GEOMETRY_EDGE_MIN_COMPONENT_AREA = 64
GEOMETRY_EDGE_MIN_COMPONENT_LENGTH = 36
GEOMETRY_EDGE_KEEP_TOP_COMPONENTS = 80
GEOMETRY_EDGE_CLOSE_KERNEL_SIZE = 3
GEOMETRY_EDGE_CLOSE_ITERATIONS = 1
GEOMETRY_EDGE_DILATE_ITERATIONS = 1
GEOMETRY_FEATURE_SHARPEN_AMOUNT = 0.28
GEOMETRY_FEATURE_SHARPEN_SIGMA = 1.10
GEOMETRY_FEATURE_NOISE_STD = 2.5
FINAL_OUTPUT_BRIGHTNESS = 1.00
FINAL_OUTPUT_SATURATION = 1.24
FINAL_OUTPUT_CONTRAST = 1.02

CATEGORY_DEFAULT_PROMPTS = {
    "animal": [
        "A single dog on the ground",
        "A single cat on the ground",
    ],
    "face": [
        "A portrait of a man with a beard",
        "A portrait of a little boy with a hat",
    ],
    "house": [
        "One traditional Chinese temple on the ground",
        "One beautiful church on the ground",
    ],
    "person": [
        "one child sitting on a sofa",
        "one tall man sitting on a sofa",
    ],
    "seen": [
        "A building of a beautiful church in a city",
        "A building of a traditional chinese temple in a forest",
    ],
}

GEOMETRY_CATEGORIES = tuple(CATEGORY_DEFAULT_PROMPTS.keys())
DEFAULT_ENABLED_GEOMETRY_CATEGORIES = ("animal", "face", "house", "seen")

SINGLE_CHILD_POSITIVE_SUFFIX = (
    "exactly one child, only one child, centered single subject, no other people, no companion"
)

SINGLE_CHILD_NEGATIVE_SUFFIX = (
    "two kids, multiple kids, second child, another child, child with friend, siblings, brother, sister, "
    "playmates, group of children, pair, duo"
)

# key or geometry.stem -> geometry baseline prompt
GEOMETRY_BASELINE_PROMPTS = {
    "animal01": "A fox in the snow.",
    "animal02": "A howling fox.",
    "animal03": "A protrait of fox.",
    "animal04": "A fox sitting on a ground",
    "animal05": "A fox sitting on a ground",
    "animal06": "A fox sitting on a ground",
    "animal07": "A fox sitting on a ground",
    "animal08": "A rabbit on a ground",
    "animal09": "A tiger on a ground",
    "animal10": "A rabbit on a ground",
    "face01": "A woman protrait",
    "face02": "A woman protrait",
    "face03": "A woman protrait",
    "face04": "A woman protrait",
    "face05": "A woman protrait",
    "face06": "A woman protrait",
    "face07": "A woman protrait",
    "face08": "A woman protrait",
    "face09": "A woman protrait",
    "face10": "A woman protrait",
    "face11": "A woman protrait",
    "house01": "A Tudor-style house with bushes.",
    "house02": "A house by a lake with a sailboat, a tree, and flowers.",
    "house03": "A house in a field with a hill.",
    "house04": "A house.",
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
    "seen03": "a path in a forest",
    "seen04": "A path and a house in a forest",
    "seen05": "The building",
    "seen06": "The building",
    "seen07": "The building",
}


@dataclass
class AssetTriple:
    key: str
    geometry_category: str
    geometry: Path
    color: Path
    texture: Path
    texture2: Optional[Path] = None


@dataclass
class RunRecord:
    triple: AssetTriple
    prompt: str
    output_path: Path


class ImageCache:
    def __init__(self) -> None:
        self._image_cache: Dict[Tuple[str, str], Image.Image] = {}
        self._geometry_cache: Dict[str, Image.Image] = {}

    @staticmethod
    def _path_key(path: Path) -> str:
        return str(Path(path).resolve())

    def _image_key(self, path: Path, mode: str) -> Tuple[str, str]:
        return (self._path_key(path), mode)

    def get_image(self, path: Path, mode: str = "RGB") -> Image.Image:
        key = self._image_key(path, mode)
        if key not in self._image_cache:
            if mode != "RGB":
                rgb_key = self._image_key(path, "RGB")
                if rgb_key in self._image_cache:
                    self._image_cache[key] = self._image_cache[rgb_key].convert(mode)
                else:
                    self._image_cache[key] = load_image(path, mode)
            else:
                self._image_cache[key] = load_image(path, mode)
        return self._image_cache[key].copy()

    def get_geometry_preprocessed(self, path: Path) -> Image.Image:
        key = self._path_key(path)
        if key not in self._geometry_cache:
            self._geometry_cache[key] = preprocess_geometry_for_rules(str(path))
        return self._geometry_cache[key].copy()

    def preload_images(
        self,
        paths: List[Path],
        mode: str = "RGB",
        workers: int = 4,
        label: str = "images",
    ) -> None:
        if workers <= 0:
            return

        unique_paths = []
        seen = set()
        for path in paths:
            if path is None:
                continue
            key = self._image_key(path, mode)
            if key in self._image_cache or key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)

        if not unique_paths:
            return

        print("[INFO] Preloading {0} {1} with {2} workers".format(len(unique_paths), label, workers))

        def load_one(path: Path) -> Tuple[Tuple[str, str], Image.Image]:
            return self._image_key(path, mode), load_image(path, mode)

        if workers == 1:
            for key, image in (load_one(path) for path in unique_paths):
                self._image_cache[key] = image
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for key, image in executor.map(load_one, unique_paths):
                self._image_cache[key] = image

    def preload_geometry_preprocessed(self, paths: List[Path], workers: int = 4) -> None:
        if workers <= 0:
            return

        unique_paths = []
        seen = set()
        for path in paths:
            if path is None:
                continue
            key = self._path_key(path)
            if key in self._geometry_cache or key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)

        if not unique_paths:
            return

        print("[INFO] Preloading {0} geometry preprocesses with {1} workers".format(len(unique_paths), workers))

        def load_one(path: Path) -> Tuple[str, Image.Image]:
            return self._path_key(path), preprocess_geometry_for_rules(str(path))

        if workers == 1:
            for key, image in (load_one(path) for path in unique_paths):
                self._geometry_cache[key] = image
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for key, image in executor.map(load_one, unique_paths):
                self._geometry_cache[key] = image

    def preload_triple_assets(self, triples: List[AssetTriple], workers: int = 4) -> None:
        geometry_paths = [triple.geometry for triple in triples]
        asset_paths = []
        for triple in triples:
            asset_paths.append(triple.geometry)
            asset_paths.append(triple.color)
            asset_paths.append(triple.texture)
            if triple.texture2 is not None:
                asset_paths.append(triple.texture2)

        self.preload_images(asset_paths, mode="RGB", workers=workers, label="asset views")
        self.preload_geometry_preprocessed(geometry_paths, workers=workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run geometry/color/texture assets and compute CLIP image-text + color histogram distance."
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--color-dir", type=Path, default=None)
    parser.add_argument("--texture-dir", type=Path, default=None)
    parser.add_argument("--pairing", choices=("auto", "stem", "cartesian"), default="cartesian")
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        type=Path,
        default=Path("results/equal_attention_ablation_no_orthogonal_equal13"),
    )
    parser.add_argument(
        "--geometry-categories",
        nargs="+",
        default=list(DEFAULT_ENABLED_GEOMETRY_CATEGORIES),
        choices=GEOMETRY_CATEGORIES,
        help="Geometry categories to include in the experiment sweep.",
    )

    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--allow-todo-baseline-prompts", action="store_true")

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=9.0)

    parser.add_argument("--color-scale", type=float, default=1.00)
    parser.add_argument("--substract-scale", type=float, default=0.00)
    parser.add_argument("--texture-scale", type=float, default=1.00)
    parser.add_argument("--texture2-scale", type=float, default=0.0)
    parser.add_argument("--geometry-scale", type=float, default=1.00)
    parser.add_argument("--geometry-sub-scale", type=float, default=0.00)
    parser.add_argument("--texture-color-decouple", type=float, default=0.00)
    parser.add_argument("--geometry-color-decouple", type=float, default=0.00)
    parser.add_argument("--color-to-geometry-decouple", type=float, default=0.00)

    parser.add_argument("--wct-guidance", type=float, default=0.28)
    parser.add_argument("--wct-starts-step-ratio", type=float, default=0.45)
    parser.add_argument("--wct-ends-step-ratio", type=float, default=0.55)
    parser.add_argument("--wctnoise-add-scale", type=float, default=0.004)
    parser.add_argument("--front-end-ratio", type=float, default=0.40)
    parser.add_argument("--mid-end-ratio", type=float, default=0.70)
    parser.add_argument("--color-stage-factors", type=float, nargs=3, default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    parser.add_argument("--geometry-stage-factors", type=float, nargs=3, default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    parser.add_argument("--texture-stage-factors", type=float, nargs=3, default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    parser.add_argument("--geometry-early-end", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--geometry-late-factor", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--texture-late-start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--texture-early-factor", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--punish-weight", type=float, default=0.0)
    parser.add_argument("--punish-type", default=None)
    parser.add_argument("--final-output-brightness", type=float, default=FINAL_OUTPUT_BRIGHTNESS)
    parser.add_argument("--final-output-saturation", type=float, default=FINAL_OUTPUT_SATURATION)
    parser.add_argument("--final-output-contrast", type=float, default=FINAL_OUTPUT_CONTRAST)

    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--hist-bins", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip image generation and compute metrics only from existing outputs in output_root/images.",
    )
    parser.add_argument(
        "--allow-partial-metrics",
        action="store_true",
        help="In metrics-only mode, compute metrics for existing outputs and warn about missing outputs.",
    )
    parser.add_argument(
        "--metrics-device",
        default=None,
        help="Device used for non-LPIPS metric models. Defaults to --device.",
    )
    parser.add_argument(
        "--lpips-device",
        default=None,
        help="Device used for LPIPS. Defaults to --metrics-device/--device.",
    )
    parser.add_argument(
        "--skip-cross-geometry-lpips",
        action="store_true",
        help="Skip the expensive pairwise LPIPS summary metric to reduce memory/time usage.",
    )
    parser.add_argument(
        "--asset-preload-workers",
        type=int,
        default=4,
        help="Number of threads used to preload reusable asset images into RAM. Set to 0 to disable.",
    )
    parser.add_argument(
        "--log-stream-schedule",
        action="store_true",
        help="Enable verbose diffusion schedule logging during sampling.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def require_exists(path: str, kind: str = "file") -> None:
    ok = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
    if not ok:
        raise FileNotFoundError("Required {0} not found: {1}".format(kind, path))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError("Missing folder: {0}".format(folder))
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def resolve_geometry_category(geometry_path: Path) -> str:
    stem = geometry_path.stem.lower()
    for category in GEOMETRY_CATEGORIES:
        if stem.startswith(category):
            return category
    raise ValueError(
        "Unsupported geometry category for {0!r}. Expected filename stem to start with one of: {1}".format(
            geometry_path.name,
            ", ".join(GEOMETRY_CATEGORIES),
        )
    )


def try_resolve_geometry_category(geometry_path: Path) -> Optional[str]:
    try:
        return resolve_geometry_category(geometry_path)
    except ValueError:
        return None


def resolve_prompts_for_triple(prompt_map: Dict[str, List[str]], triple: AssetTriple) -> List[str]:
    if triple.geometry.stem in prompt_map:
        return prompt_map[triple.geometry.stem]
    if triple.key in prompt_map:
        return prompt_map[triple.key]
    return list(CATEGORY_DEFAULT_PROMPTS[triple.geometry_category])


def build_triples(args: argparse.Namespace) -> List[AssetTriple]:
    asset_root = args.asset_root
    geometry_dir = args.geometry_dir or asset_root / "geometry"
    color_dir = args.color_dir or asset_root / "color"
    texture_dir = args.texture_dir or asset_root / "texture"
    enabled_categories = set(args.geometry_categories)

    geometries = []
    skipped_geometry_files = []
    for path in list_images(geometry_dir):
        category = try_resolve_geometry_category(path)
        if category is None:
            skipped_geometry_files.append(path.name)
            continue
        if category in enabled_categories:
            geometries.append(path)
    if skipped_geometry_files:
        print(
            "[INFO] Skipped unsupported geometry files: {0}".format(
                ", ".join(skipped_geometry_files)
            )
        )
    colors = list_images(color_dir)
    textures = list_images(texture_dir)

    def build_cartesian_triples() -> List[AssetTriple]:
        triples = []
        for g in geometries:
            for c in colors:
                for t in textures:
                    triples.append(
                        AssetTriple(
                            key="{0}__{1}__{2}".format(g.stem, c.stem, t.stem),
                            geometry_category=resolve_geometry_category(g),
                            geometry=g,
                            color=c,
                            texture=t,
                            texture2=None,
                        )
                    )
        return triples

    if args.pairing == "cartesian":
        return build_cartesian_triples()

    g_map = dict((p.stem, p) for p in geometries)
    c_map = dict((p.stem, p) for p in colors)
    t_map = dict((p.stem, p) for p in textures)
    common = sorted(set(g_map) & set(c_map) & set(t_map))
    if not common:
        if args.pairing == "auto":
            print(
                "[INFO] No matching stems across geometry/color/texture; "
                "falling back to cartesian pairing for the current assets."
            )
            return build_cartesian_triples()
        raise RuntimeError(
            "No matching filename stems across geometry/color/texture. "
            "Use --pairing cartesian if these folders should be fully combined."
        )

    return [
        AssetTriple(
            key=k,
            geometry_category=resolve_geometry_category(g_map[k]),
            geometry=g_map[k],
            color=c_map[k],
            texture=t_map[k],
            texture2=None,
        )
        for k in common
    ]


def load_prompt_map(prompt_file: Optional[Path]) -> Dict[str, List[str]]:
    if prompt_file is None:
        return {}
    with prompt_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--prompt-file must be a JSON object mapping key -> prompt or key -> [prompts]")

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[str(key)] = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            result[str(key)] = value
        else:
            raise ValueError("Invalid prompt-file entry for key {0!r}".format(key))
    return result


def append_prompt_clauses(base: str, extra: str) -> str:
    clauses = []
    seen = set()

    for part in (base, extra):
        for clause in part.split(","):
            text = clause.strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            clauses.append(text)

    return ", ".join(clauses)


def prompt_requests_single_child(prompt: str) -> bool:
    lowered = prompt.lower()
    has_child = any(token in lowered for token in ("child", "kid"))
    has_solo_intent = any(token in lowered for token in ("single", "solo", "one", "only", "exactly one"))
    return has_child and has_solo_intent


def build_effective_prompt_pair(prompt: str, negative_prompt: str) -> Tuple[str, str]:
    # Keep prompt handling aligned with infer_style_plus_color_texture.py:
    # do not inject experiment-only clauses before generation.
    return prompt, negative_prompt


def _enhance_final_output(
    img,
    brightness=FINAL_OUTPUT_BRIGHTNESS,
    saturation=FINAL_OUTPUT_SATURATION,
    contrast=FINAL_OUTPUT_CONTRAST,
):
    img = img.convert("RGB")
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def _extract_geometry_canny_edges(
    rgb,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    bilateral_d=5,
    bilateral_sigma_color=35,
    bilateral_sigma_space=35,
    canny_sigma=0.16,
):
    # Match infer_style_plus_color_texture.py and metric extract_canny_contour().
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    gray = clahe.apply(gray)
    blurred = cv2.bilateralFilter(
        gray,
        bilateral_d,
        bilateral_sigma_color,
        bilateral_sigma_space,
    )

    median_value = np.median(blurred)
    lower = int(max(0, (1.0 - canny_sigma) * median_value))
    upper = int(min(255, (1.0 + canny_sigma) * median_value))
    if upper <= lower:
        upper = min(255, lower + 1)
    return cv2.Canny(blurred, lower, upper)


def _normalize_geometry_luma_mean(
    rgb,
    target_mean=GEOMETRY_LUMA_TARGET_MEAN,
    strength=GEOMETRY_LUMA_MEAN_STRENGTH,
):
    if target_mean is None or strength <= 0:
        return rgb

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luma = lab[:, :, 0]
    current_mean = float(luma.mean())
    shift = (float(target_mean) - current_mean) * float(strength)
    lab[:, :, 0] = np.clip(luma + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _homogenize_geometry_color(
    rgb,
    target_luma=GEOMETRY_LUMA_TARGET_MEAN,
    luma_strength=GEOMETRY_LUMA_MEAN_STRENGTH,
    luma_contrast=GEOMETRY_LUMA_CONTRAST,
    chroma_strength=GEOMETRY_CHROMA_STRENGTH,
):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    luma = lab[:, :, 0]
    luma_mean = float(luma.mean())
    luma = (luma - luma_mean) * float(luma_contrast) + luma_mean
    if target_luma is not None and luma_strength > 0:
        luma += (float(target_luma) - luma_mean) * float(luma_strength)
    lab[:, :, 0] = np.clip(luma, 0, 255)

    chroma_strength = float(chroma_strength)
    if chroma_strength < 1.0:
        ab_mean = lab[:, :, 1:3].reshape(-1, 2).mean(axis=0)
        lab[:, :, 1:3] = ab_mean + (lab[:, :, 1:3] - ab_mean) * chroma_strength

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _augment_geometry_feature_rgb(
    rgb,
    noise_seed,
    sharpen_amount=GEOMETRY_FEATURE_SHARPEN_AMOUNT,
    sharpen_sigma=GEOMETRY_FEATURE_SHARPEN_SIGMA,
    noise_std=GEOMETRY_FEATURE_NOISE_STD,
):
    rgb_float = rgb.astype(np.float32)
    if sharpen_amount and sharpen_amount > 0:
        blur = cv2.GaussianBlur(rgb_float, (0, 0), sigmaX=sharpen_sigma, sigmaY=sharpen_sigma)
        rgb_float = cv2.addWeighted(rgb_float, 1.0 + sharpen_amount, blur, -sharpen_amount, 0.0)
    if noise_std and noise_std > 0:
        rng = np.random.default_rng(int(noise_seed) & 0xFFFFFFFF)
        noise = rng.normal(loc=0.0, scale=noise_std, size=rgb_float.shape).astype(np.float32)
        rgb_float = rgb_float + noise
    return np.clip(rgb_float, 0, 255).astype(np.uint8)


def _filter_small_edge_components(edge_mask, min_area=GEOMETRY_EDGE_MIN_COMPONENT_AREA):
    if min_area is None or min_area <= 0:
        return edge_mask

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    filtered = np.zeros_like(edge_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def _filter_short_edge_components(edge_mask, min_length=GEOMETRY_EDGE_MIN_COMPONENT_LENGTH):
    if min_length is None or min_length <= 0:
        return edge_mask

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    filtered = np.zeros_like(edge_mask)
    for label in range(1, num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if max(width, height) >= min_length:
            filtered[labels == label] = 255
    return filtered


def _keep_major_edge_components(edge_mask, keep_top=GEOMETRY_EDGE_KEEP_TOP_COMPONENTS):
    if keep_top is None or keep_top <= 0:
        return edge_mask

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        score = area * max(width, height)
        components.append((score, label))

    filtered = np.zeros_like(edge_mask)
    for _score, label in sorted(components, reverse=True)[:keep_top]:
        filtered[labels == label] = 255
    return filtered


def preprocess_geometry_for_rules(
    path,
    brightness_factor=1.00,
    edge_darken=0.0,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    bilateral_d=5,
    bilateral_sigma_color=35,
    bilateral_sigma_space=35,
    canny_sigma=0.16,
    edge_min_area=GEOMETRY_EDGE_MIN_COMPONENT_AREA,
    edge_min_length=GEOMETRY_EDGE_MIN_COMPONENT_LENGTH,
    edge_keep_top=GEOMETRY_EDGE_KEEP_TOP_COMPONENTS,
    close_kernel_size=GEOMETRY_EDGE_CLOSE_KERNEL_SIZE,
    close_iterations=GEOMETRY_EDGE_CLOSE_ITERATIONS,
    dilate_iterations=GEOMETRY_EDGE_DILATE_ITERATIONS,
):
    if os.path.basename(path).endswith("_geometry_preprocess.png"):
        return Image.open(path).convert("RGB")

    img = Image.open(path).convert("RGB")
    if brightness_factor != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    edge_rgb = np.array(img)
    feature_seed = zlib.crc32(os.path.abspath(path).encode("utf-8")) & 0xFFFFFFFF
    edge_rgb = _augment_geometry_feature_rgb(edge_rgb, feature_seed)
    palette_rgb = _homogenize_geometry_color(edge_rgb).astype(np.float32)

    edges = _extract_geometry_canny_edges(
        edge_rgb,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        canny_sigma=canny_sigma,
    )

    if close_iterations and close_iterations > 0:
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    edges = _filter_small_edge_components(edges, edge_min_area)
    edges = _filter_short_edge_components(edges, edge_min_length)
    edges = _keep_major_edge_components(edges, edge_keep_top)

    if dilate_iterations and dilate_iterations > 0:
        kernel = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        )
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    edge_mask = edges > 0
    fused = palette_rgb.copy()
    fused[edge_mask] = palette_rgb[edge_mask] * edge_darken

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return Image.fromarray(fused)


def compute_ms_ssim_similarity(img_a, img_b, device):
    """
    計? MS-SSIM ?似度?
    img_a, img_b: ?入已???Canny ???? PIL Image (RGB)
    """
    target_size = img_a.size

    def to_tensor(img, target_device):
        if img.size != target_size:
            img = img.resize(target_size, Image.BICUBIC)
        # 轉為 [1, 3, H, W] 且???[0, 1]
        t = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(target_device)

    t_a = to_tensor(img_a, device)
    t_b = to_tensor(img_b, device)

    # MS-SSIM ?設?要影?至?160x160 (?為??5 ??downsample ?段)
    # 返?????0~1? ???相??
    with torch.inference_mode():
        score = ms_ssim(t_a, t_b, data_range=1.0, size_average=True)
    
    return float(score.item())

def preprocess_color_palette_lowfreq(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    palette_rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return Image.fromarray(palette_rgb)


def preprocess_texture_material_gray(path):
    return Image.open(path).convert("L")


def configure_pipeline_runtime(pipe: StableDiffusionXLPipeline, device: str) -> None:
    pipe.enable_vae_tiling()

    if not str(device).startswith("cuda"):
        print("[INFO] Memory-efficient attention skipped on non-CUDA device")
        return

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] Enabled xFormers memory-efficient attention")
            return
        except Exception as exc:
            print("[INFO] xFormers attention unavailable: {0}".format(exc))

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("[INFO] Using PyTorch scaled_dot_product_attention backend (Flash kernels when supported)")
    else:
        print("[INFO] Falling back to default attention backend")


def build_ip_model(device: str) -> IPAdapterPlusXL:
    require_exists(IMAGE_ENCODER_PATH, "dir")
    require_exists(IP_CKPT, "file")
    ensure_supported_cuda_runtime(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe = pipe.to(device)
    configure_pipeline_runtime(pipe, device)

    return IPAdapterPlusXL(
        pipe,
        IMAGE_ENCODER_PATH,
        IP_CKPT,
        device,
        num_tokens=16,
        target_blocks=TARGET_BLOCKS,
    )


def build_baseline_pipe(device: str) -> StableDiffusionXLPipeline:
    ensure_supported_cuda_runtime(device)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe = pipe.to(device)
    configure_pipeline_runtime(pipe, device)
    return pipe


def safe_prompt_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)[:40]


def _is_todo_baseline_prompt(prompt: str) -> bool:
    return prompt.strip().lower().startswith("todo:")


def resolve_geometry_baseline_prompt(
    triple: AssetTriple,
    fallback_prompt: Optional[str] = None,
    allow_todo: bool = False,
) -> str:
    if triple.key in GEOMETRY_BASELINE_PROMPTS:
        prompt = GEOMETRY_BASELINE_PROMPTS[triple.key]
    elif triple.geometry.stem in GEOMETRY_BASELINE_PROMPTS:
        prompt = GEOMETRY_BASELINE_PROMPTS[triple.geometry.stem]
    elif fallback_prompt is not None:
        prompt = fallback_prompt
    else:
        raise KeyError(
            "Missing geometry baseline prompt for key={0!r}, geometry_stem={1!r}. "
            "Please add it to GEOMETRY_BASELINE_PROMPTS.".format(triple.key, triple.geometry.stem)
        )

    if _is_todo_baseline_prompt(prompt) and not allow_todo:
        raise ValueError(
            "Geometry baseline prompt is still TODO for geometry_stem={0!r}. "
            "Fill GEOMETRY_BASELINE_PROMPTS or rerun with --allow-todo-baseline-prompts for a dry run.".format(
                triple.geometry.stem
            )
        )
    return prompt


def validate_geometry_baseline_prompts(triples: List[AssetTriple]) -> None:
    missing = []
    todo = []
    for triple in triples:
        try:
            prompt = resolve_geometry_baseline_prompt(triple, allow_todo=True)
            if _is_todo_baseline_prompt(prompt):
                todo.append(triple.geometry.stem)
        except KeyError:
            missing.append((triple.key, triple.geometry.stem))
    if missing:
        seen = set()
        labels = []
        for key, stem in missing:
            if stem in seen:
                continue
            seen.add(stem)
            labels.append("{0!r} (stem={1!r})".format(key, stem))
        print(
            "[WARN] Missing geometry baseline prompts for: {0}. "
            "Geometry baseline metrics will fall back to each run's output prompt.".format(", ".join(labels))
        )
    if todo:
        labels = sorted(set(todo))
        print(
            "[WARN] TODO geometry baseline prompts remain for: {0}. "
            "Baseline/metrics will fail unless --allow-todo-baseline-prompts is set.".format(", ".join(labels))
        )


def make_torch_generator(seed: int, device: str) -> torch.Generator:
    if str(device).startswith("cuda"):
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def generate_baseline_image(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    args: argparse.Namespace,
) -> Image.Image:
    effective_prompt, effective_negative_prompt = build_effective_prompt_pair(prompt, args.negative_prompt)
    return pipe(
        prompt=effective_prompt,
        negative_prompt=effective_negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=make_torch_generator(args.seed, args.device),
        wct_guidance=0.0,
        wct_starts_step=0,
        wct_ends_step=0,
        wctnoise_add_scale=0.0,
        log_stream_schedule=args.log_stream_schedule,
        save_name="baseline",
    ).images[0]


def make_baseline_path(
    baselines_dir: Path,
    triple: AssetTriple,
    prompt_type: str,
    prompt: str,
) -> Path:
    return baselines_dir / "{0}_{1}_{2}.png".format(triple.key, prompt_type, safe_prompt_name(prompt))


def make_shared_geometry_baseline_path(
    baselines_dir: Path,
    triple: AssetTriple,
    prompt: str,
) -> Path:
    return baselines_dir / "{0}_geometry_{1}.png".format(triple.geometry.stem, safe_prompt_name(prompt))


def make_shared_output_baseline_path(
    baselines_dir: Path,
    prompt: str,
) -> Path:
    return baselines_dir / "output_{0}.png".format(safe_prompt_name(prompt))


def make_canny_debug_paths(debug_dir: Path, record: RunRecord) -> Dict[str, Path]:
    canny_dir = debug_dir / "canny"
    prompt_tag = safe_prompt_name(record.prompt)
    base_name = "{0}_{1}".format(record.triple.key, prompt_tag)
    return {
        "geometry": canny_dir / "{0}_geometry_canny.png".format(base_name),
        "output": canny_dir / "{0}_output_canny.png".format(base_name),
        "geometry_baseline": canny_dir / "{0}_geometry_baseline_canny.png".format(base_name),
        "output_baseline": canny_dir / "{0}_output_baseline_canny.png".format(base_name),
    }


def generate_one(
    ip_model: IPAdapterPlusXL,
    triple: AssetTriple,
    prompt: str,
    args: argparse.Namespace,
    image_cache: Optional[ImageCache] = None,
) -> Image.Image:
    effective_prompt, effective_negative_prompt = build_effective_prompt_pair(prompt, args.negative_prompt)
    if image_cache is None:
        geometry_ref_img = preprocess_geometry_for_rules(str(triple.geometry))
        color_ref_img = load_image(triple.color, "RGB")
        texture_ref_img = load_image(triple.texture, "RGB")
        texture_ref_img2 = load_image(triple.texture2, "RGB") if triple.texture2 else None
    else:
        geometry_ref_img = image_cache.get_geometry_preprocessed(triple.geometry)
        color_ref_img = image_cache.get_image(triple.color, "RGB")
        texture_ref_img = image_cache.get_image(triple.texture, "RGB")
        texture_ref_img2 = image_cache.get_image(triple.texture2, "RGB") if triple.texture2 else None

    color_image_gray = color_ref_img.convert("L")
    texture_image_gray = texture_ref_img.convert("L")
    color_stage_factors = tuple(float(x) for x in args.color_stage_factors)
    geometry_stage_factors = tuple(float(x) for x in args.geometry_stage_factors)
    texture_stage_factors = tuple(float(x) for x in args.texture_stage_factors)

    images = ip_model.generate(
        prompt=effective_prompt,
        negative_prompt=effective_negative_prompt,
        scale=args.scale,
        guidance_scale=args.guidance_scale,
        clr_ref_img=color_ref_img,
        clr_texture_ref_img=color_image_gray,
        texture_ref_img=texture_image_gray,
        geometry_ref_img=geometry_ref_img,
        texture_ref_img2=texture_ref_img2,
        clr_ref_img_dir=str(triple.color),
        sty_ref_img_dir=str(triple.texture),
        color_scale=args.color_scale,
        substract_scale=args.substract_scale,
        texture_scale=args.texture_scale,
        texture2_scale=args.texture2_scale,
        geometry_scale=args.geometry_scale,
        geometry_sub_scale=args.geometry_sub_scale,
        geometry_color_decouple=args.geometry_color_decouple,
        texture_color_decouple=args.texture_color_decouple,
        color_to_geometry_decouple=args.color_to_geometry_decouple,
        num_samples=args.num_samples,
        num_inference_steps=args.steps,
        seed=args.seed,
        front_end_ratio=args.front_end_ratio,
        mid_end_ratio=args.mid_end_ratio,
        color_stage_factors=color_stage_factors,
        geometry_stage_factors=geometry_stage_factors,
        texture_stage_factors=texture_stage_factors,
        log_stream_schedule=args.log_stream_schedule,
        wct_guidance=args.wct_guidance,
        wct_starts_step=args.wct_starts_step_ratio * args.steps,
        wct_ends_step=args.wct_ends_step_ratio * args.steps,
        wctnoise_add_scale=args.wctnoise_add_scale,
        punish_weight=args.punish_weight,
        punish_type=args.punish_type,
        save_name=prompt,
    )
    return _enhance_final_output(
        images[0],
        brightness=args.final_output_brightness,
        saturation=args.final_output_saturation,
        contrast=args.final_output_contrast,
    )


def load_rgb(path: Path) -> Image.Image:
    return load_image(path, "RGB")


def load_image(path: Path, mode: str = "RGB") -> Image.Image:
    with Image.open(path) as img:
        return img.convert(mode)


def clip_image_text_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image: Image.Image,
    text: str,
) -> float:
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float(torch.sum(image_features[0] * text_features[0]).item())


def clip_image_image_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image_a: Image.Image,
    image_b: Image.Image,
) -> float:
    with torch.no_grad():
        inputs_a = processor(images=image_a, return_tensors="pt").to(device)
        inputs_b = processor(images=image_b, return_tensors="pt").to(device)

        features_a = model.get_image_features(pixel_values=inputs_a["pixel_values"])
        features_b = model.get_image_features(pixel_values=inputs_b["pixel_values"])

        features_a = features_a / features_a.norm(dim=-1, keepdim=True)
        features_b = features_b / features_b.norm(dim=-1, keepdim=True)
        return float(torch.sum(features_a[0] * features_b[0]).item())


def extract_canny_contour(image: Image.Image, name: Optional[str] = None, debug_dir: Optional[Path] = None) -> Image.Image:
    """強?細??? Canny ??：使??CLAHE 增強細微幾?線?"""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # 使用 CLAHE 增強對?度?????微?線?
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 輕微?噪但????
    blurred = cv2.bilateralFilter(gray, 5, 35, 35)
    
    # ?適???
    v = np.median(blurred)
    sigma = 0.25 # 縮? sigma 以??更多??
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(blurred, lower, upper)
    
    # 不?使用 dilate，???條?幾?純粹??
    if debug_dir is not None and name:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"canny_{name}.png"), edges)

    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def cosine_similarity_from_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """計??個???餘弦?似?""
    return float(torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())

def sketchify_for_clip_geometry(
    image: Image.Image,
    name: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> Image.Image:
    return extract_canny_contour(image, name=name, debug_dir=debug_dir)


# def compute_lpips_similarity(lpips_model, img_a, img_b, device):
#     """
#     計? LPIPS ?似?(1 - Distance)
#     ?入??縮放?輯以避??Tensor Size Mismatch
#     """
#     # ????大?（以 img_a ??，??固定為 512?
#     target_size = img_a.size # (width, height)

#     def to_tensor(img):
#         # 如?大?不???強制縮放
#         if img.size != target_size:
#             img = img.resize(target_size, Image.BICUBIC)
            
#         img = np.array(img.convert("RGB"))
#         img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # [0, 1]
#         img = (img * 2.0) - 1.0 # [-1, 1]
#         return img.unsqueeze(0).to(device)

#     tensor_a = to_tensor(img_a)
#     tensor_b = to_tensor(img_b)

#     with torch.no_grad():
#         # ?在 tensor_a ??tensor_b ?尺寸?定?一?
#         dist = lpips_model(tensor_a, tensor_b).item()
    
#     return max(0.0, 1.0 - dist)


def cosine_similarity_from_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return float(torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())


def sketchify_for_clip_geometry(
    image: Image.Image,
    name: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> Image.Image:
    return extract_canny_contour(image, name=name, debug_dir=debug_dir)


def _module_device(module: torch.nn.Module, fallback: str) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _resolve_metric_device(explicit_device: Optional[str], fallback_device: str) -> str:
    return explicit_device if explicit_device is not None else fallback_device


def _maybe_empty_cuda_cache(*devices: Optional[str]) -> None:
    if not torch.cuda.is_available():
        return
    for device in devices:
        if device and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            return


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cudnn_status_alloc_failed" in message


def compute_lpips_similarity(lpips_model, img_a, img_b, device):
    """
    計? LPIPS ?似?(1 - Distance)
    要??入??img_a ??img_b 尺寸必?已?一??(建議統???Asset ??512)
    """
    # 檢查尺寸，???一???錯，??????????
    if img_a.size != img_b.size:
        # 實?上建議在 compute_metrics 裡就??output_image resize ??asset ?大?
        img_a = img_a.resize(img_b.size, Image.LANCZOS) 

    def to_tensor(img, target_device):
        # 轉為 RGB 確? 3 ??
        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb).astype(np.float32)
        
        # [H, W, C] -> [C, H, W] 並?射到 [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = (img_tensor / 127.5) - 1.0
        return img_tensor.unsqueeze(0).to(target_device)

    run_device = _module_device(lpips_model, fallback=device)
    t_a = None
    t_b = None

    try:
        t_a = to_tensor(img_a, run_device)
        t_b = to_tensor(img_b, run_device)

        with torch.inference_mode():
            dist = float(lpips_model(t_a, t_b).item())
    except RuntimeError as exc:
        if run_device.type != "cuda" or not _is_cuda_oom_error(exc):
            raise

        if not getattr(lpips_model, "_oom_fallback_warned", False):
            print("[WARN] LPIPS hit CUDA OOM. Falling back to CPU for the remaining LPIPS metrics.")
            setattr(lpips_model, "_oom_fallback_warned", True)

        del t_a
        del t_b
        t_a = None
        t_b = None
        gc.collect()
        torch.cuda.empty_cache()

        lpips_model.to("cpu")
        t_a = to_tensor(img_a, torch.device("cpu"))
        t_b = to_tensor(img_b, torch.device("cpu"))
        with torch.inference_mode():
            dist = float(lpips_model(t_a, t_b).item())
    finally:
        del t_a
        del t_b
    
    return max(0.0, 1.0 - dist)

def clip_image_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image: Image.Image,
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().float().cpu()


def clip_image_embedding_raw(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image: Image.Image,
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        return features[0].detach().float().cpu()


def l2_distance_from_vectors(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.norm(vec_a - vec_b, p=2).item())


def tensor_l2_norm(vec: torch.Tensor) -> float:
    return float(torch.norm(vec, p=2).item())


def count_canny_nonzero_pixels(image: Image.Image) -> int:
    arr = np.array(image.convert("L"))
    return int(np.count_nonzero(arr))


def color_histogram_distance(image_a: Image.Image, image_b: Image.Image, bins: int = 32) -> float:
    arr_a = np.array(image_a.convert("RGB"))
    arr_b = np.array(image_b.convert("RGB"))

    hist_a = cv2.calcHist([arr_a], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([arr_b], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    hist_a = cv2.normalize(hist_a, hist_a, norm_type=cv2.NORM_L1).flatten().astype(np.float32)
    hist_b = cv2.normalize(hist_b, hist_b, norm_type=cv2.NORM_L1).flatten().astype(np.float32)

    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def safe_mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def ensure_baseline_images(
    args: argparse.Namespace,
    records: List[RunRecord],
    baselines_dir: Path,
) -> Dict[str, dict]:
    baselines_dir.mkdir(parents=True, exist_ok=True)
    baseline_pipe = None
    metadata = {}

    try:
        for record in records:
            triple = record.triple
            output_baseline_prompt = record.prompt
            geometry_baseline_prompt = resolve_geometry_baseline_prompt(
                triple,
                fallback_prompt=output_baseline_prompt,
                allow_todo=args.allow_todo_baseline_prompts,
            )

            geometry_baseline_path = make_shared_geometry_baseline_path(
                baselines_dir,
                triple,
                geometry_baseline_prompt,
            )
            output_baseline_path = make_shared_output_baseline_path(
                baselines_dir,
                output_baseline_prompt,
            )

            metadata[str(record.output_path)] = {
                "geometry_baseline_prompt": geometry_baseline_prompt,
                "output_baseline_prompt": output_baseline_prompt,
                "geometry_baseline_path": geometry_baseline_path,
                "output_baseline_path": output_baseline_path,
            }

            if geometry_baseline_path.exists() and output_baseline_path.exists():
                continue

            if baseline_pipe is None:
                baseline_pipe = build_baseline_pipe(args.device)

            if not geometry_baseline_path.exists():
                generate_baseline_image(baseline_pipe, geometry_baseline_prompt, args).save(geometry_baseline_path)

            if not output_baseline_path.exists():
                generate_baseline_image(baseline_pipe, output_baseline_prompt, args).save(output_baseline_path)
    finally:
        if baseline_pipe is not None:
            del baseline_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return metadata


def save_canny_debug_images(
    record: RunRecord,
    debug_dir: Path,
    geometry_image: Image.Image,
    geometry_baseline_image: Image.Image,
    output_image: Image.Image,
    output_baseline_image: Image.Image,
) -> Dict[str, Path]:
    canny_paths = make_canny_debug_paths(debug_dir, record)
    canny_paths["geometry"].parent.mkdir(parents=True, exist_ok=True)

    extract_canny_contour(geometry_image).save(canny_paths["geometry"])
    extract_canny_contour(output_image).save(canny_paths["output"])
    extract_canny_contour(geometry_baseline_image).save(canny_paths["geometry_baseline"])
    extract_canny_contour(output_baseline_image).save(canny_paths["output_baseline"])
    return canny_paths


def compute_cross_geometry_consistency_sim_lpips(records: List[RunRecord], lpips_model, device):
    """
    使用 LPIPS 計?跨幾何???
    比??相?風?、??幾何】產??結??Canny ?似度?
    ?似度?低??幾?治??強??（控??）?強?
    """
    # 1. ??(Color, Texture, Prompt) ??
    groups = {}
    for r in records:
        # ?裡??stem 確????風???被歸在???
        group_key = f"{r.triple.color.stem}_{r.triple.texture.stem}_{safe_prompt_name(r.prompt)}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(r)
    
    similarities = []
    
    # 2. ??計??兩之???LPIPS ?似?
    for key, group_recs in groups.items():
        if len(group_recs) < 2:
            continue
            
        for i in range(len(group_recs)):
            for j in range(i + 1, len(group_recs)):
                # ??輸出??
                img_i_raw = load_rgb(group_recs[i].output_path)
                img_j_raw = load_rgb(group_recs[j].output_path)
                
                # 統?縮???512 以抹???干??(?設 Asset ??512)
                target_size = (512, 512) 
                img_i = img_i_raw.resize(target_size, Image.LANCZOS)
                img_j = img_j_raw.resize(target_size, Image.LANCZOS)
                
                # ?? Canny
                canny_i = sketchify_for_clip_geometry(img_i)
                canny_j = sketchify_for_clip_geometry(img_j)
                
                # 計? LPIPS ?似?
                sim = compute_lpips_similarity(lpips_model, canny_i, canny_j, device)
                similarities.append(sim)
                del img_i_raw
                del img_j_raw
                del img_i
                del img_j
                del canny_i
                del canny_j
                if len(similarities) % 16 == 0:
                    gc.collect()
                
    return safe_mean(similarities)

def compute_geometry_residual_metrics(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    geometry_image: Image.Image,
    geometry_baseline_image: Image.Image,
    output_image: Image.Image,
    output_baseline_image: Image.Image,
) -> Dict[str, float]:
    """計?殘差??L2 距離??Cosine Similarity"""
    g_canny = extract_canny_contour(geometry_image)
    gb_canny = extract_canny_contour(geometry_baseline_image)
    o_canny = extract_canny_contour(output_image)
    ob_canny = extract_canny_contour(output_baseline_image)

    # ?? Embedding
    g_emb = clip_image_embedding_raw(model, processor, device, g_canny)
    gb_emb = clip_image_embedding_raw(model, processor, device, gb_canny)
    o_emb = clip_image_embedding_raw(model, processor, device, o_canny)
    ob_emb = clip_image_embedding_raw(model, processor, device, ob_canny)

    # 計?殘差??
    g_res = g_emb - gb_emb
    o_res = o_emb - ob_emb

    return {
        "geometry_residual_l2_distance": l2_distance_from_vectors(g_res, o_res),
        "geometry_residual_cosine_similarity": cosine_similarity_from_vectors(g_res, o_res),
        "geometry_residual_norm": tensor_l2_norm(g_res),
        "output_residual_norm": tensor_l2_norm(o_res),
    }

def compute_cross_geometry_diversity(records: List[RunRecord], model, processor, device):
    """
    計? Cross-Geometry Diversity Score:
    比??? Color/Texture 下?不? Geometry ?出??Canny ?似度?
    ?似度?低?證? Governance ?強??越強??
    """
    # ??(Color, Texture, Prompt) ??
    groups = {}
    for r in records:
        group_key = f"{r.triple.color.stem}_{r.triple.texture.stem}_{safe_prompt_name(r.prompt)}"
        if group_key not in groups: groups[group_key] = []
        groups[group_key].append(r)
    
    similarities = []
    for key, group_recs in groups.items():
        if len(group_recs) < 2: continue
        # ?兩比?不? Geometry ??Output Canny
        for i in range(len(group_recs)):
            for j in range(i + 1, len(group_recs)):
                img_i = extract_canny_contour(load_rgb(group_recs[i].output_path))
                img_j = extract_canny_contour(load_rgb(group_recs[j].output_path))
                sim = clip_image_image_similarity(model, processor, device, img_i, img_j)
                similarities.append(sim)
                
    return safe_mean(similarities)

def compute_cross_geometry_diversity_lpips(records: List[RunRecord], lpips_model, device):
    """
    使用 LPIPS 計? Cross-Geometry Diversity Score:
    比??? Color/Texture 下?不? Geometry ?出??Canny ?似度?
    ?似度?低?證? Governance ?強??越強??
    """
    # 1. ??(Color, Texture, Prompt) ??
    groups = {}
    for r in records:
        # ?裡??stem 確????????被歸在???
        group_key = f"{r.triple.color.stem}_{r.triple.texture.stem}_{safe_prompt_name(r.prompt)}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(r)
    
    similarities = []
    
    # 2. ??計??兩之???LPIPS ?似?
    for key, group_recs in groups.items():
        if len(group_recs) < 2:
            continue
            
        for i in range(len(group_recs)):
            for j in range(i + 1, len(group_recs)):
                img_i = extract_canny_contour(load_rgb(group_recs[i].output_path))
                img_j = extract_canny_contour(load_rgb(group_recs[j].output_path))
                
                # 調用?們??寫好? LPIPS ?似度函?(?含 resize ?輯)
                sim = compute_lpips_similarity(lpips_model, img_i, img_j, device)
                similarities.append(sim)
                
    return safe_mean(similarities)

def compute_geometry_improvement_ratio(
    model: CLIPModel,
    processor: CLIPProcessor,
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






# def compute_metrics(
#     args: argparse.Namespace,
#     records: List[RunRecord],
#     baselines_dir: Path,
# ) -> tuple:
#     baseline_metadata = ensure_baseline_images(args, records, baselines_dir)
#     debug_dir = args.output_root / "debug"
#     device = args.device

#     model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
#     model.eval()
#     processor = CLIPProcessor.from_pretrained(args.clip_model)

#     lpips_fn = lpips.LPIPS(net='vgg').to(args.device)
#     lpips_fn.eval()

#     rows = []
#     # ????清單
#     clip_text_scores = []
#     clip_texture_gray_scores = []
#     clip_geometry_sketch_scores = []
#     lpips_geometry_sketch_scores = []
#     color_hist_scores = []
    
#     # 幾?殘差??清單 (L2 & Cosine)
#     geometry_residual_l2_distances = []
#     geometry_residual_cosines = []
#     geometry_improvement_ratios = []

#     for record in records:
#         triple = record.triple
#         prompt = record.prompt
#         output_path = record.output_path
#         baseline_info = baseline_metadata[str(output_path)]

#         # 1. 載入??影??
#         output_image = load_rgb(output_path)
#         geometry_image = load_rgb(triple.geometry)
#         color_image = load_rgb(triple.color)
#         texture_image = load_rgb(triple.texture)
#         geometry_baseline_image = load_rgb(baseline_info["geometry_baseline_path"])
#         output_baseline_image = load_rgb(baseline_info["output_baseline_path"])

#         # 2. 幾??徵?? (使用強???Canny)
#         output_sketch_rgb = sketchify_for_clip_geometry(output_image)
#         geometry_sketch_rgb = sketchify_for_clip_geometry(geometry_image)
        
#         # 3. ?? Debug Canny ??
#         canny_paths = save_canny_debug_images(
#             record, debug_dir, geometry_image, geometry_baseline_image, 
#             output_image, output_baseline_image
#         )

#         # 4. 計?標??似?
#         clip_text = clip_image_text_similarity(model, processor, args.device, output_image, prompt)
#         clip_geometry_sketch = clip_image_image_similarity(
#             model, processor, args.device, output_sketch_rgb, geometry_sketch_rgb
#         )

#         lpips_geo_sim = compute_lpips_similarity(lpips_fn, output_sketch_rgb, geometry_sketch_rgb, device)
        
#         clip_texture_gray = clip_image_image_similarity(
#             model, processor, args.device, 
#             output_image.convert("L").convert("RGB"), 
#             texture_image.convert("L").convert("RGB")
#         )
#         color_hist_dist = color_histogram_distance(output_image, color_image, bins=args.hist_bins)

#         # 5. [????] 計?幾?殘差 (L2 + Cosine)
#         # ?裡?設你已經?義? compute_geometry_residual_metrics ??
#         # 如???定義，??考????? auxiliary ??
#         res_metrics = compute_geometry_residual_metrics(
#             model, processor, args.device, 
#             geometry_image, geometry_baseline_image, 
#             output_image, output_baseline_image
#         )
        
#         # 6. 計??進?
#         imp_metrics = compute_geometry_improvement_ratio(
#             model, processor, args.device, geometry_image, output_image, output_baseline_image
#         )

#         # ????
#         clip_text_scores.append(clip_text)
#         lpips_geometry_sketch_scores.append(lpips_geo_sim)
#         clip_geometry_sketch_scores.append(clip_geometry_sketch)
#         clip_texture_gray_scores.append(clip_texture_gray)
#         color_hist_scores.append(color_hist_dist)
#         geometry_residual_l2_distances.append(res_metrics["geometry_residual_l2_distance"])
#         geometry_residual_cosines.append(res_metrics["geometry_residual_cosine_similarity"])
#         geometry_improvement_ratios.append(imp_metrics["geometry_improvement_ratio"])

#         # 記???rows
#         rows.append({
#             "key": triple.key,
#             "prompt": prompt,
#             "clip_text": clip_text,
#             "clip_geometry_sketch": clip_geometry_sketch,
#             "lpips_geometry_sketch": lpips_geo_sim,
#             "clip_texture_gray": clip_texture_gray,
#             "color_histogram_distance": color_hist_dist,
#             "geometry_residual_l2": res_metrics["geometry_residual_l2_distance"],
#             "geometry_residual_cosine": res_metrics["geometry_residual_cosine_similarity"],
#             "geometry_improvement_ratio": imp_metrics["geometry_improvement_ratio"],
#             "output_path": str(output_path)
#         })

#     # 7. [??] 跨幾何?????(??風格下??幾何?差異)
#     cross_geo_sim = compute_cross_geometry_diversity_lpips(records, lpips_fn, device)
#     summary = {
#         "num_runs": len(rows),
#         "clip_text_mean": safe_mean(clip_text_scores),
#         "clip_geometry_sketch_mean": safe_mean(clip_geometry_sketch_scores),
#         "lpips_geometry_sketch_mean": safe_mean(lpips_geometry_sketch_scores), # ?新 Summary Key
#         "clip_texture_gray_mean": safe_mean(clip_texture_gray_scores),
#         "color_histogram_distance_mean": safe_mean(color_hist_scores),
#         "geometry_residual_l2_mean": safe_mean(geometry_residual_l2_distances),
#         "geometry_residual_cosine_mean": safe_mean(geometry_residual_cosines),
#         "geometry_improvement_ratio_mean": safe_mean(geometry_improvement_ratios),
#         "cross_geometry_consistency_sim": cross_geo_sim,
#         "cross_geometry_note": "Lower is better. Measures if the model produces distinct structures for different geometry inputs.",
#     }
#     return rows, summary

# def compute_metrics(
#     args: argparse.Namespace,
#     records: List[RunRecord],
#     baselines_dir: Path,
# ) -> tuple:
#     baseline_metadata = ensure_baseline_images(args, records, baselines_dir)
#     debug_dir = args.output_root / "debug"
#     device = args.device

#     # ????CLIP
#     model = CLIPModel.from_pretrained(args.clip_model).to(device)
#     model.eval()
#     processor = CLIPProcessor.from_pretrained(args.clip_model)

#     # ????LPIPS
#     lpips_fn = lpips.LPIPS(net='vgg').to(device)
#     lpips_fn.eval()

#     rows = []
#     # ??容器 (?? MS-SSIM ??Edge Recall)
#     clip_text_scores = []
#     clip_texture_gray_scores = []
#     clip_geometry_sketch_scores = []
#     lpips_geometry_sketch_scores = []
#     ms_ssim_scores = []    # ??
#     edge_recall_scores = [] # ??
    
#     color_hist_scores = []
#     geometry_residual_l2_distances = []
#     geometry_residual_cosines = []
#     geometry_improvement_ratios = []

#     for record in records:
#         if not record.output_path.exists():
#             print(f"[SKIP METRIC] File not found: {record.output_path}")
#             continue

#         triple = record.triple
#         prompt = record.prompt
#         output_path = record.output_path
#         baseline_info = baseline_metadata[str(output_path)]

#         # 1. 載入影?
#         output_image = load_rgb(output_path)
#         geometry_image = load_rgb(triple.geometry)
#         color_image = load_rgb(triple.color)
#         texture_image = load_rgb(triple.texture)
#         geometry_baseline_image = load_rgb(baseline_info["geometry_baseline_path"])
#         output_baseline_image = load_rgb(baseline_info["output_baseline_path"])

#         # 2. ?? Canny ?徵 (?場??，確?Key 一??
#         output_sketch_rgb = sketchify_for_clip_geometry(output_image)
#         geometry_sketch_rgb = sketchify_for_clip_geometry(geometry_image)
        
#         # 3. ?? Debug ??
#         canny_paths = save_canny_debug_images(
#             record, debug_dir, geometry_image, geometry_baseline_image, 
#             output_image, output_baseline_image
#         )

#         # 4. ??計?
        
#         # A. LPIPS (?知?似?
#         lpips_geo_sim = compute_lpips_similarity(lpips_fn, output_sketch_rgb, geometry_sketch_rgb, device)
        
#         # B. MS-SSIM (結??似?- ?你??? Shape Mismatch)
#         msssim_score = compute_ms_ssim_similarity(output_sketch_rgb, geometry_sketch_rgb, device)

#         # C. Edge Recall (幾?????- ? (1024,1024) vs (512,512) ?錯)
#         arr_out = np.array(output_sketch_rgb.convert("L"))
#         # ?鍵修正：確?geometry 尺寸??output 完全一??
#         if geometry_sketch_rgb.size != output_sketch_rgb.size:
#             temp_geo = geometry_sketch_rgb.resize(output_sketch_rgb.size, Image.NEAREST)
#             arr_geo = np.array(temp_geo.convert("L"))
#         else:
#             arr_geo = np.array(geometry_sketch_rgb.convert("L"))
            
#         mask_out = arr_out > 127
#         mask_geo = arr_geo > 127
#         recall = np.sum(mask_out & mask_geo) / (np.sum(mask_geo) + 1e-8)

#         # D. CLIP ??
#         clip_text = clip_image_text_similarity(model, processor, device, output_image, prompt)
#         clip_geometry_sketch = clip_image_image_similarity(model, processor, device, output_sketch_rgb, geometry_sketch_rgb)
#         clip_texture_gray = clip_image_image_similarity(
#             model, processor, device, 
#             output_image.convert("L").convert("RGB"), 
#             texture_image.convert("L").convert("RGB")
#         )
        
#         color_hist_dist = color_histogram_distance(output_image, color_image, bins=args.hist_bins)

#         # 5. 幾?殘差?改??
#         res_metrics = compute_geometry_residual_metrics(model, processor, device, geometry_image, geometry_baseline_image, output_image, output_baseline_image)
#         imp_metrics = compute_geometry_improvement_ratio(model, processor, device, geometry_image, output_image, output_baseline_image)

#         # ????
#         clip_text_scores.append(clip_text)
#         clip_geometry_sketch_scores.append(clip_geometry_sketch)
#         lpips_geometry_sketch_scores.append(lpips_geo_sim)
#         ms_ssim_scores.append(msssim_score)
#         edge_recall_scores.append(recall)
#         clip_texture_gray_scores.append(clip_texture_gray)
#         color_hist_scores.append(color_hist_dist)
#         geometry_residual_l2_distances.append(res_metrics["geometry_residual_l2_distance"])
#         geometry_residual_cosines.append(res_metrics["geometry_residual_cosine_similarity"])
#         geometry_improvement_ratios.append(imp_metrics["geometry_improvement_ratio"])

#         # 寫入 CSV ?
#         rows.append({
#             "key": triple.key,
#             "prompt": prompt,
#             "clip_text": clip_text,
#             "lpips_geometry_sketch": lpips_geo_sim,
#             "ms_ssim_geometry": msssim_score,
#             "edge_recall_geometry": recall,
#             "clip_texture_gray": clip_texture_gray,
#             "color_histogram_distance": color_hist_dist,
#             "geometry_residual_l2": res_metrics["geometry_residual_l2_distance"],
#             "geometry_residual_cosine": res_metrics["geometry_residual_cosine_similarity"],
#             "geometry_improvement_ratio": imp_metrics["geometry_improvement_ratio"],
#             "output_path": str(output_path)
#         })

#     # 7. 跨幾何???
#     cross_geo_sim = compute_cross_geometry_diversity_lpips(records, lpips_fn, device)

#     summary = {
#         "num_runs": len(rows),
#         "clip_text_mean": safe_mean(clip_text_scores),
#         "lpips_geometry_sketch_mean": safe_mean(lpips_geometry_sketch_scores),
#         "ms_ssim_geometry_mean": safe_mean(ms_ssim_scores),
#         "edge_recall_geometry_mean": safe_mean(edge_recall_scores),
#         "clip_texture_gray_mean": safe_mean(clip_texture_gray_scores),
#         "color_histogram_distance_mean": safe_mean(color_hist_scores),
#         "geometry_residual_l2_mean": safe_mean(geometry_residual_l2_distances),
#         "geometry_residual_cosine_mean": safe_mean(geometry_residual_cosines),
#         "geometry_improvement_ratio_mean": safe_mean(geometry_improvement_ratios),
#         "cross_geometry_consistency_sim_lpips": cross_geo_sim,
#     }
    
#     return rows, summary

def compute_metrics(
    args: argparse.Namespace,
    records: List[RunRecord],
    baselines_dir: Path,
    image_cache: Optional[ImageCache] = None,
) -> tuple:
    baseline_metadata = ensure_baseline_images(args, records, baselines_dir)
    debug_dir = args.output_root / "debug"
    metrics_device = _resolve_metric_device(args.metrics_device, args.device)
    if image_cache is not None and args.asset_preload_workers > 0:
        baseline_paths = []
        for info in baseline_metadata.values():
            baseline_paths.append(info["geometry_baseline_path"])
            baseline_paths.append(info["output_baseline_path"])
        image_cache.preload_images(
            baseline_paths,
            mode="RGB",
            workers=args.asset_preload_workers,
            label="baseline references",
        )

    def load_metric_images(record: RunRecord):
        triple = record.triple
        baseline_info = baseline_metadata[str(record.output_path)]
        output_image_raw = load_rgb(record.output_path)
        if image_cache is None:
            geometry_image = load_rgb(triple.geometry)
            color_image = load_rgb(triple.color)
            texture_image = load_rgb(triple.texture)
            geometry_baseline_image = load_rgb(baseline_info["geometry_baseline_path"])
            output_baseline_image = load_rgb(baseline_info["output_baseline_path"])
        else:
            geometry_image = image_cache.get_image(triple.geometry, "RGB")
            color_image = image_cache.get_image(triple.color, "RGB")
            texture_image = image_cache.get_image(triple.texture, "RGB")
            geometry_baseline_image = image_cache.get_image(baseline_info["geometry_baseline_path"], "RGB")
            output_baseline_image = image_cache.get_image(baseline_info["output_baseline_path"], "RGB")

        target_size = geometry_image.size
        output_image = output_image_raw.resize(target_size, Image.LANCZOS)
        output_sketch_rgb = sketchify_for_clip_geometry(output_image)
        geometry_sketch_rgb = sketchify_for_clip_geometry(geometry_image)
        return (
            triple,
            output_image_raw,
            geometry_image,
            color_image,
            texture_image,
            geometry_baseline_image,
            output_baseline_image,
            output_image,
            output_sketch_rgb,
            geometry_sketch_rgb,
        )

    rows = []
    clip_text_scores = []
    clip_texture_gray_scores = []
    clip_geometry_sketch_scores = []
    ms_ssim_scores = []
    edge_recall_scores = []
    color_hist_scores = []
    geometry_residual_l2_distances = []
    geometry_residual_cosines = []
    geometry_improvement_ratios = []

    print("[INFO] Starting metric pass on {0} (LPIPS disabled)".format(metrics_device))
    model = CLIPModel.from_pretrained(args.clip_model).to(metrics_device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    for idx, record in enumerate(records, start=1):
        if not record.output_path.exists():
            continue

        (
            triple,
            output_image_raw,
            geometry_image,
            color_image,
            texture_image,
            geometry_baseline_image,
            output_baseline_image,
            output_image,
            output_sketch_rgb,
            geometry_sketch_rgb,
        ) = load_metric_images(record)

        save_canny_debug_images(
            record,
            debug_dir,
            geometry_image,
            geometry_baseline_image,
            output_image,
            output_baseline_image,
        )

        msssim_score = compute_ms_ssim_similarity(output_sketch_rgb, geometry_sketch_rgb, metrics_device)

        arr_out = np.array(output_sketch_rgb.convert("L"))
        arr_geo = np.array(geometry_sketch_rgb.convert("L"))
        mask_out = arr_out > 127
        mask_geo = arr_geo > 127
        recall = np.sum(mask_out & mask_geo) / (np.sum(mask_geo) + 1e-8)

        clip_text = clip_image_text_similarity(model, processor, metrics_device, output_image_raw, record.prompt)
        clip_geometry_sketch = clip_image_image_similarity(
            model,
            processor,
            metrics_device,
            output_sketch_rgb,
            geometry_sketch_rgb,
        )
        clip_texture_gray = clip_image_image_similarity(
            model,
            processor,
            metrics_device,
            output_image_raw.convert("L").convert("RGB"),
            texture_image.convert("L").convert("RGB"),
        )
        color_hist_dist = color_histogram_distance(output_image_raw, color_image, bins=args.hist_bins)
        res_metrics = compute_geometry_residual_metrics(
            model,
            processor,
            metrics_device,
            geometry_image,
            geometry_baseline_image,
            output_image,
            output_baseline_image,
        )
        imp_metrics = compute_geometry_improvement_ratio(
            model,
            processor,
            metrics_device,
            geometry_image,
            output_image,
            output_baseline_image,
        )

        clip_text_scores.append(clip_text)
        clip_geometry_sketch_scores.append(clip_geometry_sketch)
        ms_ssim_scores.append(msssim_score)
        edge_recall_scores.append(recall)
        clip_texture_gray_scores.append(clip_texture_gray)
        color_hist_scores.append(color_hist_dist)
        geometry_residual_l2_distances.append(res_metrics["geometry_residual_l2_distance"])
        geometry_residual_cosines.append(res_metrics["geometry_residual_cosine_similarity"])
        geometry_improvement_ratios.append(imp_metrics["geometry_improvement_ratio"])

        rows.append({
            "key": triple.key,
            "geometry_category": triple.geometry_category,
            "prompt": record.prompt,
            "clip_text": clip_text,
            "ms_ssim_geometry": msssim_score,
            "edge_recall_geometry": recall,
            "clip_texture_gray": clip_texture_gray,
            "color_histogram_distance": color_hist_dist,
            "geometry_residual_l2": res_metrics["geometry_residual_l2_distance"],
            "geometry_residual_cosine": res_metrics["geometry_residual_cosine_similarity"],
            "geometry_improvement_ratio": imp_metrics["geometry_improvement_ratio"],
            "output_path": str(record.output_path)
        })

        del output_image_raw
        del geometry_image
        del color_image
        del texture_image
        del geometry_baseline_image
        del output_baseline_image
        del output_image
        del output_sketch_rgb
        del geometry_sketch_rgb
        del arr_out
        del arr_geo

        if idx % 25 == 0 or idx == len(records):
            print("[INFO] Metric pass {0}/{1}".format(idx, len(records)))
            gc.collect()
            _maybe_empty_cuda_cache(metrics_device)

    del model
    del processor
    gc.collect()
    _maybe_empty_cuda_cache(metrics_device)

    summary = {
        "num_runs": len(rows),
        "clip_text_mean": safe_mean(clip_text_scores),
        "ms_ssim_geometry_mean": safe_mean(ms_ssim_scores),
        "edge_recall_geometry_mean": safe_mean(edge_recall_scores),
        "clip_texture_gray_mean": safe_mean(clip_texture_gray_scores),
        "color_histogram_distance_mean": safe_mean(color_hist_scores),
        "geometry_residual_l2_mean": safe_mean(geometry_residual_l2_distances),
        "geometry_residual_cosine_mean": safe_mean(geometry_residual_cosines),
        "geometry_improvement_ratio_mean": safe_mean(geometry_improvement_ratios),
    }

    return rows, summary

    # ????CLIP
    model = CLIPModel.from_pretrained(args.clip_model).to(metrics_device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # ????LPIPS
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(lpips_device)
    lpips_fn.eval()

    rows = []
    # ??容器
    clip_text_scores = []
    clip_texture_gray_scores = []
    clip_geometry_sketch_scores = []
    lpips_geometry_sketch_scores = []
    ms_ssim_scores = []    
    edge_recall_scores = [] 
    
    color_hist_scores = []
    geometry_residual_l2_distances = []
    geometry_residual_cosines = []
    geometry_improvement_ratios = []

    for record in records:
        if not record.output_path.exists():
            print(f"[SKIP METRIC] File not found: {record.output_path}")
            continue

        triple = record.triple
        prompt = record.prompt
        output_path = record.output_path
        baseline_info = baseline_metadata[str(output_path)]

        # 1. 載入影?
        output_image_raw = load_rgb(output_path)       # 1024x1024
        if image_cache is None:
            geometry_image = load_rgb(triple.geometry)     # 512x512 (Asset ??
            color_image = load_rgb(triple.color)
            texture_image = load_rgb(triple.texture)
            geometry_baseline_image = load_rgb(baseline_info["geometry_baseline_path"])
            output_baseline_image = load_rgb(baseline_info["output_baseline_path"])
        else:
            geometry_image = image_cache.get_image(triple.geometry, "RGB")
            color_image = image_cache.get_image(triple.color, "RGB")
            texture_image = image_cache.get_image(triple.texture, "RGB")
            geometry_baseline_image = image_cache.get_image(baseline_info["geometry_baseline_path"], "RGB")
            output_baseline_image = image_cache.get_image(baseline_info["output_baseline_path"], "RGB")

        # --- ?核心修??統???Asset 尺寸下?對?---
        # 將???下採? 512，這能???濾?質紋?????Canny ??
        target_size = geometry_image.size 
        output_image = output_image_raw.resize(target_size, Image.LANCZOS)
        
        # 2. ?? Canny ?徵 (?統一??512 尺寸?
        output_sketch_rgb = sketchify_for_clip_geometry(output_image)
        geometry_sketch_rgb = sketchify_for_clip_geometry(geometry_image)
        
        # 3. ?? Debug ??
        canny_paths = save_canny_debug_images(
            record, debug_dir, geometry_image, geometry_baseline_image, 
            output_image, output_baseline_image
        )

        # 4. ??計?
        
        # A. LPIPS (?知?似?- ?入已?齊尺寸?線?)
        lpips_geo_sim = compute_lpips_similarity(lpips_fn, output_sketch_rgb, geometry_sketch_rgb, lpips_device)
        
        # B. MS-SSIM (結??似?
        msssim_score = compute_ms_ssim_similarity(output_sketch_rgb, geometry_sketch_rgb, metrics_device)

        # C. Edge Recall (幾?????- 尺寸已???不???resize)
        arr_out = np.array(output_sketch_rgb.convert("L"))
        arr_geo = np.array(geometry_sketch_rgb.convert("L"))
            
        mask_out = arr_out > 127
        mask_geo = arr_geo > 127
        recall = np.sum(mask_out & mask_geo) / (np.sum(mask_geo) + 1e-8)

        # D. CLIP ?? (Text ??Texture 仍使??始解?度確?精確?
        clip_text = clip_image_text_similarity(model, processor, metrics_device, output_image_raw, prompt)
        clip_geometry_sketch = clip_image_image_similarity(model, processor, metrics_device, output_sketch_rgb, geometry_sketch_rgb)
        clip_texture_gray = clip_image_image_similarity(
            model, processor, metrics_device, 
            output_image_raw.convert("L").convert("RGB"), 
            texture_image.convert("L").convert("RGB")
        )
        
        color_hist_dist = color_histogram_distance(output_image_raw, color_image, bins=args.hist_bins)

        # 5. 幾?殘差?改?? (計? Improvement ?內???? embedding，直?傳?即??
        res_metrics = compute_geometry_residual_metrics(model, processor, metrics_device, geometry_image, geometry_baseline_image, output_image, output_baseline_image)
        imp_metrics = compute_geometry_improvement_ratio(model, processor, metrics_device, geometry_image, output_image, output_baseline_image)

        # ????
        clip_text_scores.append(clip_text)
        clip_geometry_sketch_scores.append(clip_geometry_sketch)
        lpips_geometry_sketch_scores.append(lpips_geo_sim)
        ms_ssim_scores.append(msssim_score)
        edge_recall_scores.append(recall)
        clip_texture_gray_scores.append(clip_texture_gray)
        color_hist_scores.append(color_hist_dist)
        geometry_residual_l2_distances.append(res_metrics["geometry_residual_l2_distance"])
        geometry_residual_cosines.append(res_metrics["geometry_residual_cosine_similarity"])
        geometry_improvement_ratios.append(imp_metrics["geometry_improvement_ratio"])

        # 寫入 CSV ?
        rows.append({
            "key": triple.key,
            "geometry_category": triple.geometry_category,
            "prompt": prompt,
            "clip_text": clip_text,
            "lpips_geometry_sketch": lpips_geo_sim,
            "ms_ssim_geometry": msssim_score,
            "edge_recall_geometry": recall,
            "clip_texture_gray": clip_texture_gray,
            "color_histogram_distance": color_hist_dist,
            "geometry_residual_l2": res_metrics["geometry_residual_l2_distance"],
            "geometry_residual_cosine": res_metrics["geometry_residual_cosine_similarity"],
            "geometry_improvement_ratio": imp_metrics["geometry_improvement_ratio"],
            "output_path": str(output_path)
        })
        del output_image_raw
        del geometry_image
        del color_image
        del texture_image
        del geometry_baseline_image
        del output_baseline_image
        del output_image
        del output_sketch_rgb
        del geometry_sketch_rgb
        del arr_out
        del arr_geo
        if len(rows) % 8 == 0:
            gc.collect()
            _maybe_empty_cuda_cache(metrics_device, lpips_device)

    # 7. 跨幾何???
    del model
    del processor
    gc.collect()
    _maybe_empty_cuda_cache(metrics_device)

    if args.skip_cross_geometry_lpips:
        cross_geo_sim = None
    else:
        cross_geo_sim = compute_cross_geometry_consistency_sim_lpips(records, lpips_fn, lpips_device)

    del lpips_fn
    gc.collect()
    _maybe_empty_cuda_cache(lpips_device)
    summary = {
        "num_runs": len(rows),
        "clip_text_mean": safe_mean(clip_text_scores),
        "lpips_geometry_sketch_mean": safe_mean(lpips_geometry_sketch_scores),
        "ms_ssim_geometry_mean": safe_mean(ms_ssim_scores),
        "edge_recall_geometry_mean": safe_mean(edge_recall_scores),
        "clip_texture_gray_mean": safe_mean(clip_texture_gray_scores),
        "color_histogram_distance_mean": safe_mean(color_hist_scores),
        "geometry_residual_l2_mean": safe_mean(geometry_residual_l2_distances),
        "geometry_residual_cosine_mean": safe_mean(geometry_residual_cosines),
        "geometry_improvement_ratio_mean": safe_mean(geometry_improvement_ratios),
        "cross_geometry_consistency_sim_lpips": cross_geo_sim,
    }
    
    return rows, summary

def save_reports(output_root: Path, rows: List[dict], summary: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["key"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    prompt_map = load_prompt_map(args.prompt_file)

    triples = build_triples(args)
    if args.limit > 0:
        triples = triples[:args.limit]
    if not triples:
        raise RuntimeError("No asset triples found.")
    validate_geometry_baseline_prompts(triples)

    os.makedirs(str(args.output_root), exist_ok=True)
    images_dir = args.output_root / "images"
    debug_dir = args.output_root / "debug"
    baselines_dir = args.output_root / "baselines"
    images_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] device={0}".format(args.device))
    print("[INFO] output_root={0}".format(args.output_root))
    print("[INFO] ablation=no_orthogonal_equal13")
    print("[INFO] WCT retained: guidance={0}, window=({1}, {2}), noise={3}".format(
        args.wct_guidance,
        args.wct_starts_step_ratio,
        args.wct_ends_step_ratio,
        args.wctnoise_add_scale,
    ))
    print("[INFO] stream stage factors: color={0}, geometry={1}, texture={2}".format(
        tuple(args.color_stage_factors),
        tuple(args.geometry_stage_factors),
        tuple(args.texture_stage_factors),
    ))
    print("[INFO] metrics_only={0}".format(args.metrics_only))
    print("[INFO] metrics_device={0}".format(_resolve_metric_device(args.metrics_device, args.device)))
    print("[INFO] lpips=disabled")
    print("[INFO] triples={0}".format(len(triples)))
    category_counts = {}
    for triple in triples:
        category_counts[triple.geometry_category] = category_counts.get(triple.geometry_category, 0) + 1
    print("[INFO] geometry_category_triples={0}".format(json.dumps(category_counts, sort_keys=True)))

    ip_model = None
    image_cache = ImageCache()
    if args.asset_preload_workers > 0 and not args.metrics_only:
        image_cache.preload_triple_assets(triples, workers=args.asset_preload_workers)
    elif args.metrics_only and args.asset_preload_workers > 0:
        print("[INFO] Skipping triple asset preload in metrics-only mode")

    records = []

    for triple_idx, triple in enumerate(triples, start=1):
        prompts = resolve_prompts_for_triple(prompt_map, triple)

        if not args.metrics_only:
            geometry_debug = image_cache.get_geometry_preprocessed(triple.geometry)
            geometry_debug.save(debug_dir / "{0}_geometry_debug.png".format(triple.key))
        elif triple_idx % 50 == 0 or triple_idx == len(triples):
            print("[INFO] Indexed {0}/{1} triples for metrics".format(triple_idx, len(triples)))

        for prompt in prompts:
            prompt_tag = safe_prompt_name(prompt)
            output_path = images_dir / "{0}_{1}.png".format(triple.key, prompt_tag)
            records.append(RunRecord(triple=triple, prompt=prompt, output_path=output_path))

            if args.metrics_only:
                continue

            if args.skip_existing and output_path.exists():
                print("[SKIP] {0}".format(output_path))
                continue

            if ip_model is None:
                ip_model = build_ip_model(args.device)

            result = generate_one(ip_model, triple, prompt, args, image_cache=image_cache)
            result.save(output_path)
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[DONE] {0} | prompt={1} -> {2}".format(triple.key, prompt, output_path.name))

    missing = [record.output_path for record in records if not record.output_path.exists()]
    if missing and args.metrics_only and args.allow_partial_metrics:
        before = len(records)
        records = [record for record in records if record.output_path.exists()]
        print(
            "[WARN] Partial metrics enabled: using {0}/{1} existing outputs; {2} outputs are missing.".format(
                len(records),
                before,
                len(missing),
            )
        )
    elif missing:
        raise FileNotFoundError("Some outputs are missing:\n" + "\n".join(str(p) for p in missing))

    if ip_model is not None:
        del ip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows, summary = compute_metrics(args, records, baselines_dir, image_cache=image_cache)
    save_reports(args.output_root, rows, summary)

    print(json.dumps(summary, indent=2))
    print("Saved metrics CSV to {0}".format(args.output_root / "metrics.csv"))
    print("Saved summary JSON to {0}".format(args.output_root / "summary.json"))


if __name__ == "__main__":
    main()
