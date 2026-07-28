import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pillow_avif
import zlib

from pipeline_stable_diffusion_xl_equal_attention import StableDiffusionXLPipeline
from ip_adapter_equal_attention import IPAdapterPlusXL
from util.torch_compat import ensure_supported_cuda_runtime


RUNTIME_MODE_TAG = "color_texture_geometry_equal_attention"
TARGET_BLOCKS = ["down_blocks", "mid_block", "up_blocks"]


def _parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Run equal-attention SADis inference with optional color/geometry/texture references."
    )
    parser.add_argument("--input-color", default=None)
    parser.add_argument("--input-geometry", default=None)
    parser.add_argument("--input-texture", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


RUNTIME_ARGS = _parse_runtime_args()


def _normalize_optional_text(value):
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _get_env_path_override(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return _normalize_optional_text(value)


def _get_env_text_override(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value


def _get_env_int_override(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return 0
    return int(value)


def _get_env_bool_override(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# 2026-04-13 temple texture_geometry reproduction setup.
INPUT_COLOR = r"assets/color/color05.jpg" # Optional. Enables the color stream when set.
INPUT_GEOMETRY = r"assets/geometry/face05.jpeg"  # Optional. Enables the geometry stream when set.
INPUT_TEXTURE = r"assets/texture/artwork_3.jpg"  # Optional. Enables the texture stream when set.
# INPUT_GEOMETRY = None  # Optional. Enables the geometry stream when set.
# INPUT_TEXTURE = None  # Optional. Enables the texture stream when set.

INPUT_COLOR = _get_env_path_override("SADIS_INPUT_COLOR", INPUT_COLOR)
INPUT_GEOMETRY = _get_env_path_override("SADIS_INPUT_GEOMETRY", INPUT_GEOMETRY)
INPUT_TEXTURE = _get_env_path_override("SADIS_INPUT_TEXTURE", INPUT_TEXTURE)
if RUNTIME_ARGS.input_color is not None:
    INPUT_COLOR = _normalize_optional_text(RUNTIME_ARGS.input_color)
if RUNTIME_ARGS.input_geometry is not None:
    INPUT_GEOMETRY = _normalize_optional_text(RUNTIME_ARGS.input_geometry)
if RUNTIME_ARGS.input_texture is not None:
    INPUT_TEXTURE = _normalize_optional_text(RUNTIME_ARGS.input_texture)


DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, letters, logo, lowres, low quality, worst quality, deformed, glitch, "
    "low contrast, noisy, blurry, copy reference image, same composition as reference, "
    "identical layout, copy texture reference, same objects as texture reference, "
    "same composition as texture reference, two children, multiple children, extra child, "
    "extra person, duplicate person, twins, crowd, church, cathedral, chapel, mosque, "
    "dull, desaturated, greyish, monochromatic, low saturation, flat lighting"
)

save_dir = r"results/disentangled/equal_attention/repro_20260504"
save_dir = _get_env_text_override("SADIS_SAVE_DIR", save_dir)
if RUNTIME_ARGS.save_dir is not None:
    save_dir = RUNTIME_ARGS.save_dir
os.makedirs(save_dir, exist_ok=True)
SKIP_EXISTING = _get_env_bool_override("SADIS_SKIP_EXISTING", False) or RUNTIME_ARGS.skip_existing

# prompt_list = ["A detailed landscape photograph of a rugged coastal lighthouse scene at sunset. A tall, weathered stone lighthouse with classic red and white stripes stands on a rocky promontory. Attached to the tower is a small, stone keeper's cottage with a slate roof, a smoking chimney, and a neatly stacked pile of firewood. A stone wall encloses a small garden patch with hardy coastal plants. A winding dirt path lined with weathered wooden fencing leads toward the structure. Below the cliff, crashing ocean waves hit jagged rocks covered in barnacles and dark seaweed. Driftwood, old lobster traps, and tangled fishing nets are scattered on a small pebble beach. Seagulls are circling overhead and perched on the rocks. The lighthouse lamp is just beginning to glow, casting a warm beam. The light is golden hour."
# ]
# prompt_list = ["a house with trees"]
prompt_list = [
    (
        "a man protrait"
    )
]

steps = 50
seed = 42
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_samples = 1
scale = 1.0
guidance_scale = 9.0
texture2_scale = 0.0
INPUT_TEXTURE2 = None

color_scale = 1.6
substract_scale = 1.0
texture_scale = 1.4
geometry_scale = 1.4
geometry_sub_scale = 0.80
texture_color_decouple = 0.10
geometry_color_decouple = 0.35
color_to_geometry_decouple = 0.25

wct_guidance = 0.28
wct_starts_step_ratio = 0.45
wct_ends_step_ratio = 0.55
wctnoise_add_scale = 0.004
front_end_ratio = 0.40
mid_end_ratio = 0.70
# color_stage_factors = (0.55, 0.80, 1.0)
# geometry_stage_factors = (1.35, 1.05, 0.65)
# texture_stage_factors = (0.38, 0.72, 0.82)
color_stage_factors = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
geometry_stage_factors = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
texture_stage_factors = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
punish_weight = 0.0003
punish_type = "soft-weight"
SAVE_GEOMETRY_PREPROCESS_DEBUG = True
GEOMETRY_LUMA_TARGET_MEAN = 175.0
GEOMETRY_LUMA_MEAN_STRENGTH = 0.60
GEOMETRY_LUMA_CONTRAST = 0.65
GEOMETRY_CHROMA_STRENGTH = 0.25
GEOMETRY_EDGE_MIN_COMPONENT_AREA = 64
GEOMETRY_EDGE_MIN_COMPONENT_LENGTH = _get_env_int_override("SADIS_GEOMETRY_EDGE_MIN_COMPONENT_LENGTH", 36)
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


def _require_exists(path, kind="file"):
    ok = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
    if not ok:
        raise FileNotFoundError(f"Required {kind} not found: {path}")


def _require_reference_file(path, logical_name, source_name):
    if not path:
        raise FileNotFoundError(
            f"{logical_name} reference file path is empty (resolved from {source_name})."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{logical_name} reference file not found (resolved from {source_name}): {path}"
        )


def _is_empty_ref(path):
    return path is None or path == ""


def _get_generator(seed_value, device_name):
    if seed_value is None:
        return None
    if isinstance(seed_value, list):
        return [torch.Generator(device_name).manual_seed(item) for item in seed_value]
    return torch.Generator(device_name).manual_seed(seed_value)


def _make_debug_image_path(source_path, suffix):
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(save_dir, f"{stem}_{suffix}.png")


def _build_texture_highpass_image(texture_img, blur_radius=5.0):
    texture_rgb = texture_img.convert("RGB")
    texture_blur = texture_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    texture_np = np.array(texture_rgb).astype(np.float32)
    texture_blur_np = np.array(texture_blur).astype(np.float32)
    texture_highpass_np = np.clip(texture_np - texture_blur_np + 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(texture_highpass_np)


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
    canny_sigma=0.18,
):
    # Match run_dog_tuning_experiments.extract_canny_contour(), which is used
    # to generate the Canny images for geometry metrics.
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


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

active_color = not _is_empty_ref(INPUT_COLOR)
active_geometry = not _is_empty_ref(INPUT_GEOMETRY)
active_texture = not _is_empty_ref(INPUT_TEXTURE)
pure_sdxl_mode = not (active_color or active_geometry or active_texture)

effective_guidance_scale = guidance_scale
effective_color_scale = color_scale
effective_substract_scale = substract_scale
effective_texture_scale = texture_scale
effective_texture2_scale = texture2_scale
effective_geometry_scale = geometry_scale
effective_geometry_sub_scale = geometry_sub_scale
effective_geometry_color_decouple = geometry_color_decouple
effective_texture_color_decouple = texture_color_decouple
effective_color_to_geometry_decouple = color_to_geometry_decouple
effective_wct_guidance = wct_guidance
effective_wct_starts_step_ratio = wct_starts_step_ratio
effective_wct_ends_step_ratio = wct_ends_step_ratio
effective_wctnoise_add_scale = wctnoise_add_scale
effective_punish_weight = punish_weight
effective_punish_type = punish_type

effective_color_path = None
effective_geometry_path = None
effective_texture_path = None
resolved_color_stage_factors = tuple(float(x) for x in color_stage_factors)
resolved_geometry_stage_factors = tuple(float(x) for x in geometry_stage_factors)
resolved_texture_stage_factors = tuple(float(x) for x in texture_stage_factors)

if not pure_sdxl_mode:
    effective_color_path = INPUT_COLOR if active_color else None
    effective_geometry_path = INPUT_GEOMETRY if active_geometry else None
    effective_texture_path = INPUT_TEXTURE if active_texture else None

    if active_color:
        _require_reference_file(effective_color_path, "Color", "INPUT_COLOR")
    if active_geometry:
        _require_reference_file(effective_geometry_path, "Geometry", "INPUT_GEOMETRY")
    if active_texture:
        _require_reference_file(effective_texture_path, "Texture", "INPUT_TEXTURE")
    if INPUT_TEXTURE2:
        _require_reference_file(INPUT_TEXTURE2, "Texture2", "INPUT_TEXTURE2")
    _require_exists(image_encoder_path, "dir")
    _require_exists(ip_ckpt, "file")

ensure_supported_cuda_runtime(device)

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe = pipe.to(device)
pipe.enable_vae_tiling()

ip_model = None
if pure_sdxl_mode:
    print("Mode: sdxl (no reference inputs)")
else:
    print(f"Mode: {RUNTIME_MODE_TAG}")
    print("Using reference inputs:")
    print(f"  color: {effective_color_path if active_color else 'disabled -> SDXL'}")
    print(f"  geometry: {effective_geometry_path if active_geometry else 'disabled -> SDXL'}")
    print(f"  texture: {effective_texture_path if active_texture else 'disabled -> SDXL'}")
    print("Applying runtime params:")
    print(
        "  "
        f"guidance_scale={effective_guidance_scale}, "
        f"color_scale={effective_color_scale}, "
        f"substract_scale={effective_substract_scale}, "
        f"texture_scale={effective_texture_scale}, "
        f"geometry_scale={effective_geometry_scale}, "
        f"wct_guidance={effective_wct_guidance}, "
        f"wct_window=({effective_wct_starts_step_ratio}, {effective_wct_ends_step_ratio}), "
        f"wctnoise_add_scale={effective_wctnoise_add_scale}, "
        f"punish_weight={effective_punish_weight}, "
        f"punish_type={effective_punish_type}"
    )

    ip_model = IPAdapterPlusXL(
        pipe,
        image_encoder_path,
        ip_ckpt,
        device,
        num_tokens=16,
        target_blocks=TARGET_BLOCKS,
    )

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

geometry_ref_img = None
raw_color_ref_img = None
color_image_gray = None
texture_ref_img = None
texture_image_gray = None
texture_ref_img2 = None
if not pure_sdxl_mode:
    if active_geometry:
        geometry_ref_img = preprocess_geometry_for_rules(effective_geometry_path)
        if SAVE_GEOMETRY_PREPROCESS_DEBUG:
            geometry_debug_path = _make_debug_image_path(
                effective_geometry_path,
                "geometry_preprocess",
            )
            geometry_ref_img.save(geometry_debug_path)
            print(f"Saved geometry preprocess preview to: {geometry_debug_path}")
    if active_color:
        raw_color_ref_img = Image.open(effective_color_path).convert("RGB")
        color_image_gray = raw_color_ref_img.convert("L")
    if active_texture:
        texture_ref_img = Image.open(effective_texture_path).convert("RGB")
        texture_image_gray = texture_ref_img.convert("L")
    texture_ref_img2 = None if not INPUT_TEXTURE2 else Image.open(INPUT_TEXTURE2).convert("RGB")


for prompt in prompt_list:
    safe_prompt = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in prompt)[:40]
    if pure_sdxl_mode:
        svname = f"{safe_prompt}_sdxl.png"
    else:
        svname = (
            f"{safe_prompt}_{RUNTIME_MODE_TAG}_"
            f"C{effective_color_scale}_G{effective_geometry_scale}_T{effective_texture_scale}.png"
        )
    output_path = os.path.join(save_dir, svname)
    if SKIP_EXISTING and os.path.exists(output_path):
        print(f"[SKIP] Existing output: {output_path}")
        continue

    generator = _get_generator(seed, device)
    if pure_sdxl_mode:
        images = pipe(
            prompt=prompt,
            negative_prompt=None,
            guidance_scale=effective_guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=generator,
        ).images
        mode_tag = "sdxl"
    else:
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            scale=scale,
            guidance_scale=effective_guidance_scale,
            clr_ref_img=raw_color_ref_img,
            clr_texture_ref_img=color_image_gray,
            texture_ref_img=texture_image_gray,
            geometry_ref_img=geometry_ref_img,
            texture_ref_img2=texture_ref_img2,
            color_scale=effective_color_scale,
            substract_scale=effective_substract_scale,
            texture_scale=effective_texture_scale,
            texture2_scale=effective_texture2_scale,
            geometry_scale=effective_geometry_scale,
            geometry_sub_scale=effective_geometry_sub_scale,
            geometry_color_decouple=effective_geometry_color_decouple,
            texture_color_decouple=effective_texture_color_decouple,
            color_to_geometry_decouple=effective_color_to_geometry_decouple,
            num_samples=num_samples,
            num_inference_steps=steps,
            seed=seed,
            front_end_ratio=front_end_ratio,
            mid_end_ratio=mid_end_ratio,
            color_stage_factors=resolved_color_stage_factors,
            geometry_stage_factors=resolved_geometry_stage_factors,
            texture_stage_factors=resolved_texture_stage_factors,
            log_stream_schedule=True,
            wct_guidance=effective_wct_guidance if active_color else 0.0,
            wct_starts_step=effective_wct_starts_step_ratio * steps,
            wct_ends_step=effective_wct_ends_step_ratio * steps,
            wctnoise_add_scale=effective_wctnoise_add_scale,
            punish_weight=effective_punish_weight,
            punish_type=effective_punish_type,
            clr_ref_img_dir=effective_color_path,
            sty_ref_img_dir=effective_texture_path,
            save_name=prompt,
        )
        mode_tag = RUNTIME_MODE_TAG

    final_img = _enhance_final_output(images[0])
    final_img.save(output_path)

print(f"Generation done. Results saved at: {save_dir}")
