from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results" / "gemini" / "face_beard_grid"
OUTPUT_PATH = RESULT_DIR / "face_beard_grid_effect_figure.png"

PROMPT = "A portrait of a man with a beard"

GEOMETRIES = [
    ("face01", ASSET_DIR / "geometry" / "face01.jpeg"),
    ("face04", ASSET_DIR / "geometry" / "face04.webp"),
    ("face05", ASSET_DIR / "geometry" / "face05.jpeg"),
    ("face10", ASSET_DIR / "geometry" / "face10.jpeg"),
    ("face11", ASSET_DIR / "geometry" / "face11.jpg"),
]

SETTINGS = [
    ("color1", ASSET_DIR / "1.png", "texture007", ASSET_DIR / "texture" / "texture007.jpeg"),
    ("color004", ASSET_DIR / "color" / "color004.jpeg", "texture008", ASSET_DIR / "texture" / "texture008.jpeg"),
    ("color005", ASSET_DIR / "color" / "color005.jpeg", "texture003", ASSET_DIR / "texture" / "texture003.jpeg"),
    ("color006", ASSET_DIR / "color" / "color006.png", "8", ASSET_DIR / "texture" / "8.png"),
    ("color007", ASSET_DIR / "color" / "color007.jpeg", "001", ASSET_DIR / "texture" / "001.jpg"),
]


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_PROMPT = load_font(18)
FONT_MISSING = load_font(14, bold=True)


def fit_image(path: Path, size: Tuple[int, int], background=(255, 255, 255)) -> Image.Image:
    if not path.exists():
        image = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 50, 50), width=2)
        draw.text((8, 8), "Missing", fill=(140, 30, 30), font=FONT_MISSING)
        return image

    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def result_path(geometry_name: str, color_name: str, texture_name: str) -> Path:
    stem = f"{geometry_name}_{color_name}_{texture_name}_A_portrait_of_a_man_with_a_beard"
    return RESULT_DIR / f"{stem}.png"


def draw_prompt(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "Prompt:", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 26), "A portrait of a man", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 52), "with a beard", fill=(20, 20, 20), font=FONT_PROMPT)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pad = 18
    gap = 8
    left_w = 180
    top_h = 124
    cell_w = 136
    cell_h = 136
    ref_size = 74

    width = pad * 2 + left_w + gap + len(GEOMETRIES) * cell_w + (len(GEOMETRIES) - 1) * gap
    height = pad * 2 + top_h + gap + len(SETTINGS) * cell_h + (len(SETTINGS) - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    grid_x = pad + left_w + gap
    top_y = pad
    row_y = top_y + top_h + gap

    draw.rectangle((pad, top_y, pad + left_w, top_y + top_h), fill=(248, 248, 248), outline=(198, 198, 198))
    draw_prompt(draw, pad + 16, top_y + 24)

    for col, (_, geometry_path) in enumerate(GEOMETRIES):
        x = grid_x + col * (cell_w + gap)
        draw.rectangle((x, top_y, x + cell_w, top_y + top_h), fill=(248, 248, 248), outline=(198, 198, 198))
        geometry_img = fit_image(geometry_path, (cell_w - 18, top_h - 18))
        canvas.paste(geometry_img, (x + 9, top_y + 9))

    for row, (color_name, color_path, texture_name, texture_path) in enumerate(SETTINGS):
        y = row_y + row * (cell_h + gap)
        draw.rectangle((pad, y, pad + left_w, y + cell_h), fill=(248, 248, 248), outline=(198, 198, 198))

        color_img = fit_image(color_path, (ref_size, ref_size))
        texture_img = fit_image(texture_path, (ref_size, ref_size))
        ref_y = y + (cell_h - ref_size) // 2
        canvas.paste(color_img, (pad + 14, ref_y))
        canvas.paste(texture_img, (pad + 14 + ref_size + 10, ref_y))

        for col, (geometry_name, _) in enumerate(GEOMETRIES):
            x = grid_x + col * (cell_w + gap)
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=(255, 255, 255), outline=(215, 215, 215))
            output_img = fit_image(result_path(geometry_name, color_name, texture_name), (cell_w, cell_h))
            canvas.paste(output_img, (x, y))

    canvas.save(OUTPUT_PATH)
    print(f"[SAVE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
