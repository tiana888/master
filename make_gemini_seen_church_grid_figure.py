from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results" / "gemini" / "seen_church_grid"
OUTPUT_PATH = RESULT_DIR / "seen_church_grid_effect_figure.png"

PROMPT = "A building of a beautiful church in a city"

GEOMETRIES = [
    ("seen01", ASSET_DIR / "geometry" / "seen01.jpg"),
    ("seen02", ASSET_DIR / "geometry" / "seen02.webp"),
    ("seen03", ASSET_DIR / "geometry" / "seen03.png"),
    ("seen04", ASSET_DIR / "geometry" / "seen04.png"),
    ("seen06", ASSET_DIR / "geometry" / "seen06.jpeg"),
    ("seen07", ASSET_DIR / "geometry" / "seen07.jpeg"),
]

SETTINGS = [
    ("color1", ASSET_DIR / "1.png", "artwork_167", ASSET_DIR / "texture" / "artwork_167.jpg"),
    ("color002", ASSET_DIR / "color" / "color002.jpeg", "texture007", ASSET_DIR / "texture" / "texture007.jpeg"),
    ("color003", ASSET_DIR / "color" / "color003.jpeg", "8", ASSET_DIR / "texture" / "8.png"),
    ("color007", ASSET_DIR / "color" / "color007.jpeg", "texture006", ASSET_DIR / "texture" / "texture006.jpeg"),
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


FONT_PROMPT = load_font(17)
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
    stem = f"{geometry_name}_{color_name}_{texture_name}_A_building_of_a_beautiful_church_in_a_city"
    return RESULT_DIR / f"{stem}.png"


def draw_prompt(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "Prompt:", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 25), "A building of a", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 50), "beautiful church", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 75), "in a city", fill=(20, 20, 20), font=FONT_PROMPT)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pad = 18
    gap = 7
    left_w = 170
    top_h = 116
    cell_w = 112
    cell_h = 112
    ref_size = 68

    width = pad * 2 + left_w + gap + len(GEOMETRIES) * cell_w + (len(GEOMETRIES) - 1) * gap
    height = pad * 2 + top_h + gap + len(SETTINGS) * cell_h + (len(SETTINGS) - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    grid_x = pad + left_w + gap
    top_y = pad
    row_y = top_y + top_h + gap

    draw.rectangle((pad, top_y, pad + left_w, top_y + top_h), fill=(248, 248, 248), outline=(198, 198, 198))
    draw_prompt(draw, pad + 14, top_y + 13)

    for col, (_, geometry_path) in enumerate(GEOMETRIES):
        x = grid_x + col * (cell_w + gap)
        draw.rectangle((x, top_y, x + cell_w, top_y + top_h), fill=(248, 248, 248), outline=(198, 198, 198))
        geometry_img = fit_image(geometry_path, (cell_w - 14, top_h - 14))
        canvas.paste(geometry_img, (x + 7, top_y + 7))

    for row, (color_name, color_path, texture_name, texture_path) in enumerate(SETTINGS):
        y = row_y + row * (cell_h + gap)
        draw.rectangle((pad, y, pad + left_w, y + cell_h), fill=(248, 248, 248), outline=(198, 198, 198))

        color_img = fit_image(color_path, (ref_size, ref_size))
        texture_img = fit_image(texture_path, (ref_size, ref_size))
        ref_y = y + (cell_h - ref_size) // 2
        canvas.paste(color_img, (pad + 11, ref_y))
        canvas.paste(texture_img, (pad + 11 + ref_size + 10, ref_y))

        for col, (geometry_name, _) in enumerate(GEOMETRIES):
            x = grid_x + col * (cell_w + gap)
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=(255, 255, 255), outline=(215, 215, 215))
            output_img = fit_image(result_path(geometry_name, color_name, texture_name), (cell_w, cell_h))
            canvas.paste(output_img, (x, y))

    canvas.save(OUTPUT_PATH)
    print(f"[SAVE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
