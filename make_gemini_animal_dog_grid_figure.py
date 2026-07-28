from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results" / "gemini" / "animal_dog_grid"
OUTPUT_PATH = RESULT_DIR / "animal_dog_grid_effect_figure.png"

PROMPT = "A single dog on the ground"

GEOMETRIES = [
    ("animal01", ASSET_DIR / "geometry" / "animal01.jpg"),
    ("animal05", ASSET_DIR / "geometry" / "animal05.jpg"),
    ("animal06", ASSET_DIR / "geometry" / "animal06.jpeg"),
    ("animal08", ASSET_DIR / "geometry" / "animal08.jpeg"),
]

SETTINGS = [
    ("color10", ASSET_DIR / "color" / "color10.jpg", "texture007", ASSET_DIR / "texture" / "texture007.jpeg"),
    ("color007", ASSET_DIR / "color" / "color007.jpeg", "texture8", ASSET_DIR / "texture" / "8.png"),
    ("color006", ASSET_DIR / "color" / "color006.png", "texture003", ASSET_DIR / "texture" / "texture003.jpeg"),
    ("color005", ASSET_DIR / "color" / "color005.jpeg", "texture006", ASSET_DIR / "texture" / "texture006.jpeg"),
    ("color05", ASSET_DIR / "color" / "color05.jpg", "artwork167", ASSET_DIR / "texture" / "artwork_167.jpg"),
    ("color003", ASSET_DIR / "color" / "color003.jpeg", "texture007", ASSET_DIR / "texture" / "texture007.jpeg"),
]

RESULT_TEXTURE_NAMES = {
    "texture8": "8",
    "artwork167": "artwork_167",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_HEADER = load_font(22, bold=True)
FONT_LABEL = load_font(18, bold=True)
FONT_SMALL = load_font(16)


def fit_image(path: Path, size: Tuple[int, int], background=(255, 255, 255)) -> Image.Image:
    if not path.exists():
        image = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(190, 60, 60), width=3)
        draw.text((12, 12), "Missing", fill=(140, 30, 30), font=FONT_LABEL)
        draw.text((12, 42), path.name, fill=(140, 30, 30), font=FONT_SMALL)
        return image

    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill=(20, 20, 20)) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_multiline(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], lines, font, fill=(20, 20, 20), spacing=8) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + spacing


def result_path(geometry_name: str, color_name: str, texture_label: str) -> Path:
    texture_name = RESULT_TEXTURE_NAMES.get(texture_label, texture_label)
    stem = f"{geometry_name}_{color_name}_{texture_name}_A_single_dog_on_the_ground"
    return RESULT_DIR / f"{stem}.png"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cell_w = 260
    cell_h = 260
    ref_w = 122
    left_w = 350
    top_h = 300
    title_h = 78
    gap = 16
    pad = 28

    width = pad * 2 + left_w + gap + len(GEOMETRIES) * cell_w + (len(GEOMETRIES) - 1) * gap
    height = pad * 2 + title_h + top_h + gap + len(SETTINGS) * cell_h + (len(SETTINGS) - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, pad), "Gemini animal input effect grid", font=FONT_TITLE, fill=(20, 20, 20))

    top_y = pad + title_h
    left_x = pad
    grid_x = pad + left_w + gap

    draw.rectangle((left_x, top_y, left_x + left_w, top_y + top_h), fill=(248, 248, 248), outline=(210, 210, 210))
    draw_multiline(draw, (left_x + 18, top_y + 24), ["Prompt", PROMPT], FONT_LABEL)

    for col, (geometry_name, geometry_path) in enumerate(GEOMETRIES):
        x = grid_x + col * (cell_w + gap)
        draw.rectangle((x, top_y, x + cell_w, top_y + top_h), fill=(248, 248, 248), outline=(210, 210, 210))
        image = fit_image(geometry_path, (cell_w - 34, top_h - 34))
        canvas.paste(image, (x + 17, top_y + 17))

    row_y = top_y + top_h + gap
    for row, (color_name, color_path, texture_label, texture_path) in enumerate(SETTINGS):
        y = row_y + row * (cell_h + gap)
        draw.rectangle((left_x, y, left_x + left_w, y + cell_h), fill=(248, 248, 248), outline=(210, 210, 210))
        color_img = fit_image(color_path, (ref_w + 28, ref_w + 28))
        texture_img = fit_image(texture_path, (ref_w + 28, ref_w + 28))
        canvas.paste(color_img, (left_x + 20, y + 55))
        canvas.paste(texture_img, (left_x + 20 + ref_w + 58, y + 55))

        for col, (geometry_name, _) in enumerate(GEOMETRIES):
            x = grid_x + col * (cell_w + gap)
            path = result_path(geometry_name, color_name, texture_label)
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=(255, 255, 255), outline=(220, 220, 220))
            image = fit_image(path, (cell_w, cell_h))
            canvas.paste(image, (x, y))

    canvas.save(OUTPUT_PATH)
    print(f"[SAVE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
