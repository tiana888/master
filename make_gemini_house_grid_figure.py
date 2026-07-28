from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results" / "gemini" / "house_grid"
OUTPUT_PATH = RESULT_DIR / "house_grid_effect_figure_compact.png"

PROMPT = "One beautiful church on the ground"

GEOMETRIES = [
    ("house01", ASSET_DIR / "geometry" / "house01.jpg"),
    ("house02", ASSET_DIR / "geometry" / "house02.jpg"),
    ("house03", ASSET_DIR / "geometry" / "house03.jpg"),
    ("house04", ASSET_DIR / "geometry" / "house04.jpeg"),
    ("house05", ASSET_DIR / "geometry" / "house05.jpeg"),
    ("house06", ASSET_DIR / "geometry" / "house06.jpeg"),
    ("house07", ASSET_DIR / "geometry" / "house07.jpeg"),
    ("house08", ASSET_DIR / "geometry" / "house08.jpeg"),
]

SETTINGS = [
    ("color1", ASSET_DIR / "1.png", "texture007", ASSET_DIR / "texture" / "texture007.jpeg"),
    ("color002", ASSET_DIR / "color" / "color002.jpeg", "8", ASSET_DIR / "texture" / "8.png"),
    ("color004", ASSET_DIR / "color" / "color004.jpeg", "texture008", ASSET_DIR / "texture" / "texture008.jpeg"),
    ("color05", ASSET_DIR / "color" / "color05.jpg", "artwork_3", ASSET_DIR / "texture" / "artwork_3.jpg"),
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


FONT_PROMPT = load_font(16)
FONT_LABEL = load_font(17, bold=True)
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
    stem = f"{geometry_name}_{color_name}_{texture_name}_One_beautiful_church_on_the_ground"
    return RESULT_DIR / f"{stem}.png"


def draw_centered(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + (right - left - (bbox[2] - bbox[0])) // 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def draw_prompt(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "Prompt: One beautiful", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 23), "church on the", fill=(20, 20, 20), font=FONT_PROMPT)
    draw.text((x, y + 46), "ground", fill=(20, 20, 20), font=FONT_PROMPT)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 1592, 1106
    pad = 18
    left_w = 174
    gap = 8
    header_h = 90
    label_h = 29
    row_gap = 10
    cell_w = 166
    cell_h = 166
    row_h = 181
    ref_size = 83

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    grid_x = pad + left_w + gap
    header_y = pad
    label_y = header_y + header_h
    row_y = label_y + label_h + row_gap

    draw_prompt(draw, pad, header_y + 10)

    for col, (_, geometry_path) in enumerate(GEOMETRIES):
        x = grid_x + col * (cell_w + gap)
        geometry_img = fit_image(geometry_path, (83, 83))
        canvas.paste(geometry_img, (x + (cell_w - 83) // 2, header_y))

    draw.rounded_rectangle((pad, label_y, pad + 83, label_y + label_h), radius=4, fill=(247, 248, 250), outline=(188, 192, 196))
    draw.rounded_rectangle((pad + 91, label_y, pad + 174, label_y + label_h), radius=4, fill=(247, 248, 250), outline=(188, 192, 196))
    draw.rounded_rectangle((grid_x, label_y, width - pad, label_y + label_h), radius=4, fill=(247, 248, 250), outline=(188, 192, 196))
    draw_centered(draw, (pad, label_y, pad + 83, label_y + label_h), "Color", FONT_LABEL)
    draw_centered(draw, (pad + 91, label_y, pad + 174, label_y + label_h), "Texture", FONT_LABEL)
    draw_centered(draw, (grid_x, label_y, width - pad, label_y + label_h), "Geometry", FONT_LABEL)

    for row, (color_name, color_path, texture_name, texture_path) in enumerate(SETTINGS):
        y = row_y + row * (row_h + row_gap)
        draw.rounded_rectangle((pad, y, width - pad, y + row_h), radius=4, fill=(255, 255, 255), outline=(198, 198, 198))

        ref_y = y + (row_h - ref_size) // 2
        canvas.paste(fit_image(color_path, (ref_size, ref_size)), (pad, ref_y))
        canvas.paste(fit_image(texture_path, (ref_size, ref_size)), (pad + 91, ref_y))

        for col, (geometry_name, _) in enumerate(GEOMETRIES):
            x = grid_x + col * (cell_w + gap)
            output_img = fit_image(result_path(geometry_name, color_name, texture_name), (cell_w, cell_h))
            canvas.paste(output_img, (x, y + (row_h - cell_h) // 2))

    canvas.save(OUTPUT_PATH)
    print(f"[SAVE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
