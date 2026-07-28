from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results" / "dog_tuning_experiments0508"
IMAGE_DIR = RESULT_DIR / "images"
OUTPUT_PATH = RESULT_DIR / "teaser_disentanglement_controls.png"
HOUSE_TEMPLE_GRID = (
    RESULT_DIR
    / "one_traditional_chinese_temple_on_the_ground_grids"
    / "color006_textures_x_geometry_house01-08.png"
)


def grid_crop(path: Path, x1: int, y1: int, x2: int, y2: int) -> tuple[str, Path, tuple[int, int, int, int]]:
    return ("grid_crop", path, (x1, y1, x2, y2))


ANIMAL = {
    "title": "Animal",
    "prompt": "A single dog on the ground",
    "base_refs": [
        ("G", ASSET_DIR / "geometry" / "animal01.jpg"),
        ("C", ASSET_DIR / "color" / "color005.jpeg"),
        ("T", ASSET_DIR / "texture" / "8.png"),
    ],
    "outputs": [
        (
            "Illustration",
            ("G", ASSET_DIR / "geometry" / "animal01.jpg"),
            IMAGE_DIR / "animal01__color005__8_A_single_dog_on_the_ground.png",
        ),
        (
            "Cartoon",
            ("G", ASSET_DIR / "geometry" / "animal06.jpeg"),
            IMAGE_DIR / "animal06__color005__8_A_single_dog_on_the_ground.png",
        ),
        (
            "Comic",
            ("G", ASSET_DIR / "geometry" / "animal08.jpeg"),
            IMAGE_DIR / "animal08__color005__8_A_single_dog_on_the_ground.png",
        ),
    ],
}


HOUSE = {
    "title": "House",
    "prompt": "One traditional Chinese temple on the ground",
    "base_refs": [
        ("G", ASSET_DIR / "geometry" / "house04.jpeg"),
        ("C", ASSET_DIR / "color" / "color006.png"),
        ("T", ASSET_DIR / "texture" / "texture007.jpeg"),
    ],
    "outputs": [
        (
            "Close-Up",
            ("G", ASSET_DIR / "geometry" / "house01.jpg"),
            grid_crop(HOUSE_TEMPLE_GRID, 211, 1592, 391, 1772),
        ),
        (
            "Medium Close-Up",
            ("G", ASSET_DIR / "geometry" / "house02.jpg"),
            grid_crop(HOUSE_TEMPLE_GRID, 397, 1592, 577, 1772),
        ),
        (
            "Long Shot",
            ("G", ASSET_DIR / "geometry" / "house07.jpeg"),
            grid_crop(HOUSE_TEMPLE_GRID, 1327, 1592, 1507, 1772),
        ),
    ],
}


FACE = {
    "title": "Face",
    "prompt": "A portrait of a man with a beard",
    "outputs": [
        (
            "Reality",
            ("G", ASSET_DIR / "geometry" / "face01.jpeg"),
            IMAGE_DIR / "face01__color004__texture008_A_portrait_of_a_man_with_a_beard.png",
        ),
        (
            "Abstract",
            ("G", ASSET_DIR / "geometry" / "face04.webp"),
            IMAGE_DIR / "face04__color004__texture008_A_portrait_of_a_man_with_a_beard.png",
        ),
        (
            "Flat Design",
            ("G", ASSET_DIR / "geometry" / "face11.jpg"),
            IMAGE_DIR / "face11__color004__texture008_A_portrait_of_a_man_with_a_beard.png",
        ),
    ],
}


SEEN = {
    "title": "Seen",
    "prompt": "A building of a beautiful church in a city",
    "outputs": [
        (
            "Two-Point",
            ("G", ASSET_DIR / "geometry" / "seen01.jpg"),
            IMAGE_DIR / "seen01__color007__texture006_A_building_of_a_beautiful_church_in_a_ci.png",
        ),
        (
            "Low-Angle",
            ("G", ASSET_DIR / "geometry" / "seen02.webp"),
            IMAGE_DIR / "seen02__color007__texture006_A_building_of_a_beautiful_church_in_a_ci.png",
        ),
        (
            "One-Point",
            ("G", ASSET_DIR / "geometry" / "seen03.png"),
            IMAGE_DIR / "seen03__color007__texture006_A_building_of_a_beautiful_church_in_a_ci.png",
        ),
    ],
}


PALETTE = {
    "bg": (248, 249, 250),
    "paper": (255, 255, 255),
    "ink": (24, 28, 34),
    "muted": (94, 103, 116),
    "line": (210, 216, 224),
    "geometry": (33, 127, 92),
    "color": (35, 116, 189),
    "texture": (176, 93, 42),
    "soft": (242, 244, 247),
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


FONT_TITLE = load_font(54, bold=True)
FONT_DOMAIN = load_font(44, bold=True)
FONT_HEADER = load_font(58, bold=True)
FONT_LABEL = load_font(32, bold=True)
FONT_SMALL = load_font(21, bold=True)
FONT_PROMPT = load_font(64, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font,
    fill=PALETTE["ink"],
) -> None:
    left, top, right, bottom = box
    width, height = text_size(draw, text, font)
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def fit_image(path, size: Tuple[int, int], background=PALETTE["paper"]) -> Image.Image:
    canvas = Image.new("RGB", size, background)
    if isinstance(path, tuple) and path[0] == "grid_crop":
        _, grid_path, crop_box = path
        if not grid_path.exists():
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 50, 50), width=4)
            draw.text((14, 14), "Missing grid", font=FONT_LABEL, fill=(140, 30, 30))
            draw.text((14, 44), grid_path.name[:28], font=FONT_SMALL, fill=(140, 30, 30))
            return canvas
        image = Image.open(grid_path).convert("RGB").crop(crop_box)
        image = ImageOps.contain(image, size, Image.LANCZOS)
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas

    if not path.exists():
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 50, 50), width=4)
        draw.text((14, 14), "Missing", font=FONT_LABEL, fill=(140, 30, 30))
        draw.text((14, 44), path.name[:28], font=FONT_SMALL, fill=(140, 30, 30))
        return canvas

    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, size, Image.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def center_crop_square(path: Path, size: int) -> Image.Image:
    if not path.exists():
        return fit_image(path, (size, size))

    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.LANCZOS)


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    colors = {
        "G": PALETTE["geometry"],
        "C": PALETTE["color"],
        "T": PALETTE["texture"],
    }
    fill = colors.get(label, PALETTE["ink"])
    draw.rounded_rectangle((x, y, x + 42, y + 34), radius=5, fill=fill)
    draw_centered(draw, (x, y, x + 42, y + 34), label, FONT_SMALL, fill=(255, 255, 255))


def draw_reference_stack(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    refs: Iterable[Tuple[str, Path]],
    x: int,
    y: int,
    size: int,
) -> None:
    labels = {"G": "Geometry", "C": "Color", "T": "Texture"}
    for index, (tag, path) in enumerate(refs):
        item_y = y + index * (size + 18)
        draw.rounded_rectangle(
            (x, item_y, x + size, item_y + size),
            radius=7,
            fill=PALETTE["paper"],
            outline=PALETTE["line"],
            width=2,
        )
        image = fit_image(path, (size - 8, size - 8))
        canvas.paste(image, (x + 4, item_y + 4))
        draw_tag(draw, x + size + 12, item_y + 3, tag)
        draw.text((x + size + 68, item_y + 2), labels[tag], font=FONT_LABEL, fill=PALETTE["ink"])


def draw_output_column(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    header: str,
    output_path: Path,
    changed_ref,
    image_size: int,
) -> None:
    draw_centered(draw, (x, y, x + image_size, y + 42), header, FONT_HEADER)
    image_y = y + 58

    draw.rounded_rectangle(
        (x, image_y, x + image_size, image_y + image_size),
        radius=8,
        fill=PALETTE["paper"],
        outline=PALETTE["line"],
        width=2,
    )
    output = fit_image(output_path, (image_size - 10, image_size - 10))
    canvas.paste(output, (x + 5, image_y + 5))

    if changed_ref is not None:
        tag, ref_path = changed_ref
        ref_size = 205
        ref_pad = 18
        ref_x = x + ref_pad
        ref_y = image_y + image_size - ref_size - ref_pad
        shadow = Image.new("RGBA", (ref_size + 14, ref_size + 14), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(
            (7, 7, ref_size + 13, ref_size + 13),
            radius=8,
            fill=(24, 28, 34, 76),
        )
        canvas.paste(shadow, (ref_x - 7, ref_y - 7), shadow)
        draw.rounded_rectangle(
            (ref_x, ref_y, ref_x + ref_size, ref_y + ref_size),
            radius=8,
            fill=PALETTE["paper"],
            outline=(255, 255, 255),
            width=5,
        )
        ref_img = center_crop_square(ref_path, ref_size - 10)
        canvas.paste(ref_img, (ref_x + 5, ref_y + 5))


def draw_group(canvas: Image.Image, draw: ImageDraw.ImageDraw, config: dict, x: int, y: int) -> None:
    group_w = 2390
    group_h = 1030
    pad = 34
    image_size = 740
    gap = 50

    draw.rounded_rectangle(
        (x, y, x + group_w, y + group_h),
        radius=10,
        fill=PALETTE["paper"],
        outline=PALETTE["line"],
        width=2,
    )

    draw.text((x + pad, y + 34), f'Prompt: "{config["prompt"]}"', font=FONT_PROMPT, fill=PALETTE["ink"])

    columns_x = x + pad
    for index, (header, changed_ref, output_path) in enumerate(config["outputs"]):
        col_x = columns_x + index * (image_size + gap)
        draw_output_column(canvas, draw, col_x, y + 165, header, output_path, changed_ref, image_size)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    width = 5020
    height = 2200
    canvas = Image.new("RGB", (width, height), PALETTE["bg"])
    draw = ImageDraw.Draw(canvas)

    left_x = 60
    right_x = 2570
    top_y = 40
    bottom_y = 1110

    draw_group(canvas, draw, ANIMAL, left_x, top_y)
    draw_group(canvas, draw, FACE, left_x, bottom_y)
    draw_group(canvas, draw, HOUSE, right_x, top_y)
    draw_group(canvas, draw, SEEN, right_x, bottom_y)

    canvas.save(OUTPUT_PATH)
    print(f"[SAVE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
