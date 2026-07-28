from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
IMAGE_DIR = ROOT / "results" / "dog_tuning_experiments0508" / "images"
OUT_DIR = ROOT / "results" / "dog_tuning_experiments0508" / "teaser_candidates"


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(34, True)
FONT_LABEL = font(24, True)
FONT_SMALL = font(18)


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    if not path.exists():
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 40, 40), width=4)
        draw.text((14, 14), "Missing", font=FONT_LABEL, fill=(180, 40, 40))
        draw.text((14, 48), path.name[:35], font=FONT_SMALL, fill=(180, 40, 40))
        return canvas
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_sheet(title: str, entries: list[tuple[str, Path, Path]], output_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_w = 360
    cell_h = 560
    ref_h = 150
    out_h = 330
    cols = 4
    rows = (len(entries) + cols - 1) // cols
    width = cols * cell_w + 60
    height = rows * cell_h + 110
    canvas = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 26), title, font=FONT_TITLE, fill=(24, 28, 34))

    for index, (label, ref_path, out_path) in enumerate(entries):
        col = index % cols
        row = index // cols
        x = 30 + col * cell_w
        y = 82 + row * cell_h
        draw.text((x, y), label, font=FONT_LABEL, fill=(24, 28, 34))
        ref = fit(ref_path, (cell_w - 40, ref_h))
        out = fit(out_path, (cell_w - 40, out_h))
        draw.rectangle((x, y + 38, x + cell_w - 24, y + 38 + ref_h), outline=(210, 216, 224), width=2)
        canvas.paste(ref, (x + 12, y + 40))
        draw.rectangle((x, y + 210, x + cell_w - 24, y + 210 + out_h), outline=(210, 216, 224), width=2)
        canvas.paste(out, (x + 12, y + 212))

    path = OUT_DIR / output_name
    canvas.save(path)
    print(path)


def main() -> None:
    geometries = [f"house{i:02d}" for i in range(1, 9)]
    colors = ["color002", "color003", "color004", "color005", "color006", "color007"]
    color_ext = {
        "color002": "jpeg",
        "color003": "jpeg",
        "color004": "jpeg",
        "color005": "jpeg",
        "color006": "png",
        "color007": "jpeg",
    }
    textures = ["001", "8", "artwork_167", "artwork_3", "texture003", "texture004", "texture005", "texture006", "texture007", "texture008"]
    texture_ext = {
        "001": "jpg",
        "8": "png",
        "artwork_167": "jpg",
        "artwork_3": "jpg",
        "texture003": "jpeg",
        "texture004": "jpeg",
        "texture005": "jpeg",
        "texture006": "jpeg",
        "texture007": "jpeg",
        "texture008": "jpeg",
    }
    geometry_ext = {"house01": "jpg", "house02": "jpg", "house03": "jpg", "house04": "jpeg", "house05": "jpeg", "house06": "jpeg", "house07": "jpeg", "house08": "jpeg"}

    make_sheet(
        "House Geometry Candidates: fixed color003 + artwork_167",
        [
            (
                geo,
                ASSET_DIR / "geometry" / f"{geo}.{geometry_ext[geo]}",
                IMAGE_DIR / f"{geo}__color003__artwork_167_One_beautiful_church_on_the_ground.png",
            )
            for geo in geometries
        ],
        "house_geometry_candidates.png",
    )

    make_sheet(
        "House Geometry Candidates: fixed color004 + artwork_167",
        [
            (
                geo,
                ASSET_DIR / "geometry" / f"{geo}.{geometry_ext[geo]}",
                IMAGE_DIR / f"{geo}__color004__artwork_167_One_beautiful_church_on_the_ground.png",
            )
            for geo in geometries
        ],
        "house_geometry_candidates_color004.png",
    )

    make_sheet(
        "House Color Candidates: fixed house04 + artwork_167",
        [
            (
                color,
                ASSET_DIR / "color" / f"{color}.{color_ext[color]}",
                IMAGE_DIR / f"house04__{color}__artwork_167_One_beautiful_church_on_the_ground.png",
            )
            for color in colors
        ],
        "house_color_candidates.png",
    )

    make_sheet(
        "House Texture Candidates: fixed house04 + color003",
        [
            (
                texture,
                ASSET_DIR / "texture" / f"{texture}.{texture_ext[texture]}",
                IMAGE_DIR / f"house04__color003__{texture}_One_beautiful_church_on_the_ground.png",
            )
            for texture in textures
        ],
        "house_texture_candidates.png",
    )

    make_sheet(
        "House Texture Candidates: fixed house04 + color004",
        [
            (
                texture,
                ASSET_DIR / "texture" / f"{texture}.{texture_ext[texture]}",
                IMAGE_DIR / f"house04__color004__{texture}_One_beautiful_church_on_the_ground.png",
            )
            for texture in textures
        ],
        "house_texture_candidates_color004.png",
    )


if __name__ == "__main__":
    main()
