import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INFER_SCRIPT = SCRIPT_DIR / "infer_style_plus_color_texture.py"

MODE_INPUTS = [
    ("color_only", True, False, False),
    ("color_texture_legacy", True, False, True),
    ("color_geometry", True, True, False),
    ("color_texture_geometry", True, True, True),
]


def _read_literal_assignments(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in names or name in values:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
        except Exception:
            continue
    return values


def _resolve_for_check(raw_path):
    if raw_path is None:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return SCRIPT_DIR / candidate


def _validate_input(raw_path, label):
    if raw_path is None:
        raise ValueError(f"{label} input is empty. Set it in infer_style_plus_color_texture.py or pass it by CLI.")
    resolved = _resolve_for_check(raw_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} input not found: {resolved}")
    return raw_path


def _build_mode_env(base_env, color_path, geometry_path, texture_path, save_dir):
    env = dict(base_env)
    env["SADIS_INPUT_COLOR"] = color_path if color_path is not None else "None"
    env["SADIS_INPUT_GEOMETRY"] = geometry_path if geometry_path is not None else "None"
    env["SADIS_INPUT_TEXTURE"] = texture_path if texture_path is not None else "None"
    env["SADIS_SAVE_DIR"] = str(save_dir)
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Run infer_style_plus_color_texture.py across color-related input mode combinations."
    )
    parser.add_argument("--color", help="Override the color input path.")
    parser.add_argument("--geometry", help="Override the geometry input path.")
    parser.add_argument("--texture", help="Override the texture input path.")
    parser.add_argument(
        "--save-root",
        help="Root output directory for all mode runs. Defaults to <save_dir>/all_modes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the mode plan without executing infer_style_plus_color_texture.py.",
    )
    args = parser.parse_args()

    defaults = _read_literal_assignments(
        INFER_SCRIPT,
        {"INPUT_COLOR", "INPUT_GEOMETRY", "INPUT_TEXTURE", "save_dir"},
    )

    color_input = args.color if args.color is not None else defaults.get("INPUT_COLOR")
    geometry_input = args.geometry if args.geometry is not None else defaults.get("INPUT_GEOMETRY")
    texture_input = args.texture if args.texture is not None else defaults.get("INPUT_TEXTURE")
    base_save_dir = args.save_root if args.save_root is not None else os.path.join(
        defaults.get("save_dir", "results"),
        "all_modes",
    )

    color_input = _validate_input(color_input, "Color")
    geometry_input = _validate_input(geometry_input, "Geometry")
    texture_input = _validate_input(texture_input, "Texture")

    print(f"Running {len(MODE_INPUTS)} color-related modes with inputs:")
    print(f"  color: {color_input}")
    print(f"  geometry: {geometry_input}")
    print(f"  texture: {texture_input}")
    print(f"  save_root: {base_save_dir}")

    failures = []
    for mode_name, use_color, use_geometry, use_texture in MODE_INPUTS:
        mode_save_dir = Path(base_save_dir) / mode_name
        mode_save_dir.mkdir(parents=True, exist_ok=True)

        env = _build_mode_env(
            os.environ,
            color_input if use_color else None,
            geometry_input if use_geometry else None,
            texture_input if use_texture else None,
            mode_save_dir,
        )

        command = [sys.executable, str(INFER_SCRIPT)]
        print(f"\n=== Running {mode_name} ===")
        print(f"save_dir: {mode_save_dir}")
        if args.dry_run:
            print("dry-run command:", " ".join(command))
            continue

        result = subprocess.run(command, cwd=str(SCRIPT_DIR), env=env, check=False)
        if result.returncode != 0:
            failures.append((mode_name, result.returncode))
            print(f"{mode_name} failed with exit code {result.returncode}")
        else:
            print(f"{mode_name} completed")

    if failures:
        print("\nCompleted with failures:")
        for mode_name, returncode in failures:
            print(f"  {mode_name}: exit code {returncode}")
        raise SystemExit(1)

    print(f"\nAll {len(MODE_INPUTS)} color-related modes completed successfully.")


if __name__ == "__main__":
    main()
