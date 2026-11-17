"""Assemble the static browser app and its YOLOX model for GitHub Pages."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SITE = ROOT / "docs"
OUTPUT = ROOT / "dist"
LOCAL_MODEL = ROOT / "weights" / "yolox_s.onnx"
MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_s.onnx"
)
MIN_MODEL_SIZE = 1_000_000


def safe_reset_output() -> None:
    resolved_root = ROOT.resolve()
    resolved_output = OUTPUT.resolve()
    if resolved_output == resolved_root or resolved_root not in resolved_output.parents:
        raise RuntimeError(f"Refusing to replace output outside project: {resolved_output}")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)


def obtain_model(destination: Path) -> None:
    if LOCAL_MODEL.is_file():
        shutil.copy2(LOCAL_MODEL, destination)
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "roadwatch-pages-build/1.0"},
    )
    print(f"Downloading official YOLOX-S ONNX model from {MODEL_URL}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size < MIN_MODEL_SIZE:
            raise RuntimeError("Downloaded model is too small; release asset may be incomplete.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if not (SOURCE_SITE / "index.html").is_file():
        print(f"Static site entry point is missing: {SOURCE_SITE / 'index.html'}", file=sys.stderr)
        return 1

    safe_reset_output()
    shutil.copytree(SOURCE_SITE, OUTPUT, dirs_exist_ok=True)
    model_dir = OUTPUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    obtain_model(model_dir / "yolox_s.onnx")
    (OUTPUT / ".nojekyll").touch()
    print(f"GitHub Pages artifact prepared: {OUTPUT}")
    print(f"Model size: {(model_dir / 'yolox_s.onnx').stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
