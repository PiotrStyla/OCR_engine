"""Human-seeded newspaper column crops with OpenCV separator diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def segment(image_path, config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    with Image.open(image_path) as source:
        source.load()
        width, height = source.size
        rw, rh = config["coordinate_size"]
        xs = [round(x * width / rw) for x in config["column_edges"]]
        top, body, bottom = [round(y * height / rh) for y in config["vertical_edges"]]
        if not (0 <= top < body < bottom <= height and
                all(0 <= a < b <= width for a, b in zip(xs, xs[1:]))):
            raise ValueError("Invalid region boundaries")
        gray = np.array(source.convert("L"))
        small = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
        ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
        rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100)))
        cv2.imwrite(str(output / "vertical-rules.png"), rules)
        regions = [("00-masthead", (xs[0], top, xs[-1], body))]
        regions += [(f"{i+1:02d}-column", (a, body, b, bottom))
                    for i, (a, b) in enumerate(zip(xs, xs[1:]))]
        manifest = []
        preview = source.convert("RGB")
        preview.thumbnail((rw, rh))
        draw = ImageDraw.Draw(preview)
        for order, (name, box) in enumerate(regions):
            relative = f"crops/{name}.png"
            crop = source.crop(box)
            crop.save(output / relative)
            with Image.open(output / relative) as check:
                assert check.tobytes() == crop.tobytes()
            manifest.append({"id": name, "image": relative, "bbox_xyxy": box,
                             "reading_order": order, "reference_available": False,
                             "sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest()})
            rect = [round(box[0]*preview.width/width), round(box[1]*preview.height/height),
                    round(box[2]*preview.width/width), round(box[3]*preview.height/height)]
            color = (210, 20, 35) if order % 2 else (0, 110, 220)
            draw.rectangle(rect, outline=color, width=3)
            draw.rectangle((rect[0], rect[1], rect[0]+90, rect[1]+18), fill="white")
            draw.text((rect[0]+3, rect[1]+2), name, fill=color)
        preview.save(output / "segmentation-preview.png")
    (output / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in manifest), encoding="utf-8")
    report = {"method": "Human-selected boundaries; OpenCV vertical-rule diagnostic only, not automatic article segmentation",
              "source": str(image_path.resolve()), "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "config": config, "source_size": [width, height], "regions": len(manifest),
              "opencv": cv2.__version__, "crop_pixels": "Unchanged; no deskew, resize or contrast adjustment",
              "limitations": "Left-to-right column order; article continuations and internal article boundaries not resolved. Margins excluded."}
    (output / "segmentation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    checksums = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    segment(args.image, args.config, args.output)
