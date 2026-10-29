#!/usr/bin/env python3
"""Display fixed-scale K similarities with RGB queries and no cluster labels."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scan_vace_k_affinity import affinity_features, token_labels


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def cosine_colors(values):
    magnitude = np.abs(values)[..., None]
    endpoint = np.where((values >= 0)[..., None], [190, 28, 45], [35, 95, 185])
    return np.clip(255 * (1 - magnitude) + endpoint * magnitude, 0, 255).astype(np.uint8)


def panel(pixels, caption, cell, grid, width=320):
    pixels = np.asarray(pixels).copy()
    height, size = pixels.shape[:2]
    point = (int((cell % grid + .5) * size / grid), int((cell // grid + .5) * height / grid))
    cv2.drawMarker(pixels, point, (0, 0, 0), cv2.MARKER_CROSS, 11, 3)
    cv2.drawMarker(pixels, point, (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
    image = Image.new("RGB", (width, width + 44), "white")
    image.paste(Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST), (0, 44))
    ImageDraw.Draw(image).text((8, 12), caption, font=ImageFont.truetype(FONT, 15), fill="black")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--grid-queries", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input / f"{args.case}.npz") as source:
        rgb = source["rgb"][args.frame]
        frame_index = int(source["frame_indices"][args.frame])
        keys = source["layer_13_k"]
        names = source["label_names"].tolist()
        pixels = source["instance_labels"][args.frame]
    grid = int(round(keys.shape[-2] ** .5))
    features = [affinity_features(keys, args.frame, mode, head)
                for mode, head in (("raw", None), ("centered", None), ("centered", 8))]
    if args.grid_queries:
        positions = np.linspace(0, grid - 1, 4, dtype=int)
        queries = [(f"grid ({x},{y})", y * grid + x) for y in positions for x in positions]
    else:
        labels = token_labels(pixels, len(names), grid)
        queries = []
        for name in args.labels:
            mask = (labels.reshape(grid, grid) == names.index(name)).astype(np.uint8)
            if mask.any():
                cell = int(cv2.distanceTransform(mask, cv2.DIST_L2, 3).argmax())
                queries.append((name, cell))
    if not queries:
        raise ValueError("No visible query pixels")
    panels = []
    for name, cell in queries:
        panels.append(panel(rgb, name, cell, grid))
        for feat, caption in zip(features, ("T13 / 12 heads / raw", "T13 / 12 heads / centered", "T13H8 / centered")):
            heat = cosine_colors((feat @ feat[cell]).reshape(grid, grid))
            heat = cv2.resize(heat, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            panels.append(panel(heat, caption, cell, grid))
    width, height, gap = panels[0].width, panels[0].height, 8
    sheet = Image.new("RGB", (4 * width + 5 * gap, len(queries) * (height + gap) + 92), (235, 237, 240))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"{args.case} | RGB frame {frame_index} | frozen, empty prompt, observed video", font=ImageFont.truetype(FONT, 17), fill="black")
    for index, item in enumerate(panels):
        sheet.paste(item, (gap + index % 4 * (width + gap), 40 + index // 4 * (height + gap)))
    note = "Queries: fixed spatial grid, no GT" if args.grid_queries else "GT selects display queries only; no GT used to calculate heatmaps"
    draw.text((10, sheet.height - 40), "Fixed cosine scale: -1 blue | 0 white | +1 red. " + note, font=ImageFont.truetype(FONT, 15), fill="black")
    suffix = "grid_queries" if args.grid_queries else "objects"
    output = args.output / f"{args.case}_frame{frame_index:04d}_{suffix}.jpg"
    sheet.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
