"""PAGE XML region-order diagnostics for detected OCR line boxes."""
from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree

from ocr.detector import sort_reading_order
from ocr.result import BBox


@dataclass(frozen=True)
class PageRegion:
    identifier: str
    order: int
    bbox: BBox
    text: str


def _region_bbox(region) -> BBox | None:
    coords = region.find("./{*}Coords")
    if coords is None:
        return None
    points = []
    for point in coords.findall("./{*}Point"):
        points.append((int(point.attrib["x"]), int(point.attrib["y"])))
    if not points and coords.attrib.get("points"):
        points = [
            tuple(map(int, value.split(",")))
            for value in coords.attrib["points"].split()
        ]
    if not points:
        return None
    xs, ys = zip(*points)
    return BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def parse_pagexml_regions(xml: bytes | str) -> list[PageRegion]:
    root = ElementTree.fromstring(xml)
    ordered_ids = {
        item.attrib["regionRef"]: int(item.attrib["index"])
        for item in root.findall(".//{*}RegionRefIndexed")
        if "regionRef" in item.attrib and "index" in item.attrib
    }
    regions = []
    for element in root.findall(".//{*}TextRegion"):
        identifier = element.attrib.get("id")
        bbox = _region_bbox(element)
        direct = element.find("./{*}TextEquiv/{*}Unicode")
        if direct is not None:
            text = direct.text or ""
        else:
            text = "\n".join(
                node.text or ""
                for node in element.findall(".//{*}TextLine/{*}TextEquiv/{*}Unicode")
            )
        if not identifier or bbox is None or not text.strip():
            continue
        regions.append(PageRegion(
            identifier=identifier,
            order=ordered_ids.get(identifier, 1_000_000),
            bbox=bbox,
            text=text,
        ))
    regions.sort(key=lambda item: (item.order, item.bbox.y1, item.bbox.x1, item.identifier))
    return regions


def _overlap_area(left: BBox, right: BBox) -> int:
    width = min(left.x2, right.x2) - max(left.x1, right.x1)
    height = min(left.y2, right.y2) - max(left.y1, right.y1)
    return max(0, width) * max(0, height)


def order_by_pagexml_regions(
    bboxes: list[BBox], xml: bytes | str, image_height: int
) -> tuple[list[BBox], dict]:
    """Order every detected box by annotated PAGE region order.

    Boxes overlapping no text region are assigned to the nearest region so the
    diagnostic remains a permutation rather than silently dropping detections.
    """
    regions = parse_pagexml_regions(xml)
    if not regions:
        return sort_reading_order(bboxes, image_height), {
            "text_regions": 0,
            "overlap_assigned_lines": 0,
            "nearest_assigned_lines": len(bboxes),
            "regions_with_assigned_lines": 0,
        }

    groups = [[] for _ in regions]
    overlap_assigned = 0
    for box in bboxes:
        overlaps = [_overlap_area(box, region.bbox) for region in regions]
        best = max(range(len(regions)), key=lambda index: (overlaps[index], -index))
        if overlaps[best] > 0:
            overlap_assigned += 1
        else:
            center_x = (box.x1 + box.x2) / 2
            center_y = (box.y1 + box.y2) / 2
            best = min(
                range(len(regions)),
                key=lambda index: (
                    (center_x - (regions[index].bbox.x1 + regions[index].bbox.x2) / 2) ** 2
                    + (center_y - (regions[index].bbox.y1 + regions[index].bbox.y2) / 2) ** 2,
                    index,
                ),
            )
        groups[best].append(box)

    ordered = []
    for group in groups:
        ordered.extend(sort_reading_order(group, image_height))
    if len(ordered) != len(bboxes):
        raise RuntimeError("PAGE ordering did not preserve the detected boxes")
    return ordered, {
        "text_regions": len(regions),
        "overlap_assigned_lines": overlap_assigned,
        "nearest_assigned_lines": len(bboxes) - overlap_assigned,
        "regions_with_assigned_lines": sum(bool(group) for group in groups),
    }
