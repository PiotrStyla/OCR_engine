from ocr.detector import sort_reading_order_columns
from ocr.result import BBox
from training.pagexml_order import order_by_pagexml_regions, parse_pagexml_regions


def test_column_order_reads_left_column_before_right_column():
    boxes = [
        BBox(10, 10, 90, 30),
        BBox(110, 10, 190, 30),
        BBox(10, 40, 90, 60),
        BBox(110, 40, 190, 60),
    ]
    ordered = sort_reading_order_columns(boxes, image_width=200, image_height=100)
    assert [(box.x1, box.y1) for box in ordered] == [
        (10, 10), (10, 40), (110, 10), (110, 40)
    ]


def test_column_order_retains_spanning_header_and_separator():
    header = BBox(5, 0, 195, 20)
    separator = BBox(5, 70, 195, 90)
    boxes = [
        BBox(10, 30, 90, 50), BBox(110, 30, 190, 50), header,
        BBox(10, 100, 90, 120), BBox(110, 100, 190, 120), separator,
        BBox(10, 130, 90, 150), BBox(110, 130, 190, 150),
    ]
    ordered = sort_reading_order_columns(boxes, image_width=200, image_height=170)
    assert ordered == [
        header,
        BBox(10, 30, 90, 50), BBox(110, 30, 190, 50),
        separator,
        BBox(10, 100, 90, 120), BBox(10, 130, 90, 150),
        BBox(110, 100, 190, 120), BBox(110, 130, 190, 150),
    ]


def test_column_order_falls_back_for_single_column():
    boxes = [BBox(20, y, 180, y + 20) for y in (80, 10, 45, 120)]
    ordered = sort_reading_order_columns(boxes, image_width=200, image_height=160)
    assert [box.y1 for box in ordered] == [10, 45, 80, 120]


PAGE_XML = b'''<PcGts xmlns="urn:test"><Page><ReadingOrder><OrderedGroup>
<RegionRefIndexed index="0" regionRef="right"/>
<RegionRefIndexed index="1" regionRef="left"/>
</OrderedGroup></ReadingOrder>
<TextRegion id="left"><Coords points="0,0 90,0 90,100 0,100"/>
<TextEquiv><Unicode>left</Unicode></TextEquiv></TextRegion>
<TextRegion id="right"><Coords points="110,0 200,0 200,100 110,100"/>
<TextEquiv><Unicode>right</Unicode></TextEquiv></TextRegion>
</Page></PcGts>'''


def test_pagexml_parser_honors_explicit_region_order():
    regions = parse_pagexml_regions(PAGE_XML)
    assert [region.identifier for region in regions] == ["right", "left"]


def test_pagexml_order_is_a_complete_permutation():
    left = BBox(10, 10, 80, 30)
    right = BBox(120, 10, 190, 30)
    outside = BBox(145, 150, 155, 160)
    ordered, stats = order_by_pagexml_regions(
        [left, outside, right], PAGE_XML, image_height=200
    )
    assert ordered[:2] == [right, outside]
    assert ordered[2:] == [left]
    assert stats == {
        "text_regions": 2,
        "overlap_assigned_lines": 2,
        "nearest_assigned_lines": 1,
        "regions_with_assigned_lines": 2,
    }
