import pytest
from training.kaggle_ehri_control import parse_alto, normalize


def fixture(tmp_path, geometry='1 2 8 2', strings='<String CONTENT="one"/><SP/><String CONTENT="two"/>'):
    path = tmp_path / 'sample.xml'
    path.write_text(f'<alto xmlns="urn:alto"><Description><MeasurementUnit>pixel</MeasurementUnit></Description><Layout><Page WIDTH="10" HEIGHT="10"><TextLine ID="a" BASELINE="{geometry}"><Shape><Polygon POINTS="1 1 8 1 8 3 1 3"/></Shape>{strings}</TextLine></Page></Layout></alto>')
    return path


def test_nested_polygon_and_word_spacing(tmp_path):
    size, rows = parse_alto(fixture(tmp_path))
    assert size == (10, 10)
    assert rows[0]['text'] == 'one two'
    assert len(rows[0]['boundary']) == 4


@pytest.mark.parametrize('geometry', ['1 2 8', '1 2 11 2', 'nan 1 2 3'])
def test_invalid_geometry_rejected(tmp_path, geometry):
    with pytest.raises(ValueError):
        parse_alto(fixture(tmp_path, geometry))


def test_empty_text_is_retained(tmp_path):
    _, rows = parse_alto(fixture(tmp_path, strings='<String CONTENT=""/>'))
    assert rows[0]['text'] == ''
    assert normalize(' e\u0301\n  x ') == '\u00e9 x'
