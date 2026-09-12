from types import SimpleNamespace
import pytest
from PIL import Image
from ocr.page_parser import RemotePageParser


def test_full_page_contract_and_truncation(tmp_path):
    path = tmp_path/'page.png'
    Image.new('RGB',(20,30),'white').save(path)
    seen=[]
    choice=SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='literal text'))
    response=SimpleNamespace(choices=[choice],model='actual-model',usage=None)
    def create(**kwargs):
        seen.append(kwargs)
        return response
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    parser=RemotePageParser(client,'explicit-model')
    result=parser.parse(path)
    assert result.text=='literal text'
    assert result.returned_model=='actual-model'
    assert seen[0]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    choice.finish_reason='length'
    with pytest.raises(ValueError,match='Incomplete'):
        parser.parse(path)
