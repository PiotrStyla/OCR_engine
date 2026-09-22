"""Separate full-page interface for a confirmed image-capable remote endpoint.

No local neural model is loaded. Fabryka image support is not assumed.
"""
import base64
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import time


PROMPT = ("Transcribe the visible text in reading order. Preserve spelling, case, "
          "numbers and diacritics exactly as visible. Do not correct or infer missing "
          "characters. Return only the transcription; no explanation. Return an empty "
          "string for a blank page.")


@dataclass(frozen=True)
class PageTranscription:
    text: str
    requested_model: str
    returned_model: str | None
    source_sha256: str
    elapsed_seconds: float
    usage: dict | None

    def to_dict(self):
        return asdict(self)


def image_message(path, prompt=PROMPT):
    path = Path(path)
    content = path.read_bytes()
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif content.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    else:
        raise ValueError('Expected PNG or JPEG; render PDF pages first')
    return [{'type':'text','text':prompt},
            {'type':'image_url','image_url':{'url':f'data:{mime};base64,' + base64.b64encode(content).decode('ascii')}}]


class RemotePageParser:
    def __init__(self, client, model, max_tokens=2048):
        if not model or max_tokens <= 0:
            raise ValueError('Explicit model and positive token limit required')
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def parse(self, image):
        messages = [{'role':'user','content':image_message(image)}]
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens)
        choice = response.choices[0]
        if choice.finish_reason != 'stop':
            raise ValueError(f'Incomplete page response: finish_reason={choice.finish_reason}')
        if choice.message.content is None:
            raise ValueError('Endpoint returned no transcription content')
        usage = response.usage.model_dump() if response.usage is not None else None
        return PageTranscription(choice.message.content, self.model, response.model,
                                 hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                                 time.perf_counter()-start, usage)
