"""Build a private, static bbox review panel from consensus evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath


LAYERS = {
    "qwen3-vl-4b": "#7c3aed",
    "doclayout-yolo": "#007c91",
    "surya-layout2": "#d97706",
    "accepted": "#16803c",
    "review": "#d22f27",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _safe_members(archive: zipfile.ZipFile) -> dict[str, bytes]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP member")
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {name}")
    return {name: archive.read(name) for name in names if not name.endswith("/")}


def _verify_checksums(files: dict[str, bytes]) -> None:
    for line in files["consensus/checksums.sha256"].decode("utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if digest(files[f"consensus/{name}"]) != expected:
            raise ValueError(f"Consensus checksum mismatch: {name}")
    for teacher in ("qwen3-vl-4b", "doclayout-yolo", "surya-layout2"):
        prefix = f"teachers/{teacher}/"
        checksums = json.loads(files[prefix + "checksums.json"])
        for name, expected in checksums.items():
            if digest(files[prefix + name]) != expected:
                raise ValueError(f"Teacher checksum mismatch: {teacher}/{name}")


def _box(item: dict) -> dict:
    return {
        "id": item["id"],
        "label": item["label"],
        "bbox": item["bbox_xyxy"],
        "score": item.get("score", item.get("mean_score")),
        "teachers": item.get("teachers", []),
        "reasons": item.get("reasons", []),
    }


def build_review(archive_path: str | Path, image_dir: str | Path,
                 output_path: str | Path) -> Path:
    archive_path, image_dir, output_path = map(Path, (archive_path, image_dir, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    with zipfile.ZipFile(archive_path) as archive:
        files = _safe_members(archive)
    _verify_checksums(files)
    run = json.loads(files["run.json"])
    consensus = _jsonl(files["consensus/consensus.jsonl"])
    reviews = _jsonl(files["consensus/review-queue.jsonl"])
    by_page = {
        row["page_id"]: {
            "page_id": row["page_id"],
            "image": row["image"],
            "accepted": [_box(item) for item in row["objects"]],
            "review": [],
            "teachers": {},
        }
        for row in consensus
    }
    for item in reviews:
        by_page[item["page_id"]]["review"].append(_box(item))
    for teacher in ("qwen3-vl-4b", "doclayout-yolo", "surya-layout2"):
        rows = _jsonl(files[f"teachers/{teacher}/teacher-proposals.jsonl"])
        for row in rows:
            by_page[row["page_id"]]["teachers"][teacher] = [
                _box(item) for item in row["detections"]
            ]
    pages = []
    for page in by_page.values():
        image_name = Path(page["image"]["file_name"]).name
        image_path = image_dir / image_name
        if not image_path.is_file() or digest(image_path.read_bytes()) != page["image"]["sha256"]:
            raise ValueError(f"Missing or mismatched image: {image_name}")
        page["image_src"] = os.path.relpath(image_path, output_path.parent).replace("\\", "/")
        pages.append(page)
    payload = json.dumps({
        "archive_sha256": digest(archive_path.read_bytes()),
        "run": run,
        "pages": pages,
        "layers": LAYERS,
    }, ensure_ascii=False).replace("<", "\\u003c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html(payload), encoding="utf-8", newline="\n")
    return output_path


def _html(payload: str) -> str:
    return f'''<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLAYER layout review</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; color: #172033; background: #eef1f5; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100vh; }}
header {{ height: 58px; display: flex; align-items: center; gap: 18px; padding: 0 20px; background: #161b26; color: white; }}
header strong {{ font-size: 17px; }}
#summary {{ color: #cbd2df; font-size: 13px; }}
.layout {{ display: grid; grid-template-columns: 270px minmax(0, 1fr); min-height: calc(100vh - 58px); }}
aside {{ padding: 16px; background: white; border-right: 1px solid #d5dae3; }}
.pages {{ display: grid; gap: 6px; margin-bottom: 18px; }}
.page-button {{ width: 100%; min-height: 42px; padding: 8px 10px; border: 1px solid #ccd3df; background: white; color: #263247; text-align: left; cursor: pointer; font-weight: 600; }}
.page-button.active {{ border-color: #007c91; box-shadow: inset 3px 0 #007c91; background: #f2fbfc; }}
.layers {{ display: grid; gap: 10px; }}
.layer {{ display: grid; grid-template-columns: 18px 12px 1fr auto; align-items: center; gap: 8px; font-size: 13px; }}
.swatch {{ width: 12px; height: 12px; border-radius: 2px; }}
.count {{ color: #657086; font-variant-numeric: tabular-nums; }}
main {{ min-width: 0; padding: 16px; display: grid; grid-template-rows: minmax(420px, 1fr) 220px; gap: 12px; }}
.viewer {{ min-height: 0; overflow: auto; background: #262b35; border: 1px solid #1b2029; }}
svg {{ display: block; margin: 0 auto; width: min(100%, 1050px); height: auto; background: white; }}
.bbox {{ fill: none; vector-effect: non-scaling-stroke; stroke-width: 3; }}
.bbox-label {{ font-size: 25px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 5px; stroke-linejoin: round; }}
.details {{ overflow: auto; background: white; border: 1px solid #d5dae3; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e6ed; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f7f8fa; color: #4a566c; }}
code {{ font-family: Consolas, monospace; }}
@media (max-width: 780px) {{ .layout {{ grid-template-columns: 1fr; }} aside {{ border-right: 0; border-bottom: 1px solid #d5dae3; }} main {{ grid-template-rows: minmax(360px, 70vh) 220px; }} }}
</style>
</head>
<body>
<header><strong>SLAYER layout review</strong><span id="summary"></span></header>
<div class="layout">
  <aside><div class="pages" id="pages"></div><div class="layers" id="layers"></div></aside>
  <main><div class="viewer" id="viewer"></div><div class="details"><table><thead><tr><th>Warstwa</th><th>Klasa</th><th>Score</th><th>Teachers / powód</th><th>Bbox</th></tr></thead><tbody id="rows"></tbody></table></div></main>
</div>
<script>const DATA={payload};
const state={{page:0, visible:new Set(Object.keys(DATA.layers))}};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function allLayers(page){{return [
  ...Object.entries(page.teachers).map(([name,boxes])=>[name,boxes]),
  ['accepted',page.accepted],['review',page.review]
]}}
function render(){{
 const page=DATA.pages[state.page], meta=page.image;
 document.getElementById('summary').textContent=`${{DATA.run.pages}} strony · ${{DATA.run.consensus.accepted_objects}} accepted · ${{DATA.run.consensus.review_objects}} review`;
 document.getElementById('pages').innerHTML=DATA.pages.map((p,i)=>`<button class="page-button ${{i===state.page?'active':''}}" data-page="${{i}}">${{esc(p.page_id)}}</button>`).join('');
 document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>{{state.page=Number(b.dataset.page);render()}});
 const layers=allLayers(page);
 document.getElementById('layers').innerHTML=layers.map(([name,boxes])=>`<label class="layer"><input type="checkbox" data-layer="${{name}}" ${{state.visible.has(name)?'checked':''}}><span class="swatch" style="background:${{DATA.layers[name]}}"></span><span>${{esc(name)}}</span><span class="count">${{boxes.length}}</span></label>`).join('');
 document.querySelectorAll('[data-layer]').forEach(c=>c.onchange=()=>{{c.checked?state.visible.add(c.dataset.layer):state.visible.delete(c.dataset.layer);renderCanvas()}});
 renderCanvas();
}}
function renderCanvas(){{
 const page=DATA.pages[state.page], meta=page.image;
 let shapes='', rows='';
 for(const [layer,boxes] of allLayers(page)){{if(!state.visible.has(layer))continue; const color=DATA.layers[layer];
  boxes.forEach((b,i)=>{{const [x1,y1,x2,y2]=b.bbox, label=`${{layer}} · ${{b.label}}`;
   shapes+=`<g><rect class="bbox" x="${{x1}}" y="${{y1}}" width="${{x2-x1}}" height="${{y2-y1}}" stroke="${{color}}"/><text class="bbox-label" x="${{x1+5}}" y="${{Math.max(28,y1+28)}}" fill="${{color}}">${{esc(label)}}</text></g>`;
   const note=b.reasons?.length?b.reasons.join(', '):(b.teachers?.join(', ')||'');
   rows+=`<tr><td style="color:${{color}};font-weight:700">${{esc(layer)}}</td><td>${{esc(b.label)}}</td><td>${{b.score==null?'—':Number(b.score).toFixed(3)}}</td><td>${{esc(note)}}</td><td><code>${{b.bbox.map(v=>Math.round(v)).join(', ')}}</code></td></tr>`;
  }});
 }}
 document.getElementById('viewer').innerHTML=`<svg viewBox="0 0 ${{meta.width}} ${{meta.height}}" aria-label="${{esc(page.page_id)}}"><image href="${{page.image_src}}" width="${{meta.width}}" height="${{meta.height}}"/>${{shapes}}</svg>`;
 document.getElementById('rows').innerHTML=rows;
}}
render();</script>
</body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_review(args.evidence, args.images, args.output))


if __name__ == "__main__":
    main()
