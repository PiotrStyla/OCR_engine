"""Bounded Wikisource scan acquisition; source validation is not our own review."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from training.full_page_pilot import digest, read_rows, safe_relative, write_json, write_rows


WORKS = [
    {"id": "zeromski-dzieje-grzechu", "file": "PL Stefan \u017beromski - Dzieje grzechu 01.djvu",
     "sha1": "6282f3c246246216d412d9aea5796d599ce96c17", "year": 1928,
     "split": "replay-candidate", "pages": [14, 31, 52, 165, 238, 243, 351, 381]},
    {"id": "prus-lalka", "file": "PL Boles\u0142aw Prus - Lalka Tom2.djvu",
     "sha1": "9502f67f514f92d8a4db0326e33193e3c5c9857b", "year": 1890,
     "split": "replay-probe", "pages": [128, 130, 275, 501]},
]
PROTECTED_WORKS = ["krolicki-odezwa-do-matek", "torunski-elementarz-1910",
                   "protective-family-nowe-ateny", "protective-family-wyprawa-1634"]
USER_AGENT = "SLAYER-OCR-source-pilot/1.0 (https://github.com/PiotrStyla/OCR_engine)"


def download(url, limit=8_000_000):
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in {
        "pl.wikisource.org", "commons.wikimedia.org", "upload.wikimedia.org",
        "thumb.wikimedia.org"}:
        raise ValueError("Unexpected source host")
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=90) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Source exceeds bounded download")
    return body


def api(host, parameters):
    url = f"https://{host}/w/api.php?" + urlencode({
        "format": "json", "formatversion": 2, **parameters})
    body = download(url)
    result = json.loads(body)
    if "error" in result:
        raise ValueError(result["error"])
    return result, body, url


def reference_from_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    bodies = soup.select("div.pagetext")
    if len(bodies) != 1:
        raise ValueError("Expected exactly one proofread page body")
    body = bodies[0]
    if body.select("table, .reference, .references, script, style"):
        raise ValueError("Complex/footnote body needs a separate extraction protocol")
    for node in body.select("br"):
        node.replace_with("\n")
    text = body.get_text().strip()
    if len(text) < 150 or "\ufffd" in text:
        raise ValueError("Invalid or too short source transcription")
    return text


def quality_level(wikitext):
    import re
    tags = re.findall(r"<pagequality\b[^>]*?/?>", wikitext)
    if len(tags) != 1:
        raise ValueError("Missing/ambiguous proofread status")
    return int(ElementTree.fromstring(tags[0]).attrib["level"])


def validate_works(works):
    names = [work["id"] for work in works]
    if len(set(names)) != len(names) or set(names) & set(PROTECTED_WORKS):
        raise ValueError("Duplicate/protected work")
    if {w["split"] for w in works} != {"replay-candidate", "replay-probe"}:
        raise ValueError("Require separate candidate/probe works")
    if any(not w["pages"] or len(set(w["pages"])) != len(w["pages"])
           or any(type(p) is not int or p < 1 for p in w["pages"]) for w in works):
        raise ValueError("Invalid page selection")
    files = [w["file"] for w in works]
    hashes = [w["sha1"] for w in works]
    if len(set(files)) != len(files) or len(set(hashes)) != len(hashes):
        raise ValueError("Repeated scan across work identities/splits")
    for work in works:
        safe_relative(work["id"])
        if "/" in work["id"] or not work["file"].endswith(".djvu") or (
                len(work["sha1"]) != 40 or any(c not in "0123456789abcdef" for c in work["sha1"])):
            raise ValueError("Invalid work identifier/scan hash")


def collect(output, *, works=None):
    from PIL import Image
    output = Path(output)
    works = WORKS if works is None else works
    validate_works(works)
    output.mkdir(parents=True, exist_ok=False)
    rows, rejected = [], []
    write_json(output / "source-policy.json", {
        "schema": "slayer-printed-replay-source-policy-v1", "works": works,
        "protected_work_families": PROTECTED_WORKS, "required_quality": 4,
        "status": "source-pilot-not-training-data", "evaluation_eligible": False,
        "normalization": "No spelling changes; rendered page body retained verbatim",
        "limitations": "Upstream validated historical print, not project-reviewed crops or representative modern documents; not PolEval split reconstruction."})
    for work in works:
        for page in work["pages"]:
            identifier = f"{work['id']}-{page:04d}"
            title = f"Strona:{work['file']}/{page}"
            directory = output / "sources" / identifier
            directory.mkdir(parents=True)
            try:
                revision, raw, url = api("pl.wikisource.org", {
                    "action": "query", "titles": title, "prop": "revisions",
                    "rvprop": "ids|timestamp|content", "rvslots": "main"})
                (directory / "revision.json").write_bytes(raw)
                source_page = revision["query"]["pages"][0]
                rev = source_page["revisions"][0]
                raw_text = rev["slots"]["main"]["content"]
                (directory / "source.wikitext").write_text(raw_text, encoding="utf-8")
                if quality_level(raw_text) != 4:
                    raise ValueError("Source page is not independently validated (quality 4)")
                parsed, raw, parse_url = api("pl.wikisource.org", {
                    "action": "parse", "oldid": rev["revid"], "prop": "text|revid"})
                (directory / "parse.json").write_bytes(raw)
                if parsed["parse"]["revid"] != rev["revid"]:
                    raise ValueError("Rendered source revision mismatch")
                text = reference_from_html(parsed["parse"]["text"])
                (directory / "reference.txt").write_text(text, encoding="utf-8")
                info, raw, info_url = api("commons.wikimedia.org", {
                    "action": "query", "titles": "File:" + work["file"],
                    "prop": "imageinfo", "iiprop": "url|sha1|extmetadata|size",
                    "iiurlwidth": 1600, "iiurlparam": f"page{page}"})
                (directory / "imageinfo.json").write_bytes(raw)
                imageinfo = info["query"]["pages"][0]["imageinfo"][0]
                license_metadata = imageinfo["extmetadata"]
                if imageinfo["sha1"] != work["sha1"]:
                    raise ValueError("Original scan SHA1 changed")
                if license_metadata["License"]["value"] != "pd":
                    raise ValueError("Expected public-domain scan license")
                image_url = imageinfo["thumburl"]
                image_bytes = download(image_url)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    if image.format != "JPEG" or min(width, height) < 400:
                        raise ValueError("Unexpected/too small scan derivative")
                image_path = directory / "page.jpg"
                image_path.write_bytes(image_bytes)
                rows.append({"id": identifier, "work_family": work["id"],
                    "domain": work.get("domain", "historical-print"),
                    "split": work["split"], "year": work["year"], "scan_page": page,
                    "source_title": source_page["title"], "source_page_id": source_page["pageid"],
                    "source_revision": rev["revid"], "source_timestamp": rev["timestamp"],
                    "source_url": f"https://pl.wikisource.org/w/index.php?oldid={rev['revid']}",
                    "revision_api_url": url, "parse_api_url": parse_url,
                    "imageinfo_api_url": info_url, "commons_url": imageinfo["descriptionurl"],
                    "image_url": image_url, "original_scan_sha1": work["sha1"],
                    "original_scan_url": imageinfo["url"],
                    "image": image_path.relative_to(output).as_posix(),
                    "image_sha256": digest(image_path), "width": width, "height": height,
                    "reference_file": (directory / "reference.txt").relative_to(output).as_posix(),
                    "reference_sha256": digest(directory / "reference.txt"),
                    "scan_license": "public-domain", "transcription_license": "CC-BY-SA-4.0",
                    "transcription_attribution": "Polish Wikisource contributors; source revision and history linked",
                    "source_quality": 4, "human_reviewed_by_project": False,
                    "eligible_for_training": False, "eligible_for_evaluation": False})
                print("SOURCE_READY", identifier, width, height, flush=True)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                rejected.append({"id": identifier, "reason": str(exc), "type": type(exc).__name__})
                print("SOURCE_REJECTED", identifier, str(exc), flush=True)
            time.sleep(0.3)
    write_rows(output / "manifest.jsonl", rows)
    write_rows(output / "rejected.jsonl", rejected)
    report = {"schema": "slayer-printed-replay-sources-v1", "collected_at": datetime.now(timezone.utc).isoformat(),
              "pages_requested": sum(len(w["pages"]) for w in works), "pages": len(rows),
              "work_families": len({r["work_family"] for r in rows}),
              "splits": dict(Counter(r["split"] for r in rows)),
              "rejected": len(rejected), "ready_for_line_alignment": bool(rows) and
              {r["split"] for r in rows} == {"replay-candidate", "replay-probe"},
              "eligible_for_training": False, "eligible_for_evaluation": False}
    write_json(output / "report.json", report)
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    return report


def verify_source_package(root, *, works=None):
    root = Path(root)
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "checksums.json"}
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Symlink in source package")
    if set(checksums) != actual or any(digest(root / name) != sha for name, sha in checksums.items()):
        raise ValueError("Source package checksum/coverage mismatch")
    rows = read_rows(root / "manifest.jsonl")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source page")
    if len({r["image_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source page image")
    groups = {}
    works = WORKS if works is None else works
    validate_works(works)
    identities = {w["id"]: w for w in works}
    for row in rows:
        work = identities.get(row["work_family"])
        if work is None or row["split"] != work["split"] or row["scan_page"] not in work["pages"]:
            raise ValueError("Unexpected source identity/split")
        if row["original_scan_sha1"] != work["sha1"] or row["source_quality"] != 4:
            raise ValueError("Source scan/validation mismatch")
        safe_relative(row["image"])
        safe_relative(row["reference_file"])
        if row["work_family"] in PROTECTED_WORKS or row["eligible_for_training"] or row["eligible_for_evaluation"]:
            raise ValueError("Protected source/invalid eligibility")
        if groups.setdefault(row["work_family"], row["split"]) != row["split"]:
            raise ValueError("Work family crosses source splits")
        if digest(root / row["image"]) != row["image_sha256"] or digest(root / row["reference_file"]) != row["reference_sha256"]:
            raise ValueError("Manifest payload mismatch")
    return rows


def package(root, *, works=None):
    root = Path(root)
    rows = verify_source_package(root, works=works)
    notice = ("# Printed replay source pilot V1\n\n"
        "12 physical-book scan pages; 8 candidate pages from Dzieje grzechu (1928), "
        "4 probe pages from Lalka (1890). These are two separate work families.\n\n"
        "Scans: public domain, Wikimedia Commons / CBN Polona. "
        "Transcription: Polish Wikisource contributors, CC BY-SA 4.0; "
        "retain attribution, revision links, license and share-alike for derivatives.\n"
        "https://creativecommons.org/licenses/by-sa/4.0/\n\n"
        "Each manifest record links the exact page revision. Contributor histories: "
        "open source_url, then page history. Revision/API and image-license snapshots "
        "are preserved in sources/. Rendered reference text preserves spelling; "
        "HTML presentation markup was removed. No spelling modernization.\n\n"
        "Preview JPEGs are not native scan inputs. The miner downloads original "
        "DjVu files and checks pinned SHA1 before native-resolution decoding. "
        "No claim of new human crop review, independent benchmark gold, "
        "training eligibility or SOTA.\n\n"
        "Original training/development/test exposure of upstream models is unknown. "
        "The probe is not a certified untouched test.\n")
    if works is not None:
        counts = Counter(r["split"] for r in rows)
        notice = notice.replace("# Printed replay source pilot V1", "# Printed replay source expansion V1").replace(
            "12 physical-book scan pages; 8 candidate pages from Dzieje grzechu (1928), "
            "4 probe pages from Lalka (1890). These are two separate work families.",
            f"{len(rows)} physical-book scan pages from {len({r['work_family'] for r in rows})} work families; "
            f"{counts['replay-candidate']} candidate pages and {counts['replay-probe']} probe pages. "
            "Work-level splits and rejected pages are recorded in the policy and manifests.")
    (root / "NOTICE.md").write_text(notice, encoding="utf-8")
    write_json(root / "checksums.json", {p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    verify_source_package(root, works=works)
    target = root.parent / (root.name + ".zip")
    if target.exists():
        raise FileExistsError(target)
    shutil.make_archive(str(target.with_suffix("")), "zip", root)
    return {"archive": str(target), "sha256": digest(target), "pages": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--works-config", help="Explicit, reviewed work identities and page selections")
    args = parser.parse_args()
    works = json.loads(Path(args.works_config).read_text(encoding="utf-8"))["works"] if args.works_config else None
    print(json.dumps(package(args.output, works=works) if args.package else collect(args.output, works=works), indent=2))
