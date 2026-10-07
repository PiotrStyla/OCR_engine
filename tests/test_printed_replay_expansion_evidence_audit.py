import copy
import hashlib
import inspect
import json

import pytest

from training import audit_printed_replay_expansion_evidence as expansion
from training.audit_printed_replay_v2_evidence import audit, visual_sample, V2_REVISION


def test_sample_covers_pages_and_families_without_mutation():
    rows = [
        {"id": "a", "page_id": "p1", "work_family": "w1", "text": "long text", "min_word_confidence": 99},
        {"id": "b", "page_id": "p1", "work_family": "w1", "text": "short", "min_word_confidence": 90},
        {"id": "c", "page_id": "p2", "work_family": "w2", "text": "text", "min_word_confidence": 91},
        {"id": "d", "page_id": "p2", "work_family": "w2", "text": "txt", "min_word_confidence": 98},
    ]
    original = copy.deepcopy(rows)
    assert [r["id"] for r in visual_sample(rows)] == ["a", "b", "c"]
    assert {r["id"] for r in visual_sample(rows[::-1])} == {"a", "b", "c"}
    assert rows == original
    assert visual_sample([]) == []


def test_legacy_audit_still_pins_v2_and_reviews_all():
    parameters = inspect.signature(audit).parameters
    assert parameters["code_revision"].default == V2_REVISION
    assert parameters["sample_review"].default is False
    assert parameters["works"].default is None


def test_expansion_distrusts_changed_pinned_config(monkeypatch, tmp_path):
    monkeypatch.setattr(expansion.subprocess, "check_output", lambda command: b'{"works": []}')
    with pytest.raises(ValueError, match="selection mismatch"):
        expansion.expansion_audit("unread.zip", tmp_path)


def test_expansion_passes_trusted_pins(monkeypatch, tmp_path):
    payload = json.dumps({"works": [{"id": "trusted"}]}).encode()
    monkeypatch.setattr(expansion, "WORKS_SHA256", hashlib.sha256(payload).hexdigest())
    commands = []
    monkeypatch.setattr(expansion.subprocess, "check_output", lambda cmd: commands.append(cmd) or payload)
    calls = []
    monkeypatch.setattr(expansion, "audit", lambda *args, **kwargs: calls.append(kwargs) or {"ok": True})
    monkeypatch.setattr(expansion, "add_geometry_diagnostic", lambda output: None)
    assert expansion.expansion_audit("input.zip", tmp_path) == {"ok": True}
    assert commands[0] == ["git", "show", expansion.REVISION + ":" + expansion.WORKS_PATH]
    assert calls[0]["code_revision"] == expansion.REVISION
    assert calls[0]["source_checksums"] == expansion.SOURCE_CHECKSUMS
    assert calls[0]["works"] == [{"id": "trusted"}]
    assert calls[0]["sample_review"] is True


def row(index, top, bottom, span):
    return {"page_id": "p", "line_index": index, "bbox": [10, top, 100, bottom],
            "source_span": span, "reason": "anchor-order-or-line-overlap"}


def test_geometry_and_text_order_separated():
    result = expansion.diagnose_conflicts([row(1, 10, 25, [0, 20]), row(2, 23, 40, [21, 40]),
                                          row(3, 50, 60, [5, 18])])
    assert result["rejected_exact_anchors"] == 3
    assert result["bbox_extent_conflicts"] == result["source_order_conflicts"] == 1
    assert result["pages"][0]["conflicts"][0]["vertical_extent_overlap_px"] == 2
    assert result["rescued_pairs"] == 0 and result["training_ready"] is False


def test_invalid_gate_rejection_rejected():
    with pytest.raises(ValueError, match="without a reproducible conflict"):
        expansion.diagnose_conflicts([row(1, 10, 20, [0, 10]), row(2, 21, 30, [11, 20])])


def test_unrelated_rejections_ignored():
    assert expansion.diagnose_conflicts([{"reason": "word-confidence-below-90"}])["pages"] == []
