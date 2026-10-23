"""Durable epoch results; the measured DEV baseline wins ties and regressions."""
import json
import math
import os
from pathlib import Path


class EpochLedger:
    def __init__(self, output, metadata):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "dev-metrics.json"
        if self.path.exists():
            raise FileExistsError(f"Existing epoch results must be preserved: {self.path}")
        reserved = {"epochs", "selected", "failure"}
        if reserved.intersection(metadata):
            raise ValueError("Metadata collides with epoch ledger fields")
        self.metadata = dict(metadata)
        self.epochs = []
        self.selected = None
        self.failure = None

    def _persist(self):
        payload = dict(self.metadata, epochs=self.epochs, selected=self.selected, failure=self.failure)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def record(self, epoch, metrics, adapter=None):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Invalid epoch")
        if any(row["epoch"] == epoch for row in self.epochs):
            raise ValueError(f"Duplicate epoch: {epoch}")
        if epoch == 0 and adapter is not None:
            raise ValueError("Baseline epoch cannot have an adapter")
        if epoch > 0 and not adapter:
            raise ValueError("Trained epoch requires a saved adapter")
        values = {}
        for key in ("cer_micro", "wer_micro"):
            value = metrics[key]
            if value is None or not math.isfinite(value):
                values[key] = None
            elif value < 0:
                raise ValueError(f"Negative metric: {key}")
            else:
                values[key] = float(value)
        for key in ("pages", "errors_or_missing"):
            value = metrics[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid metric: {key}")
            values[key] = value
        if values["errors_or_missing"] > values["pages"]:
            raise ValueError("Errors exceed the evaluated page count")
        row = {"epoch": epoch, "adapter": str(adapter) if adapter else None, "metrics": values}
        self.epochs.append(row)
        eligible = (values["pages"] > 0 and values["errors_or_missing"] == 0
                    and values["cer_micro"] is not None and values["wer_micro"] is not None)
        if eligible and (self.selected is None or values["cer_micro"] < self.selected["cer_micro"]):
            self.selected = {"epoch": epoch, "adapter": row["adapter"],
                             "cer_micro": values["cer_micro"], "wer_micro": values["wer_micro"]}
        self._persist()

    def mark_failure(self, phase, error):
        self.failure = {"phase": phase, "error": str(error)}
        self._persist()
