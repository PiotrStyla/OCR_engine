(() => {
  "use strict";

  const DATA = window.REVIEW_DATA;
  const clone = value => JSON.parse(JSON.stringify(value));
  const esc = value => String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const storageKey = `slayer-layout-gt-review:${DATA.source_audit_sha256}:${DATA.policy_version}`;
  const seededPages = DATA.pages.map(page => ({
    page_id: page.page_id,
    status: "pending",
    note: "",
    dirty: false,
    annotations: clone(page.original_annotations),
  }));
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_error) { saved = {}; }
  const validSaved = Array.isArray(saved.pages)
    && saved.pages.length === DATA.pages.length
    && saved.pages.every((page, index) => page.page_id === DATA.pages[index].page_id);
  const state = {
    pageIndex: 0,
    pages: validSaved ? saved.pages : seededPages,
    reviewer: typeof saved.reviewer === "string" ? saved.reviewer : "",
    selectedId: null,
    mode: "select",
    drawClass: "text_region",
    showOriginal: false,
    showPredictions: true,
    predictionThreshold: DATA.default_prediction_threshold,
    zoom: 100,
    history: [],
    future: [],
    drag: null,
    manualCounter: 0,
  };

  const root = document.getElementById("app");
  root.innerHTML = `
    <header class="topbar">
      <div class="brand"><strong>SLAYER layout GT review</strong><span id="summary"></span></div>
      <div class="top-actions">
        <input id="reviewer" class="reviewer" maxlength="100" placeholder="Reviewer">
        <button id="export-draft" class="button secondary">Export draft</button>
        <button id="export-final" class="button primary">Export final</button>
      </div>
    </header>
    <div class="workspace">
      <aside class="sidebar">
        <div class="sidebar-title">Development pages</div>
        <nav id="page-list" class="page-list"></nav>
        <details class="policy" open>
          <summary>Annotation policy v1</summary>
          <ol id="policy-list"></ol>
        </details>
      </aside>
      <main class="main-area">
        <div class="toolbar">
          <div class="segmented" aria-label="Editor mode">
            <button id="mode-select" class="segment active">Select</button>
            <button id="mode-draw" class="segment">Draw</button>
          </div>
          <select id="draw-class" aria-label="Draw class"></select>
          <button id="undo" class="icon-button" title="Undo">Undo</button>
          <button id="redo" class="icon-button" title="Redo">Redo</button>
          <label class="toggle"><input id="show-original" type="checkbox"> Original GT</label>
          <label class="toggle"><input id="show-predictions" type="checkbox" checked> RF-DETR</label>
          <label class="range-label">Score <input id="threshold" type="range" min="0.05" max="0.90" step="0.05"><span id="threshold-value"></span></label>
          <label class="range-label">Zoom <input id="zoom" type="range" min="50" max="200" step="10" value="100"><span id="zoom-value">100%</span></label>
        </div>
        <div id="viewer" class="viewer"><svg id="canvas" role="img"></svg></div>
      </main>
      <aside class="inspector">
        <section class="page-review">
          <h2 id="page-title"></h2>
          <div id="page-audit" class="audit-line"></div>
          <div class="status-control" id="status-control"></div>
          <label class="field-label" for="page-note">Page note</label>
          <textarea id="page-note" rows="3" maxlength="2000"></textarea>
        </section>
        <section id="selection" class="selection"></section>
        <section class="objects">
          <div class="objects-header"><h3>Current annotations</h3><span id="object-count"></span></div>
          <div id="object-list" class="object-list"></div>
        </section>
      </aside>
    </div>`;

  const elements = Object.fromEntries([
    "summary", "reviewer", "export-draft", "export-final", "page-list", "policy-list",
    "mode-select", "mode-draw", "draw-class", "undo", "redo", "show-original",
    "show-predictions", "threshold", "threshold-value", "zoom", "zoom-value",
    "viewer", "canvas", "page-title", "page-audit", "status-control", "page-note",
    "selection", "object-count", "object-list",
  ].map(id => [id, document.getElementById(id)]));

  DATA.policy.forEach(rule => {
    const item = document.createElement("li");
    item.textContent = rule;
    elements["policy-list"].appendChild(item);
  });
  elements["draw-class"].innerHTML = DATA.categories.map(
    name => `<option value="${esc(name)}">${esc(name)}</option>`).join("");

  function currentSource() { return DATA.pages[state.pageIndex]; }
  function currentReview() { return state.pages[state.pageIndex]; }
  function selected() {
    return currentReview().annotations.find(item => item.id === state.selectedId) || null;
  }
  function save() {
    localStorage.setItem(storageKey, JSON.stringify({
      reviewer: state.reviewer,
      pages: state.pages,
    }));
  }
  function snapshot() {
    state.history.push(clone(state.pages));
    if (state.history.length > 60) state.history.shift();
    state.future = [];
  }
  function restore(target, source) {
    target.length = 0;
    target.push(...clone(source));
    state.selectedId = null;
    save();
    render();
  }
  function markEdited() {
    const page = currentReview();
    page.dirty = true;
    if (page.status !== "needs-review") page.status = "edited";
  }
  function setMode(mode) {
    state.mode = mode;
    elements["mode-select"].classList.toggle("active", mode === "select");
    elements["mode-draw"].classList.toggle("active", mode === "draw");
    elements.canvas.classList.toggle("draw-mode", mode === "draw");
  }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function normalizeBox(box, width, height) {
    let [x1, y1, x2, y2] = box.map(Number);
    [x1, x2] = [Math.min(x1, x2), Math.max(x1, x2)];
    [y1, y2] = [Math.min(y1, y2), Math.max(y1, y2)];
    return [
      clamp(x1, 0, width), clamp(y1, 0, height),
      clamp(x2, 0, width), clamp(y2, 0, height),
    ];
  }
  function point(event) {
    const svgPoint = elements.canvas.createSVGPoint();
    svgPoint.x = event.clientX;
    svgPoint.y = event.clientY;
    return svgPoint.matrixTransform(elements.canvas.getScreenCTM().inverse());
  }
  function pageComplete(page) { return page.status === "verified" || page.status === "edited"; }

  function render() {
    const complete = state.pages.filter(pageComplete).length;
    const unresolved = state.pages.filter(page => page.status === "needs-review").length;
    elements.summary.textContent = `${complete}/${state.pages.length} complete · ${unresolved} unresolved`;
    elements.reviewer.value = state.reviewer;
    elements["export-final"].disabled = complete !== state.pages.length || !state.reviewer.trim();
    elements["export-draft"].disabled = !state.reviewer.trim();
    elements.undo.disabled = state.history.length === 0;
    elements.redo.disabled = state.future.length === 0;
    elements["page-list"].innerHTML = DATA.pages.map((page, index) => {
      const review = state.pages[index];
      return `<button class="page-button ${index === state.pageIndex ? "active" : ""}" data-page="${index}">
        <span>${esc(page.page_id)}</span><span class="page-meta">${page.error_count} errors · ${esc(review.status)}</span>
      </button>`;
    }).join("");
    elements["page-list"].querySelectorAll("[data-page]").forEach(button => {
      button.onclick = () => {
        state.pageIndex = Number(button.dataset.page);
        state.selectedId = null;
        render();
      };
    });
    renderPageReview();
    renderCanvas();
    renderInspector();
  }

  function renderPageReview() {
    const source = currentSource();
    const review = currentReview();
    const totals = source.audit.totals;
    elements["page-title"].textContent = source.page_id;
    elements["page-audit"].textContent = `TP ${totals.tp || 0} · FP ${totals.fp || 0} · FN ${totals.fn || 0}`;
    elements["status-control"].innerHTML = [
      ["pending", "Pending"], ["verified", "Verified"],
      ["edited", "Edited"], ["needs-review", "Needs review"],
    ].map(([value, label]) => `<button data-status="${value}" class="status-button ${review.status === value ? "active" : ""}">${label}</button>`).join("");
    elements["status-control"].querySelectorAll("[data-status]").forEach(button => {
      button.onclick = () => {
        snapshot();
        review.status = button.dataset.status;
        save();
        render();
      };
    });
    elements["page-note"].value = review.note || "";
  }

  function shape(box, color, className, label, id = "") {
    const [x1, y1, x2, y2] = box;
    return `<g><rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" class="${className}" stroke="${color}" ${id ? `data-box-id="${esc(id)}"` : ""}/>
      <text x="${x1 + 6}" y="${Math.max(30, y1 + 30)}" class="box-label" fill="${color}">${esc(label)}</text></g>`;
  }
  function handles(item) {
    const [x1, y1, x2, y2] = item.bbox_xyxy;
    const positions = {
      nw: [x1, y1], n: [(x1 + x2) / 2, y1], ne: [x2, y1],
      e: [x2, (y1 + y2) / 2], se: [x2, y2], s: [(x1 + x2) / 2, y2],
      sw: [x1, y2], w: [x1, (y1 + y2) / 2],
    };
    return Object.entries(positions).map(([name, value]) =>
      `<circle cx="${value[0]}" cy="${value[1]}" r="18" class="resize-handle" data-handle="${name}" data-box-id="${esc(item.id)}"/>`).join("");
  }
  function renderCanvas() {
    const source = currentSource();
    const review = currentReview();
    const parts = [`<image href="${esc(source.image_src)}" width="${source.width}" height="${source.height}"/>`];
    if (state.showOriginal) {
      source.original_annotations.forEach(item => parts.push(shape(
        item.bbox_xyxy, "#697386", "original-box", `original · ${item.label}`)));
    }
    if (state.showPredictions) {
      source.predictions.filter(item => item.score >= state.predictionThreshold).forEach(item => parts.push(shape(
        item.bbox_xyxy, DATA.colors[item.label] || "#555", "prediction-box",
        `RF ${item.label} ${item.score.toFixed(2)}`)));
    }
    review.annotations.forEach(item => {
      const selectedClass = item.id === state.selectedId ? " current-box selected-box" : " current-box";
      parts.push(shape(item.bbox_xyxy, DATA.colors[item.label], selectedClass, item.label, item.id));
    });
    const item = selected();
    if (item) parts.push(handles(item));
    if (state.drag && state.drag.type === "draw" && state.drag.preview) {
      parts.push(shape(state.drag.preview, DATA.colors[state.drawClass], "draft-box", state.drawClass));
    }
    elements.canvas.setAttribute("viewBox", `0 0 ${source.width} ${source.height}`);
    elements.canvas.setAttribute("aria-label", source.page_id);
    elements.canvas.style.width = `${state.zoom}%`;
    elements.canvas.innerHTML = parts.join("");
  }

  function renderInspector() {
    const review = currentReview();
    const item = selected();
    elements["object-count"].textContent = String(review.annotations.length);
    elements["object-list"].innerHTML = review.annotations.map(annotation =>
      `<button data-object-id="${esc(annotation.id)}" class="object-row ${annotation.id === state.selectedId ? "active" : ""}">
        <span class="object-swatch" style="background:${DATA.colors[annotation.label]}"></span>
        <span>${esc(annotation.label)}</span><code>${annotation.bbox_xyxy.map(value => Math.round(value)).join(", ")}</code>
      </button>`).join("");
    elements["object-list"].querySelectorAll("[data-object-id]").forEach(button => {
      button.onclick = () => { state.selectedId = button.dataset.objectId; setMode("select"); render(); };
    });
    if (!item) {
      elements.selection.innerHTML = `<div class="empty-selection">No annotation selected</div>`;
      return;
    }
    elements.selection.innerHTML = `
      <div class="selection-header"><h3>Selected annotation</h3><div><button id="duplicate-object" class="button compact">Duplicate</button><button id="delete-object" class="button danger compact">Delete</button></div></div>
      <label class="field-label">Class</label>
      <select id="selected-class">${DATA.categories.map(name => `<option value="${esc(name)}" ${name === item.label ? "selected" : ""}>${esc(name)}</option>`).join("")}</select>
      <div class="coords">${item.bbox_xyxy.map((value, index) => `<label>${["x1", "y1", "x2", "y2"][index]}<input data-coordinate="${index}" type="number" step="1" value="${Math.round(value)}"></label>`).join("")}</div>
      <div class="origin">${esc(item.origin || "manual-review")}</div>`;
    document.getElementById("selected-class").onchange = event => {
      snapshot();
      item.label = event.target.value;
      markEdited();
      save();
      render();
    };
    elements.selection.querySelectorAll("[data-coordinate]").forEach(input => {
      input.onchange = () => {
        snapshot();
        item.bbox_xyxy[Number(input.dataset.coordinate)] = Number(input.value);
        item.bbox_xyxy = normalizeBox(item.bbox_xyxy, currentSource().width, currentSource().height);
        markEdited();
        save();
        render();
      };
    });
    document.getElementById("delete-object").onclick = deleteSelected;
    document.getElementById("duplicate-object").onclick = () => {
      snapshot();
      const copy = clone(item);
      copy.id = `manual:${Date.now()}:${++state.manualCounter}`;
      copy.source_annotation_id = null;
      copy.origin = "human-review";
      copy.bbox_xyxy = normalizeBox(copy.bbox_xyxy.map(value => value + 12), currentSource().width, currentSource().height);
      review.annotations.push(copy);
      state.selectedId = copy.id;
      markEdited();
      save();
      render();
    };
  }

  function deleteSelected() {
    if (!selected()) return;
    snapshot();
    const review = currentReview();
    review.annotations = review.annotations.filter(item => item.id !== state.selectedId);
    state.selectedId = null;
    markEdited();
    save();
    render();
  }

  function startPointer(event) {
    const position = point(event);
    const source = currentSource();
    if (state.mode === "draw") {
      snapshot();
      state.drag = {type: "draw", start: [position.x, position.y], preview: [position.x, position.y, position.x, position.y]};
      elements.canvas.setPointerCapture(event.pointerId);
      renderCanvas();
      return;
    }
    const id = event.target.dataset.boxId;
    const handle = event.target.dataset.handle;
    if (!id) { state.selectedId = null; render(); return; }
    state.selectedId = id;
    const item = selected();
    if (!item) return;
    snapshot();
    state.drag = {type: handle ? "resize" : "move", handle, start: [position.x, position.y], original: clone(item.bbox_xyxy)};
    elements.canvas.setPointerCapture(event.pointerId);
    render();
  }
  function movePointer(event) {
    if (!state.drag) return;
    const position = point(event);
    const source = currentSource();
    if (state.drag.type === "draw") {
      state.drag.preview = normalizeBox([
        state.drag.start[0], state.drag.start[1], position.x, position.y,
      ], source.width, source.height);
      renderCanvas();
      return;
    }
    const item = selected();
    if (!item) return;
    const dx = position.x - state.drag.start[0];
    const dy = position.y - state.drag.start[1];
    let box = clone(state.drag.original);
    if (state.drag.type === "move") {
      const width = box[2] - box[0], height = box[3] - box[1];
      const x1 = clamp(box[0] + dx, 0, source.width - width);
      const y1 = clamp(box[1] + dy, 0, source.height - height);
      box = [x1, y1, x1 + width, y1 + height];
    } else {
      const handle = state.drag.handle;
      if (handle.includes("w")) box[0] += dx;
      if (handle.includes("e")) box[2] += dx;
      if (handle.includes("n")) box[1] += dy;
      if (handle.includes("s")) box[3] += dy;
      box = normalizeBox(box, source.width, source.height);
    }
    item.bbox_xyxy = box;
    renderCanvas();
  }
  function endPointer(event) {
    if (!state.drag) return;
    const drag = state.drag;
    state.drag = null;
    if (elements.canvas.hasPointerCapture(event.pointerId)) elements.canvas.releasePointerCapture(event.pointerId);
    if (drag.type === "draw" && drag.preview) {
      const [x1, y1, x2, y2] = drag.preview;
      if (x2 - x1 >= 12 && y2 - y1 >= 12) {
        const item = {
          id: `manual:${Date.now()}:${++state.manualCounter}`,
          source_annotation_id: null,
          label: state.drawClass,
          bbox_xyxy: drag.preview,
          origin: "human-review",
        };
        currentReview().annotations.push(item);
        state.selectedId = item.id;
        markEdited();
      }
    } else {
      markEdited();
    }
    save();
    render();
  }

  function exportReview(final) {
    if (!state.reviewer.trim()) return;
    if (final && !state.pages.every(pageComplete)) return;
    const packet = {
      schema: "slayer-layout-gt-review-v1",
      state: final ? "complete" : "draft",
      source_audit_sha256: DATA.source_audit_sha256,
      source_audit_report_sha256: DATA.source_audit_report_sha256,
      source_dataset_archive_sha256: DATA.source_dataset_archive_sha256,
      source_coco_sha256: DATA.source_coco_sha256,
      checkpoint_sha256: DATA.checkpoint_sha256,
      policy_version: DATA.policy_version,
      reviewer: state.reviewer.trim(),
      timestamp: new Date().toISOString(),
      pages: state.pages.map(page => ({
        page_id: page.page_id,
        status: page.status,
        note: page.note || "",
        changed: !!page.dirty,
        annotations: page.annotations.map(item => ({
          id: item.id,
          source_annotation_id: item.source_annotation_id ?? null,
          label: item.label,
          bbox_xyxy: item.bbox_xyxy.map(value => Number(value.toFixed(3))),
          origin: item.origin || "human-review",
        })),
      })),
      release_status: "private-review-not-published",
    };
    const blob = new Blob([JSON.stringify(packet, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = final ? "slayer-layout-gt-review-v1.json" : "slayer-layout-gt-review-draft-v1.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  elements.reviewer.onchange = event => { state.reviewer = event.target.value.trim(); save(); render(); };
  elements["export-draft"].onclick = () => exportReview(false);
  elements["export-final"].onclick = () => exportReview(true);
  elements["mode-select"].onclick = () => setMode("select");
  elements["mode-draw"].onclick = () => setMode("draw");
  elements["draw-class"].onchange = event => { state.drawClass = event.target.value; setMode("draw"); };
  elements.undo.onclick = () => {
    if (!state.history.length) return;
    state.future.push(clone(state.pages));
    restore(state.pages, state.history.pop());
  };
  elements.redo.onclick = () => {
    if (!state.future.length) return;
    state.history.push(clone(state.pages));
    restore(state.pages, state.future.pop());
  };
  elements["show-original"].onchange = event => { state.showOriginal = event.target.checked; renderCanvas(); };
  elements["show-predictions"].onchange = event => { state.showPredictions = event.target.checked; renderCanvas(); };
  elements.threshold.value = state.predictionThreshold;
  elements.threshold.oninput = event => {
    state.predictionThreshold = Number(event.target.value);
    elements["threshold-value"].textContent = state.predictionThreshold.toFixed(2);
    renderCanvas();
  };
  elements.zoom.oninput = event => {
    state.zoom = Number(event.target.value);
    elements["zoom-value"].textContent = `${state.zoom}%`;
    renderCanvas();
  };
  elements["page-note"].onchange = event => {
    snapshot();
    currentReview().note = event.target.value;
    save();
    render();
  };
  elements.canvas.addEventListener("pointerdown", startPointer);
  elements.canvas.addEventListener("pointermove", movePointer);
  elements.canvas.addEventListener("pointerup", endPointer);
  elements.canvas.addEventListener("pointercancel", endPointer);
  document.addEventListener("keydown", event => {
    if ((event.key === "Delete" || event.key === "Backspace")
        && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      event.preventDefault();
      deleteSelected();
    }
    if (event.key === "Escape") { state.selectedId = null; setMode("select"); render(); }
  });

  elements["threshold-value"].textContent = state.predictionThreshold.toFixed(2);
  setMode("select");
  render();
})();
