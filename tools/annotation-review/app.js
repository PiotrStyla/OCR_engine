'use strict';
const source = JSON.parse(document.getElementById('source-data').textContent);
const geometryReview = source.geometry_review_required === true;
const reviewSchema = geometryReview ? 'slayer-recognizer-line-review-v1' : 'polocrbench-review-patch-v1';
const geometryLabels = {'unreviewed': 'Nie sprawdzono', 'complete-line': 'Cała jedna linia', 'reject-crop': 'Odrzuć wycinek'};
const geometryStates = {'unreviewed': 'Granice do sprawdzenia', 'complete-line': 'Cała linia', 'reject-crop': 'Wycinek odrzucony'};
document.body.classList.toggle('line-review', geometryReview);
const byId = new Map(source.pages.map(page => [page.id, page]));
const storageKey = `polocrbench-review-v1:${source.manifest_sha256}`;
const $ = id => document.getElementById(id);
const labels = {'needs-review': 'Do wyjaśnienia', proposed: 'Propozycja korekty', verified: 'Zweryfikowano wzrokowo'};
let events = [], active = source.pages[0].id, dirty = false, timer, storageBlocked = false;

function notify(text, error = false) {
  clearTimeout(timer);
  $('message').textContent = text;
  $('message').className = `visible${error ? ' error' : ''}`;
  if (!error) timer = setTimeout(() => $('message').className = '', 5500);
}
function envelope(history = events) {
  return {schema: reviewSchema, manifest_sha256: source.manifest_sha256, events: history};
}
function validate(packet) {
  if (packet?.schema !== reviewSchema || packet.manifest_sha256 !== source.manifest_sha256 || !Array.isArray(packet.events)) {
    throw new Error('Plik nie odpowiada temu manifestowi lub formatowi historii.');
  }
  const texts = new Map(source.pages.map(page => [page.id, page.text]));
  const ids = new Set();
  for (const event of packet.events) {
    const page = byId.get(event.page_id);
    if (!page || typeof event.id !== 'string' || !event.id || ids.has(event.id) ||
        event.original_text_sha256 !== page.text_sha256 || event.image_sha256 !== page.image_sha256 ||
        event.before !== texts.get(event.page_id) || typeof event.after !== 'string' || event.after.length > 200000 ||
        typeof event.reviewer !== 'string' || !event.reviewer.trim() || event.reviewer.length > 100 ||
        typeof event.note !== 'string' || event.note.length > 2000 ||
        !Object.hasOwn(labels, event.decision) || typeof event.timestamp !== 'string' || !Number.isFinite(Date.parse(event.timestamp))) {
      throw new Error('Niespójna historia zmian lub nieprawidłowy rekord.');
    }
    if (geometryReview && (!Object.hasOwn(geometryLabels, event.geometry_decision) ||
        event.context_image_sha256 !== page.context.sha256 ||
        (event.decision === 'verified' && event.geometry_decision === 'unreviewed') ||
        (event.geometry_decision === 'reject-crop' && !event.note.trim()))) {
      throw new Error('Sprawdź granice wycinka; odrzucenie wymaga uzasadnienia.');
    }
    ids.add(event.id);
    texts.set(event.page_id, event.after);
  }
  return packet.events;
}
function persist() {
  if (storageBlocked) { notify('Zapis lokalny niedostępny. Wyeksportuj historię przed zamknięciem.', true); return; }
  try { localStorage.setItem(storageKey, JSON.stringify(envelope())); }
  catch { storageBlocked = true; notify('Brak miejsca na zapis lokalny. Wyeksportuj historię.', true); }
}
function latest(id) { return events.findLast(event => event.page_id === id); }
function decisionLabel(event) { return labels[event.decision] + (geometryReview ? ` · ${geometryStates[event.geometry_decision]}` : ''); }
function currentText(id) { return latest(id)?.after ?? byId.get(id).text; }
function visiblePages() {
  const query = $('search').value.toLowerCase(), filter = $('filter').value;
  return source.pages.filter(page => page.id.toLowerCase().includes(query) &&
    (filter === 'all' || (filter === 'flagged' && (page.issues.length || page.diagnostics?.items.length)) ||
     (filter === 'pending' && !latest(page.id)) || (filter === 'reviewed' && latest(page.id))));
}
function queue() {
  $('queue').replaceChildren();
  const visible = visiblePages();
  for (const page of visible) {
    const button = document.createElement('button'); button.className = 'page-row';
    button.setAttribute('aria-current', String(page.id === active));
    const title = document.createElement('strong'); title.textContent = page.id;
    const detail = document.createElement('small');
    const count = page.diagnostics?.review_status_label ?? (page.diagnostics ? `${page.diagnostics.items.length} różnic OCR` : `${page.issues.length} znaków`);
    detail.textContent = `${count} · ${latest(page.id) ? decisionLabel(latest(page.id)) : 'Bez decyzji'}`;
    button.append(title, detail); button.onclick = () => navigate(page.id); $('queue').append(button);
  }
  if (!visible.length) { const empty = document.createElement('p'); empty.className = 'empty'; empty.textContent = 'Brak stron dla wybranych filtrów.'; $('queue').append(empty); }
  $('progress').textContent = `${new Set(events.map(event => event.page_id)).size} / ${source.pages.length}`;
  const index = visible.findIndex(page => page.id === active);
  $('previous').disabled = index <= 0;
  $('next').disabled = index < 0 || index >= visible.length - 1;
}
function findIssues(text) {
  const result = [];
  let offset = 0;
  for (const character of text) {
    if (character === '\ufffd' || /\p{Co}/u.test(character)) result.push({offset, length: character.length,
      codepoint: `U+${character.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}`,
      context: text.slice(Math.max(0, offset - 22), offset + character.length + 22)});
    offset += character.length;
  }
  return result;
}
function renderIssues() {
  const found = findIssues($('text').value); $('issue-count').textContent = found.length;
  $('issues').replaceChildren();
  for (const issue of found) {
    const button = document.createElement('button'); button.className = 'issue';
    const code = document.createElement('code'); code.textContent = issue.codepoint;
    const context = document.createElement('span'); context.textContent = issue.context.replace(/\s+/g, ' ');
    button.append(code, context); button.title = `Pozycja ${issue.offset + 1}: ${issue.codepoint}`;
    button.onclick = () => { $('text').focus(); $('text').setSelectionRange(issue.offset, issue.offset + issue.length); };
    $('issues').append(button);
  }
  if (!found.length) { const empty = document.createElement('p'); empty.className = 'empty'; empty.textContent = 'Brak znaków U+FFFD i prywatnego zakresu Unicode.'; $('issues').append(empty); }
}
function history() {
  const pageEvents = events.filter(event => event.page_id === active);
  $('history-count').textContent = `(${pageEvents.length})`; $('history-list').replaceChildren();
  for (const event of pageEvents.toReversed()) {
    const li = document.createElement('li');
    const title = document.createElement('strong'); title.textContent = `${decisionLabel(event)} · ${event.reviewer}`;
    const note = document.createElement('p'); note.textContent = `${new Date(event.timestamp).toLocaleString('pl-PL')} · ${event.note || 'Bez uwag'}`;
    const details = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = 'Tekst przed i po';
    const before = document.createElement('pre'), after = document.createElement('pre'); before.textContent = event.before; after.textContent = event.after;
    details.append(summary, before, after); li.append(title, note, details); $('history-list').append(li);
  }
}
function updateDirty() {
  const last = latest(active);
  dirty = $('text').value !== currentText(active) || $('note').value !== (last?.note ?? '') || $('decision').value !== (last?.decision ?? 'needs-review');
  if (geometryReview) dirty ||= $('geometry').value !== (last?.geometry_decision ?? 'unreviewed');
  $('dirty').textContent = dirty ? 'Niezapisane zmiany' : '';
}
function render() {
  const page = byId.get(active), last = latest(active);
  $('page-id').textContent = page.id;
  $('page-state').textContent = last ? `${decisionLabel(last)} · ${last.reviewer}` :
    (geometryReview ? 'Niezweryfikowana etykieta · Bez decyzji' : `${page.issues.length} oznaczeń w źródle · Bez decyzji`);
  $('scan').src = page.image; $('scan').alt = `Skan ${page.id}`;
  $('scan').style.width = '100%'; $('zoom').value = '100';
  $('source-context').hidden = !page.context;
  $('source-context').open = geometryReview;
  if (page.context) {
    $('context-scan').src = page.context.image;
    $('context-scan').style.width = '100%';
    $('context-text').textContent = page.context.text;
  }
  $('geometry-field').hidden = !geometryReview;
  $('geometry').value = last?.geometry_decision ?? 'unreviewed';
  $('original-text').textContent = page.text; $('text').value = currentText(active);
  renderDiagnostics(page);
  $('decision').value = last?.decision ?? 'needs-review'; $('note').value = last?.note ?? '';
  dirty = false; $('dirty').textContent = ''; queue(); renderIssues(); history();
}
function renderDiagnostics(page) {
  const diagnostic = page.diagnostics;
  $('ocr-diagnostics').hidden = !diagnostic;
  $('diagnostic-items').replaceChildren();
  if (!diagnostic) return;
  $('diagnostic-count').textContent = geometryReview ? '' : `(${diagnostic.items.length})`;
  for (const item of diagnostic.items) {
    const button = document.createElement('button'); button.className = 'diagnostic-item';
    const change = document.createElement('strong');
    change.textContent = `${item.reference || '∅'} → ${item.candidate || '∅'}`;
    const context = document.createElement('span'); context.textContent = item.reference_context;
    button.append(change, context);
    button.title = 'Fragment draftu i surowy odczyt OCR';
    button.onclick = () => {
      if ($('text').value !== page.text) {
        notify('Po zmianie tekstu pozycja źródłowa może być nieaktualna. Sprawdź kontekst.', true);
        return;
      }
      $('text').focus(); $('text').setSelectionRange(item.offset, item.offset + item.length);
      $('text').scrollTop = $('text').scrollHeight * item.offset / Math.max(1, page.text.length) - $('text').clientHeight / 2;
    };
    $('diagnostic-items').append(button);
  }
  $('candidate-label').textContent = diagnostic.candidate_label;
  $('candidate-text').textContent = diagnostic.candidate_text;
  $('baseline-label').textContent = diagnostic.baseline_label;
  $('baseline-text').textContent = diagnostic.baseline_text;
  if (geometryReview) {
    $('candidate-label').parentElement.open = true;
    $('baseline-label').parentElement.open = true;
  }
}
function navigate(id) {
  if (id === active) return;
  if (dirty && !confirm('Odrzucić niezapisane zmiany na tej stronie?')) return;
  active = id; render();
}
$('search').oninput = queue; $('filter').onchange = queue;
for (const [id, delta] of [['previous', -1], ['next', 1]]) $(id).onclick = () => {
  const list = visiblePages(), index = list.findIndex(page => page.id === active), page = list[index + delta];
  if (page) navigate(page.id);
};
$('zoom').onchange = () => { $('scan').style.width = `${$('zoom').value}%`; $('context-scan').style.width = `${$('zoom').value}%`; };
$('context-scan').onerror = () => notify('Nie udało się wczytać regionu źródłowego.', true);
$('scan').onerror = () => notify('Nie udało się wczytać skanu. Sprawdź katalog images.', true);
$('text').oninput = () => { updateDirty(); renderIssues(); };
$('note').oninput = updateDirty; $('decision').onchange = updateDirty; $('geometry').onchange = updateDirty;
$('restore').onclick = () => {
  if ($('text').value !== byId.get(active).text && confirm('Przywrócić tekst źródłowy w edytorze? Historia pozostanie zachowana.')) {
    $('text').value = byId.get(active).text; updateDirty(); renderIssues();
  }
};
$('decision-form').onsubmit = event => {
  event.preventDefault();
  if (!$('reviewer').value.trim()) { $('reviewer').focus(); return; }
  const page = byId.get(active);
  const entry = {id: crypto.randomUUID(), page_id: active, timestamp: new Date().toISOString(),
    reviewer: $('reviewer').value.trim(), decision: $('decision').value, note: $('note').value,
    original_text_sha256: page.text_sha256, image_sha256: page.image_sha256,
    before: currentText(active), after: $('text').value};
  if (geometryReview) {
    entry.geometry_decision = $('geometry').value;
    entry.context_image_sha256 = page.context.sha256;
  }
  try { validate(envelope([...events, entry])); }
  catch (error) { notify(error.message, true); return; }
  events.push(entry); persist(); render();
  if (!storageBlocked) notify('Decyzja zapisana lokalnie.');
};
$('export').onclick = () => {
  const blob = new Blob([JSON.stringify(envelope(), null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `${geometryReview ? 'slayer-recognizer-line-review' : 'polocrbench-review'}-${source.manifest_sha256.slice(0, 12)}.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify(dirty ? 'Wyeksportowano zapisane decyzje. Edytor zawiera niezapisane zmiany.' : `Wyeksportowano ${events.length} decyzji.`);
};
$('import').onclick = () => $('import-file').click();
$('import-file').onchange = async () => {
  const file = $('import-file').files[0]; if (!file) return;
  try {
    if (file.size > 20_000_000) throw new Error('Plik przekracza limit 20 MB.');
    if (dirty && !confirm('Import zastąpi niezapisany tekst w edytorze. Kontynuować?')) return;
    const incoming = validate(JSON.parse(await file.text()));
    const common = Math.min(incoming.length, events.length);
    for (let i = 0; i < common; i++) {
      if (JSON.stringify(incoming[i]) !== JSON.stringify(events[i])) throw new Error('Historie są rozbieżne. Import nie nadpisze lokalnych decyzji.');
    }
    if (incoming.length > events.length) { events = incoming; persist(); }
    render(); if (!storageBlocked) notify(`Historia zawiera ${events.length} decyzji.`);
  } catch (error) { notify(error.message || 'Nieprawidłowy plik historii.', true); }
  finally { $('import-file').value = ''; }
};
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
$('source-hash').textContent = source.manifest_sha256;
try { const saved = localStorage.getItem(storageKey); if (saved) events = validate(JSON.parse(saved)); }
catch { storageBlocked = true; notify('Nie można odczytać lokalnej historii. Istniejący zapis nie zostanie nadpisany.', true); }
render();
