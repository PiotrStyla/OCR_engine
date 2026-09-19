'use strict';
const source = JSON.parse(document.getElementById('source-data').textContent);
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
  return {schema: 'polocrbench-review-patch-v1', manifest_sha256: source.manifest_sha256, events: history};
}
function validate(packet) {
  if (packet?.schema !== 'polocrbench-review-patch-v1' || packet.manifest_sha256 !== source.manifest_sha256 || !Array.isArray(packet.events)) {
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
function currentText(id) { return latest(id)?.after ?? byId.get(id).text; }
function visiblePages() {
  const query = $('search').value.toLowerCase(), filter = $('filter').value;
  return source.pages.filter(page => page.id.toLowerCase().includes(query) &&
    (filter === 'all' || (filter === 'flagged' && page.issues.length) ||
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
    detail.textContent = `${page.issues.length} znaków · ${latest(page.id) ? labels[latest(page.id).decision] : 'Bez decyzji'}`;
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
    const title = document.createElement('strong'); title.textContent = `${labels[event.decision]} · ${event.reviewer}`;
    const note = document.createElement('p'); note.textContent = `${new Date(event.timestamp).toLocaleString('pl-PL')} · ${event.note || 'Bez uwag'}`;
    const details = document.createElement('details'), summary = document.createElement('summary'); summary.textContent = 'Tekst przed i po';
    const before = document.createElement('pre'), after = document.createElement('pre'); before.textContent = event.before; after.textContent = event.after;
    details.append(summary, before, after); li.append(title, note, details); $('history-list').append(li);
  }
}
function updateDirty() {
  const last = latest(active);
  dirty = $('text').value !== currentText(active) || $('note').value !== (last?.note ?? '') || $('decision').value !== (last?.decision ?? 'needs-review');
  $('dirty').textContent = dirty ? 'Niezapisane zmiany' : '';
}
function render() {
  const page = byId.get(active), last = latest(active);
  $('page-id').textContent = page.id;
  $('page-state').textContent = last ? `${labels[last.decision]} · ${last.reviewer}` : `${page.issues.length} oznaczeń w źródle · Bez decyzji`;
  $('scan').src = page.image; $('scan').alt = `Skan ${page.id}`;
  $('scan').style.width = '100%'; $('zoom').value = '100';
  $('original-text').textContent = page.text; $('text').value = currentText(active);
  $('decision').value = last?.decision ?? 'needs-review'; $('note').value = last?.note ?? '';
  dirty = false; $('dirty').textContent = ''; queue(); renderIssues(); history();
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
$('zoom').onchange = () => $('scan').style.width = `${$('zoom').value}%`;
$('scan').onerror = () => notify('Nie udało się wczytać skanu. Sprawdź katalog images.', true);
$('text').oninput = () => { updateDirty(); renderIssues(); };
$('note').oninput = updateDirty; $('decision').onchange = updateDirty;
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
  try { validate(envelope([...events, entry])); }
  catch (error) { notify(error.message, true); return; }
  events.push(entry); persist(); render();
  if (!storageBlocked) notify('Decyzja zapisana lokalnie.');
};
$('export').onclick = () => {
  const blob = new Blob([JSON.stringify(envelope(), null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `polocrbench-review-${source.manifest_sha256.slice(0, 12)}.json`;
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
