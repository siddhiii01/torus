"""Single-page review UI (P9.1), served as one self-contained HTML string.

Document text is only ever inserted with ``textContent`` (never ``innerHTML``),
so OCR output cannot inject markup. Keyboard first:

    j / k      next / previous item        Enter   accept the suggestion
    1-9        take reader N's reading     e       edit the value
    Enter (in the edit box) save correction   Esc    leave the edit box
    s          skip                        p       toggle page-in-context
"""

from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Torus review</title>
<style>
  body { font: 15px system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
  #queue { width: 260px; overflow-y: auto; border-right: 1px solid #ccc; }
  #queue div { padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #eee; }
  #queue div.cur { background: #def; }
  #queue div.done { opacity: .45; }
  #queue small { color: #666; }
  #main { flex: 1; overflow-y: auto; padding: 12px 18px; }
  img { max-width: 100%; border: 1px solid #aaa; display: block; margin: 6px 0; }
  .bad { color: #b00; } .ok { color: #070; }
  ol { padding-left: 22px; }
  input { font: inherit; width: 24em; padding: 4px; }
  kbd { background: #eee; border: 1px solid #bbb; border-radius: 3px; padding: 0 4px; }
  #help { color: #555; margin-top: 14px; }
</style>
</head>
<body>
<div id="queue"></div>
<div id="main">
  <h2 id="title">Loading…</h2>
  <div id="meta"></div>
  <img id="crop" alt="field crop">
  <div>Suggestion: <b id="sugg"></b></div>
  <h4>Readings</h4><ol id="readings"></ol>
  <h4>Checks</h4><ul id="checks"></ul>
  <div>Value: <input id="edit" autocomplete="off" spellcheck="false"></div>
  <p id="msg"></p>
  <img id="page" alt="page in context" hidden>
  <div id="help">
    <kbd>j</kbd>/<kbd>k</kbd> move · <kbd>Enter</kbd> accept ·
    <kbd>1</kbd>-<kbd>9</kbd> take a reading · <kbd>e</kbd> edit ·
    <kbd>s</kbd> skip · <kbd>p</kbd> page view
  </div>
</div>
<script>
const TOKEN = "__TOKEN__";
let queue = [], cur = 0, item = null;
const $ = (id) => document.getElementById(id);

function el(tag, text, cls) {
  const e = document.createElement(tag);
  e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

function renderQueue() {
  const q = $("queue");
  q.replaceChildren();
  queue.forEach((it, i) => {
    const d = el("div", it.column_key + " ");
    d.appendChild(el("small", "r" + (it.row + 1) + " · " + it.status +
      (it.decision ? " · " + it.decision : "")));
    if (i === cur) d.classList.add("cur");
    if (it.decision) d.classList.add("done");
    d.onclick = () => show(i);
    q.appendChild(d);
  });
}

async function show(i) {
  if (!queue.length) { $("title").textContent = "Nothing to review"; return; }
  cur = Math.max(0, Math.min(queue.length - 1, i));
  renderQueue();
  const id = encodeURIComponent(queue[cur].item_id);
  item = await api("/api/item/" + id);
  $("title").textContent = item.column_key + " (row " + (item.row + 1) + ")";
  $("meta").textContent = item.status + " · confidence " +
    item.confidence.toFixed(2) + (item.reason ? " · " + item.reason : "");
  $("sugg").textContent = item.suggestion === null ? "(none)" : item.suggestion;
  $("edit").value = item.suggestion || "";
  $("msg").textContent = "";
  const hasEv = item.evidence.length > 0;
  $("crop").hidden = !hasEv;
  $("page").hidden = true;
  if (hasEv) $("crop").src = "/img/crop/" + id + "/0.png";
  const rs = $("readings");
  rs.replaceChildren();
  item.readings.forEach((r, n) => {
    const li = el("li", (n < 9 ? "[" + (n + 1) + "] " : "") + r.reader_name + ": ");
    li.appendChild(el("b", r.error ? "(error: " + r.error + ")" : r.text));
    rs.appendChild(li);
  });
  const cs = $("checks");
  cs.replaceChildren();
  item.validators.forEach((v) => {
    cs.appendChild(el("li", (v.passed ? "pass " : "FAIL ") + v.validator_name +
      (v.message ? ": " + v.message : ""), v.passed ? "ok" : "bad"));
  });
}

async function decide(action, value) {
  try {
    await api("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Torus-Token": TOKEN },
      body: JSON.stringify({ item_id: item.item_id, action, value }),
    });
    queue[cur].decision = action;
    let next = queue.findIndex((q, k) => k > cur && !q.decision);
    if (next < 0) next = queue.findIndex((q) => !q.decision);
    if (next < 0) { renderQueue(); $("msg").textContent = "Queue finished."; return; }
    await show(next);
  } catch (e) {
    $("msg").textContent = e.message;
  }
}

document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey || !item) return;
  if (document.activeElement === $("edit")) {
    if (ev.key === "Enter") { ev.preventDefault(); decide("correct", $("edit").value); }
    else if (ev.key === "Escape") { $("edit").blur(); }
    return;
  }
  if (ev.key === "j") show(cur + 1);
  else if (ev.key === "k") show(cur - 1);
  else if (ev.key === "Enter") decide("accept");
  else if (ev.key === "s") decide("skip");
  else if (ev.key === "e") {
    ev.preventDefault();
    $("edit").focus();
    $("edit").select();
  }
  else if (ev.key === "p" && item.evidence.length) {
    const pg = $("page");
    pg.hidden = !pg.hidden;
    if (!pg.hidden) pg.src = "/img/page/" + encodeURIComponent(item.item_id) + "/0.png";
  } else if (/^[1-9]$/.test(ev.key)) {
    const r = item.readings[Number(ev.key) - 1];
    if (r && !r.error) { $("edit").value = r.text; $("edit").focus(); }
  }
});

api("/api/queue").then((q) => { queue = q; return show(0); })
  .catch((e) => { $("title").textContent = e.message; });
</script>
</body>
</html>
"""
