"use strict";

const ALL_CAPS = [
  { id: "predict", label: "Predict" },
  { id: "simulate", label: "Simulate" },
  { id: "edge", label: "Edge" },
];
const PANELS = ["dashboard", "fixtures", "outrights", "history", "predict", "simulate", "edge", "bankroll", "settings", "placeholder"];

const state = { engines: [], current: null, activeCap: "predict", view: "engine" };

// ---------- persisted UI preferences (local to this Mac) ----------
const prefs = {
  get(k, fallback) { try { const v = localStorage.getItem("sp-" + k); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(k, v) { try { localStorage.setItem("sp-" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
};

// ---------- theme ----------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("sp-theme", theme); } catch (e) { /* ignore */ }
}
function initTheme() {
  let theme;
  try { theme = localStorage.getItem("sp-theme"); } catch (e) { theme = null; }
  if (!theme) theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(theme);
  const btn = $("theme-toggle");
  if (btn) btn.onclick = () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    if (state.view === "dashboard") openDashboard();  // re-render charts in new theme
  };
}

// ---------- toast ----------
function toast(msg, kind = "") {
  const host = $("toast-host");
  if (!host) return;
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 250); }, 2600);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}
const post = (path, payload) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const gbp = (x) => `£${Number(x).toFixed(2)}`;
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[ch]));
const safeClass = (v) => String(v || "").replace(/[^a-z0-9_-]/gi, "");

function showPanel(name) {
  PANELS.forEach((p) => { $("panel-" + p).hidden = p !== name; });
}

async function init() {
  initTheme();
  $("nav-dashboard").onclick = openDashboard;
  $("nav-fixtures").onclick = openFixtures;
  $("nav-outrights").onclick = openOutrights;
  $("nav-history").onclick = openHistory;
  $("nav-bankroll").onclick = openBankroll;
  $("nav-settings").onclick = openSettings;
  try {
    const { engines } = await api("/api/engines");
    state.engines = engines;
    renderSidebar();
    openDashboard();
  } catch (e) {
    $("engine-title").textContent = "Failed to load engines";
    console.error(e);
  }
}

function renderSidebar() {
  const list = $("engine-list");
  list.innerHTML = "";
  state.engines.forEach((eng) => {
    const btn = document.createElement("button");
    const active = state.view === "engine" && state.current === eng.id;
    btn.className = "nav-item" + (active ? " active" : "");
    btn.innerHTML = `<span class="sport-dot"></span>${esc(eng.name)}`;
    btn.onclick = () => selectEngine(eng.id);
    list.appendChild(btn);
  });
  [["dashboard", "nav-dashboard"], ["fixtures", "nav-fixtures"], ["outrights", "nav-outrights"],
   ["history", "nav-history"], ["bankroll", "nav-bankroll"], ["settings", "nav-settings"]]
    .forEach(([v, id]) => $(id).classList.toggle("active", state.view === v));
}

const currentEngine = () => state.engines.find((e) => e.id === state.current);

function selectEngine(id) {
  state.view = "engine";
  state.current = id;
  prefs.set("last-engine", id);
  const eng = currentEngine();
  $("engine-title").textContent = eng.name;
  state.activeCap = eng.capabilities.includes("predict") ? "predict" : eng.capabilities[0];
  renderSidebar();
  renderTabs();
  renderActiveCap();
}

function renderTabs() {
  const eng = currentEngine();
  const tabs = $("tabs");
  tabs.innerHTML = "";
  if (state.view !== "engine") return;
  ALL_CAPS.forEach((cap) => {
    if (!eng.capabilities.includes(cap.id)) return;
    const btn = document.createElement("button");
    btn.className = "tab" + (state.activeCap === cap.id ? " active" : "");
    btn.textContent = cap.label;
    btn.onclick = () => { state.activeCap = cap.id; renderTabs(); renderActiveCap(); };
    tabs.appendChild(btn);
  });
}

function renderActiveCap() {
  const cap = state.activeCap;
  showPanel(cap);
  if (cap === "predict") setupPredict();
  else if (cap === "simulate") setupSimulate();
  else if (cap === "edge") setupEdge();
}

// ---------- PREDICT ----------
function setupPredict() {
  const eng = currentEngine();
  const schema = (eng.schemas && eng.schemas.predict) || eng.predict_schema || {};
  const names = schema.names || [];
  const label = schema.team_label || "Team";
  $("label-team1").textContent = `${label} 1`;
  $("label-team2").textContent = `${label} 2`;
  $("team-names").innerHTML = names.map((n) => `<option value="${esc(n)}"></option>`).join("");
  const models = schema.models || [];
  const modelSel = $("model");
  modelSel.innerHTML = models.map((m) => `<option>${esc(m)}</option>`).join("");
  modelSel.parentElement.style.display = models.length > 1 ? "" : "none";
  const savedModel = prefs.get("model-" + eng.id);
  if (savedModel && models.includes(savedModel)) modelSel.value = savedModel;
  modelSel.onchange = () => prefs.set("model-" + eng.id, modelSel.value);
  // model comparison available when an engine exposes more than one model
  const cmp = $("compare-btn");
  cmp.hidden = models.length < 2;
  cmp.onclick = runCompare;
  $("compare-result").hidden = true;
  renderFilters("predict-filters", schema.filters || [], "predict");

  // Venue control: some engines use a home-advantage toggle (soccer), others a
  // neutral-site toggle (CFB defaults to home). Driven by the schema.
  const wrap = $("home-wrap");
  const cb = $("home");
  const t1 = $("team1");
  if (schema.neutral_toggle) {
    wrap.style.display = "";
    wrap.dataset.mode = "neutral";
    cb.checked = false;
    $("home-team-name").parentElement.firstChild.textContent = "Neutral site";
    $("home-team-name").textContent = "";
  } else if (schema.supports_home) {
    wrap.style.display = "";
    wrap.dataset.mode = "home";
    $("home-team-name").parentElement.firstChild.textContent = "Home advantage for ";
    t1.oninput = () => { $("home-team-name").textContent = t1.value || `${label} 1`; };
    $("home-team-name").textContent = t1.value || `${label} 1`;
  } else {
    wrap.style.display = "none";
    wrap.dataset.mode = "";
  }
  $("predict-btn").onclick = runPredict;
}

async function runPredict() {
  const eng = currentEngine();
  const btn = $("predict-btn"), err = $("predict-error"), result = $("result");
  err.hidden = true;
  const mode = $("home-wrap").dataset.mode;
  const params = {
    team1: $("team1").value.trim(), team2: $("team2").value.trim(),
    model: $("model").value || undefined,
  };
  if (mode === "home") params.home = $("home").checked;
  else if (mode === "neutral") params.neutral = $("home").checked;
  readFilters("predict-filters", "predict", params);
  await withSpin(btn, "Predict", async () => {
    try {
      const r = await post("/api/predict", { engine: eng.id, params });
      renderPredict(r); result.hidden = false;
    } catch (e) { err.textContent = e.message; err.hidden = false; result.hidden = true; }
  });
}

// Generic, engine-agnostic prediction result (competitors, outcome bar,
// optional stat chips, optional table).
function renderPredict(r) {
  const comp = (r.competitors || []).map((c, i) =>
    `<div class="team-block${i ? " right" : ""}"><div class="team-name">${esc(c.name)}</div>
     <div class="elo">${esc(c.sub || "")}</div></div>`);
  const headSep = comp.length === 2 ? `<div class="xg">vs</div>` : "";
  const head = `<div class="result-head">${comp[0] || ""}${headSep}${comp[1] || ""}</div>`;
  const headline = r.headline ? `<div class="headline">${esc(r.headline)}</div>` : "";

  const outs = r.outcomes || [];
  const bar = `<div class="prob-bar">` +
    outs.map((o) => `<div class="seg ${safeClass(o.kind) || "neutral"}" style="width:${o.prob * 100}%">${o.prob > 0.08 ? esc(pct(o.prob)) : ""}</div>`).join("") +
    `</div>`;
  const legend = `<div class="prob-legend">` +
    outs.map((o) => `<span><i class="dot ${safeClass(o.kind) || "neutral"}"></i>${esc(o.label)} <b>${esc(pct(o.prob))}</b></span>`).join("") +
    `</div>`;
  const probs = outs.length ? `<div class="probs">${bar}${legend}</div>` : "";

  const stats = (r.stats && r.stats.length)
    ? `<div class="stat-chips">` + r.stats.map((s) =>
        `<div class="chip"><span class="chip-label">${esc(s.label)}</span><span class="chip-val">${esc(s.value)}</span></div>`).join("") + `</div>`
    : "";

  const table = r.table ? `<div class="scorelines"><div class="scorelines-title">${esc(r.table.title || "")}</div>${renderTable(r.table.columns, r.table.rows, r.table.bar)}</div>` : "";

  $("result").innerHTML = head + headline + probs + stats + table;
}

// Run the same fixture through every model the engine exposes, side by side.
async function runCompare() {
  const eng = currentEngine();
  const models = (eng.schemas.predict || {}).models || [];
  const err = $("predict-error"), box = $("compare-result");
  err.hidden = true;
  const mode = $("home-wrap").dataset.mode;
  const base = { team1: $("team1").value.trim(), team2: $("team2").value.trim() };
  if (mode === "home") base.home = $("home").checked;
  else if (mode === "neutral") base.neutral = $("home").checked;
  readFilters("predict-filters", "predict", base);
  await withSpin($("compare-btn"), "Compare models", async () => {
    try {
      const results = [];
      for (const m of models) {
        const r = await post("/api/predict", { engine: eng.id, params: Object.assign({}, base, { model: m }) });
        results.push({ model: m, r });
      }
      renderCompare(results); box.hidden = false;
    } catch (e) { err.textContent = e.message; err.hidden = false; box.hidden = true; }
  });
}

function renderCompare(results) {
  const labels = (results[0].r.outcomes || []).map((o) => o.label);
  const columns = [{ key: "model", label: "Model", fmt: "text" }]
    .concat(labels.map((l) => ({ key: l, label: l, fmt: "pct" })))
    .concat([{ key: "likely", label: "Likely", fmt: "text" }]);
  const rows = results.map(({ model, r }) => {
    const row = { model };
    (r.outcomes || []).forEach((o) => { row[o.label] = o.prob; });
    if (r.table && r.table.rows && r.table.rows.length) {
      const top = r.table.rows[0];
      row.likely = top.score || top.label || top.scoreline || "";
    } else row.likely = r.headline ? "" : "";
    return row;
  });
  $("compare-result").innerHTML = `<div class="card-title">Model comparison</div>` + renderTable(columns, rows);
}

// Generic table from column metadata. `barKey` (optional) draws a mini-bar
// behind that column's values, scaled to the column max.
function fmtCell(val, fmt) {
  if (val == null || val === "") return "";
  switch (fmt) {
    case "pct": return pct(val);
    case "pct1": return `${(val * 100).toFixed(2)}%`;
    case "gbp": return gbp(val);
    case "gbp_signed": return `${Number(val) >= 0 ? "+" : ""}${gbp(val)}`;
    case "signed_pct": return `${(val * 100 >= 0 ? "" : "") + (val * 100).toFixed(1)}%`;
    case "signed_num": return `${val >= 0 ? "+" : ""}${Number(val).toFixed(2)}`;
    case "num1": return Number(val).toFixed(1);
    case "num": return typeof val === "number" ? String(val) : val;
    default: return val;
  }
}
function renderTable(columns, rows, barKey) {
  if (!rows || !rows.length) return `<table class="data-table"><tr><td class="muted">No rows.</td></tr></table>`;
  const numFmts = new Set(["pct", "pct1", "gbp", "gbp_signed", "signed_pct", "signed_num", "num", "num1"]);
  const head = "<tr>" + columns.map((c) => `<th class="${numFmts.has(c.fmt) ? "num" : ""}">${esc(c.label)}</th>`).join("") + "</tr>";
  const max = barKey ? Math.max(...rows.map((r) => Number(r[barKey]) || 0), 0.0001) : 0;
  const body = rows.map((row) => "<tr>" + columns.map((c) => {
    const raw = row[c.key];
    let cls = numFmts.has(c.fmt) ? "num" : "";
    if (c.fmt === "signed_pct") cls += raw > 0.03 ? " pos" : (raw < 0 ? " neg" : "");
    if (c.fmt === "signed_num") cls += raw > 0 ? " pos" : (raw < 0 ? " neg" : "");
    if (c.fmt === "gbp_signed") cls += raw > 0 ? " pos" : (raw < 0 ? " neg" : "");
    const cell = fmtCell(raw, c.fmt);
    if (barKey && c.key === barKey) {
      const w = Math.max(0, Math.min(100, ((Number(raw) || 0) / max) * 100));
      return `<td class="${cls}"><div class="cellbar"><span style="width:${w}%"></span><em>${esc(cell)}</em></div></td>`;
    }
    return `<td class="${cls}">${esc(cell)}</td>`;
  }).join("") + "</tr>").join("");
  return `<table class="data-table">${head}${body}</table>`;
}

// Interactive table: click-to-sort headers, optional search box, optional
// expandable per-row detail. Mounts into `host` and manages its own state.
const NUM_FMTS = new Set(["pct", "pct1", "gbp", "gbp_signed", "signed_pct", "signed_num", "num", "num1"]);
function mountTable(host, columns, rows, opts = {}) {
  if (!host) return;
  const data = (rows || []).map((r, i) => Object.assign({ _id: i }, r));
  const st = { sort: opts.initialSort || null, q: "", open: new Set() };
  const tools = [];
  if (opts.search) tools.push(`<input class="table-search mt-search" placeholder="Search…">`);
  if (opts.export) tools.push(`<button type="button" class="ghost mt-export">Export CSV</button>`);
  const toolbar = tools.length
    ? `<div class="table-toolbar"><span class="spacer"></span>${tools.join("")}</div>` : "";
  host.innerHTML = toolbar + `<div class="mt-scroll"></div>`;
  const scroll = host.querySelector(".mt-scroll");
  const searchEl = host.querySelector(".mt-search");
  if (searchEl) searchEl.oninput = () => { st.q = searchEl.value.trim().toLowerCase(); draw(); };
  const exportEl = host.querySelector(".mt-export");
  if (exportEl) exportEl.onclick = () => downloadCSV(opts.export, columns, view());

  function view() {
    let r = data.slice();
    if (st.q) r = r.filter((row) => columns.some((c) => String(row[c.key] ?? "").toLowerCase().includes(st.q)));
    if (st.sort) {
      const { key, dir } = st.sort;
      r.sort((a, b) => {
        const av = a[key], bv = b[key];
        const na = parseFloat(av), nb = parseFloat(bv);
        const numeric = !isNaN(na) && !isNaN(nb) && av !== "" && bv !== "" && av != null && bv != null;
        const cmp = numeric ? na - nb : String(av ?? "").localeCompare(String(bv ?? ""));
        return dir === "asc" ? cmp : -cmp;
      });
    }
    return r;
  }
  function draw() {
    const r = view();
    if (!r.length) { scroll.innerHTML = `<table class="data-table"><tr><td class="muted">No rows.</td></tr></table>`; return; }
    const head = "<tr>" + (opts.detail ? "<th></th>" : "") + columns.map((c) => {
      const ind = st.sort && st.sort.key === c.key ? `<span class="sort-ind">${st.sort.dir === "asc" ? "▲" : "▼"}</span>` : "";
      return `<th class="sortable ${NUM_FMTS.has(c.fmt) ? "num" : ""}" data-key="${esc(c.key)}">${esc(c.label)}${ind}</th>`;
    }).join("") + "</tr>";
    const bodyRows = r.map((row) => {
      const cells = columns.map((c) => {
        const raw = row[c.key];
        let cls = NUM_FMTS.has(c.fmt) ? "num" : "";
        if (c.fmt === "signed_pct") cls += raw > 0.0001 ? " pos" : (raw < 0 ? " neg" : "");
        if (c.fmt === "signed_num" || c.fmt === "gbp_signed") cls += raw > 0 ? " pos" : (raw < 0 ? " neg" : "");
        if (c.status) cls += raw === "won" ? " pos" : (raw === "lost" ? " neg" : "");
        return `<td class="${cls}">${esc(fmtCell(raw, c.fmt))}</td>`;
      }).join("");
      const toggle = opts.detail ? `<td class="mt-toggle">${st.open.has(row._id) ? "▾" : "▸"}</td>` : "";
      let tr = `<tr class="${opts.detail ? "mt-row" : ""}" data-id="${row._id}">${toggle}${cells}</tr>`;
      if (opts.detail && st.open.has(row._id))
        tr += `<tr class="mt-detail"><td colspan="${columns.length + 1}">${opts.detail(row, columns) || ""}</td></tr>`;
      return tr;
    }).join("");
    scroll.innerHTML = `<table class="data-table">${head}${bodyRows}</table>`;
    scroll.querySelectorAll("th.sortable").forEach((th) => th.onclick = () => {
      const key = th.dataset.key;
      st.sort = st.sort && st.sort.key === key ? { key, dir: st.sort.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" };
      draw();
    });
    if (opts.detail) scroll.querySelectorAll("tr.mt-row").forEach((tr) => tr.onclick = () => {
      const id = Number(tr.dataset.id);
      st.open.has(id) ? st.open.delete(id) : st.open.add(id);
      draw();
    });
  }
  draw();
}

// CSV export of the currently-shown rows (respects search/filter/sort). Exports
// only the visible columns — no internal (_id) or secret fields ever leak.
function downloadCSV(name, columns, rows) {
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = columns.map((c) => cell(c.label)).join(",");
  const body = (rows || []).map((row) =>
    columns.map((c) => cell(row[c.key])).join(",")).join("\n");
  const blob = new Blob([header + "\n" + body], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${(name || "table").replace(/[^a-z0-9_-]+/gi, "_")}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// Generic key/value drill-down for a table row (shows every field).
function kvDetail(row, columns) {
  const shown = new Map(columns.map((c) => [c.key, c]));
  const pairs = columns.map((c) => [c.label, fmtCell(row[c.key], c.fmt)]);
  Object.keys(row).filter((k) => !shown.has(k) && !k.startsWith("_")).forEach((k) => pairs.push([k, row[k]]));
  return `<div class="kv">` + pairs.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("") + `</div>`;
}

function flash(el) { if (!el) return; el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }

// ---------- SIMULATE ----------
function setupSimulate() {
  const s = currentEngine().schemas.simulate || {};
  fillSelect("sim-model", s.models || ["blend"]);
  const counts = s.sim_options || [10000];
  $("sim-count").innerHTML = counts.map((c) => `<option value="${esc(c)}">${esc(c.toLocaleString())}</option>`).join("");
  if (s.default_sims) $("sim-count").value = s.default_sims;
  $("sim-btn").onclick = runSimulate;
}

async function runSimulate() {
  const eng = currentEngine();
  const btn = $("sim-btn"), err = $("sim-error"), note = $("sim-note"), result = $("sim-result");
  err.hidden = true;
  const params = { model: $("sim-model").value, sims: Number($("sim-count").value) };
  await withSpin(btn, "Run simulation", async () => {
    try {
      const r = await post("/api/simulate", { engine: eng.id, params });
      note.textContent = r.note || ""; note.hidden = false;
      mountTable($("sim-result").querySelector(".table-scroll"), r.columns, r.rows,
        { search: true, export: `${eng.id}_simulation` });
      result.hidden = false;
    } catch (e) { err.textContent = e.message; err.hidden = false; result.hidden = true; }
  });
}

// ---------- EDGE ----------
const edgeState = { full: [], columns: [], engineId: null };

function setupEdge() {
  const eng = currentEngine();
  const s = eng.schemas.edge || {};
  const models = s.models || [];
  fillSelect("edge-model", models);
  $("edge-model").parentElement.style.display = models.length > 1 ? "" : "none";
  $("edge-source").innerHTML = (s.odds_sources || []).map((o) => `<option value="${esc(o.id)}">${esc(o.label)}</option>`).join("");
  $("edge-source").parentElement.style.display = (s.odds_sources || []).length > 1 ? "" : "none";
  const opts = s.adjustments || s.options || [];
  $("edge-options").innerHTML = opts.map((o) =>
    `<label class="checkbox"><input type="checkbox" data-edge-option="${esc(o.id)}" ${o.default ? "checked" : ""}/><span>${esc(o.label)}</span></label>`
  ).join("");
  $("edge-options").style.display = opts.length ? "" : "none";
  renderFilters("edge-filters", s.filters || (eng.schemas.predict && eng.schemas.predict.filters) || [], "edge");
  $("edge-template-btn").style.display = s.has_template ? "" : "none";
  $("edge-btn").onclick = () => runEdge(false);
  $("edge-template-btn").onclick = runEdgeTemplate;
  $("edge-record-btn").onclick = () => runEdge(true);
  $("edge-record-btn").hidden = true;
  $("edge-result").hidden = true;
  $("edge-issues").hidden = true;
  loadEdgeAudit(eng.id);
}

// Compact "is this engine fit to bet?" panel (offline audit endpoint).
async function loadEdgeAudit(engineId) {
  const host = $("edge-audit");
  host.hidden = true;
  try {
    const a = await api(`/api/engines/${encodeURIComponent(engineId)}/audit`);
    const v = a.validation || {};
    const badge = `<span class="audit-badge ${safeClass((v.status || "unknown").toLowerCase())}">${esc(v.status || "unknown")}</span>`;
    const age = a.params_age_days == null ? "—" : `${a.params_age_days.toFixed(1)}d`;
    const warns = (a.freshness_warnings || []).length
      ? `<div class="audit-warn">⚠ ${a.freshness_warnings.map(esc).join(" · ")}</div>` : "";
    const flags = (a.flags || []).map((f) =>
      `<span class="audit-flag ${f.active ? "on" : "off"}">${esc(f.label)}${f.note ? `: ${esc(f.note)}` : ""}</span>`
    ).join("");
    host.innerHTML =
      `<div class="audit-head">Model audit ${badge}
         <span class="audit-meta">params ${esc(age)} old${v.summary ? " · " + esc(v.summary) : ""}</span></div>
       ${warns}<div class="audit-flags">${flags}</div>`;
    host.hidden = false;
  } catch (e) { /* audit is advisory — never block the Edge tab */ }
}

async function runEdgeTemplate() {
  const eng = currentEngine();
  const note = $("edge-note"), err = $("edge-error");
  err.hidden = true;
  try {
    const r = await post("/api/edge/template", { engine: eng.id });
    const where = r.abs_path ? r.abs_path : r.path;
    const rows = (r.rows != null) ? ` (${r.rows} row${r.rows === 1 ? "" : "s"})` : "";
    note.textContent = `Wrote ${where}${rows}. Fill in decimal odds, then choose "Manual odds.csv".`;
    note.hidden = false;
  } catch (e) { err.textContent = e.message; err.hidden = false; }
}

async function runEdge(record) {
  const eng = currentEngine();
  const btn = record ? $("edge-record-btn") : $("edge-btn");
  const err = $("edge-error"), note = $("edge-note"), issues = $("edge-issues"), result = $("edge-result");
  err.hidden = true;
  const params = {
    model: $("edge-model").value, odds_source: $("edge-source").value,
    record: !!record,
  };
  document.querySelectorAll("#edge-options input[data-edge-option]").forEach((i) => {
    params[i.dataset.edgeOption] = i.checked;
  });
  readFilters("edge-filters", "edge", params);
  await withSpin(btn, record ? "Record recommended" : "Find edges", async () => {
    try {
      const r = await post("/api/edge", { engine: eng.id, params });
      let msg = `${r.note} · bankroll ${gbp(r.bankroll)}`;
      if (record) msg += ` · recorded ${r.recorded || 0} bet(s)`;
      note.textContent = msg; note.hidden = false;
      if (r.odds_issues && r.odds_issues.length) {
        issues.textContent = `Odds file issues: ${r.odds_issues.join(" · ")}`;
        issues.hidden = false;
      } else issues.hidden = true;
      edgeState.full = r.rows || [];
      edgeState.columns = r.columns || [];
      edgeState.engineId = eng.id;
      renderEdgeResultFilters();
      applyEdgeFilters();
      result.hidden = false;
      const recCount = edgeState.full.filter((x) => x.recommended).length;
      const recBtn = $("edge-record-btn");
      if (record) { recBtn.hidden = true; if (r.recorded) toast(`Recorded ${r.recorded} bet(s)`, "ok"); }
      else { recBtn.hidden = recCount === 0; recBtn.textContent = `Record ${recCount} recommended`; }
    } catch (e) { err.textContent = e.message; err.hidden = false; result.hidden = true; }
  });
}

// Result-level edge filters (client-side): min edge, min EV, market, source,
// recommended-only. Built from the columns/rows actually returned.
function renderEdgeResultFilters() {
  const host = $("edge-result-filters");
  if (!host) return;
  const rows = edgeState.full;
  const distinct = (key) => [...new Set(rows.map((r) => r[key]).filter((v) => v != null && v !== ""))];
  const hasMarket = rows.some((r) => r.market != null && r.market !== "");
  const hasSource = rows.some((r) => r.source != null && r.source !== "");
  const opt = (vals) => `<option value="">all</option>` + vals.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  host.innerHTML =
    `<label class="rf">min edge %<input type="number" step="0.5" id="ef-edge" class="rf-num"></label>
     <label class="rf">min EV<input type="number" step="0.01" id="ef-ev" class="rf-num"></label>
     ${hasMarket ? `<label class="rf">market<select id="ef-market">${opt(distinct("market"))}</select></label>` : ""}
     ${hasSource ? `<label class="rf">source<select id="ef-source">${opt(distinct("source"))}</select></label>` : ""}
     <label class="rf checkbox"><input type="checkbox" id="ef-rec"><span>recommended only</span></label>`;
  ["ef-edge", "ef-ev", "ef-market", "ef-source", "ef-rec"].forEach((id) => {
    const el = $(id); if (el) el.oninput = el.onchange = applyEdgeFilters;
  });
}

function applyEdgeFilters() {
  const num = (id) => { const el = $(id); const v = el ? parseFloat(el.value) : NaN; return isNaN(v) ? null : v; };
  const val = (id) => { const el = $(id); return el ? el.value : ""; };
  const minEdge = num("ef-edge"), minEV = num("ef-ev");
  const market = val("ef-market"), source = val("ef-source");
  const recOnly = $("ef-rec") && $("ef-rec").checked;
  const rows = edgeState.full.filter((r) => {
    if (minEdge != null && !(Number(r.edge) * 100 >= minEdge)) return false;
    if (minEV != null && !(Number(r.ev_per_unit) >= minEV)) return false;
    if (market && String(r.market) !== market) return false;
    if (source && String(r.source) !== source) return false;
    if (recOnly && !r.recommended) return false;
    return true;
  });
  mountTable($("edge-result").querySelector(".table-scroll"), edgeState.columns, rows,
    { search: true, detail: kvDetail, export: `${edgeState.engineId}_edges` });
}

// ---------- DASHBOARD (suite home) ----------
async function openDashboard() {
  state.view = "dashboard"; state.current = null;
  $("engine-title").textContent = "Dashboard";
  $("tabs").innerHTML = "";
  renderSidebar(); showPanel("dashboard");
  const stats = $("dash-stats"), grid = $("dash-grid");
  stats.innerHTML = `<div class="stat"><div class="skeleton skeleton-line" style="width:60%"></div><div class="skeleton skeleton-line" style="width:40%;height:24px"></div></div>`.repeat(4);
  grid.innerHTML = `<div class="dash-card"><div class="skeleton skeleton-line" style="width:40%"></div><div class="skeleton" style="height:150px;margin-top:12px"></div></div>`.repeat(4);
  try {
    renderDashboard(await api("/api/dashboard"));
  } catch (e) {
    grid.innerHTML = `<div class="dash-card"><div class="empty-mini">Could not load dashboard: ${esc(e.message)}</div></div>`;
  }
}

function statTile(label, value, opts = {}) {
  const cls = opts.cls ? " " + opts.cls : "";
  const sub = opts.sub ? `<div class="stat-sub">${esc(opts.sub)}</div>` : "";
  const spark = opts.spark || "";
  return `<div class="stat"><div class="stat-label">${esc(label)}</div>
    <div class="stat-value${cls}">${esc(value)}</div>${sub}${spark}</div>`;
}

function dashCard(title, meta, bodyHtml, span) {
  const m = meta ? `<span class="h-meta">${esc(meta)}</span>` : "";
  return `<div class="dash-card${span ? " span-2" : ""}"><h2>${esc(title)}${m}</h2>${bodyHtml}</div>`;
}

function renderDashboard(d) {
  const bk = d.bankroll;
  const curveVals = bk.curve.map((p) => p.v);
  const pnlCls = bk.net_pnl > 0 ? "pos" : bk.net_pnl < 0 ? "neg" : "";
  const pnlStr = (bk.net_pnl >= 0 ? "+" : "") + gbp(bk.net_pnl);
  $("dash-stats").innerHTML =
    statTile("Bankroll", gbp(bk.bankroll), { sub: `peak ${gbp(bk.peak)}`, spark: Charts.sparkline(curveVals, { color: Charts.cssVar("--pos") }) }) +
    statTile("Net P&L", pnlStr, { cls: pnlCls, sub: `${bk.settled_count} settled` }) +
    statTile("Open at risk", gbp(bk.open_stake), { sub: `${bk.open_count} open bet${bk.open_count === 1 ? "" : "s"}` }) +
    statTile("Record / ROI", `${bk.won}/${bk.settled_count}`, { sub: `hit ${pct(bk.hit_rate)} · ROI ${(bk.roi >= 0 ? "+" : "") + pct(bk.roi)}` });

  const grid = $("dash-grid");
  grid.innerHTML = "";

  // Bankroll curve (wide)
  grid.insertAdjacentHTML("beforeend", dashCard("Bankroll curve", `start ${gbp(bk.start)}`, `<div class="chart-host" id="ch-bankroll"></div>`, true));
  Charts.areaLine($("ch-bankroll"), curveVals, { color: Charts.cssVar("--pos"), fmt: (v) => "£" + v.toFixed(0), baseline: bk.start });

  // CLV
  grid.insertAdjacentHTML("beforeend", dashCard("CLV trend", "rolling mean", `<div class="chart-host" id="ch-clv"></div>`));
  if (d.clv && d.clv.length) Charts.areaLine($("ch-clv"), d.clv.map((p) => p.v), { color: Charts.cssVar("--accent"), fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(1) + "%", baseline: 0 });
  else $("ch-clv").innerHTML = `<div class="empty-mini">No closing-odds snapshots yet (clv.py --snapshot before kickoffs).</div>`;

  // Calibration
  grid.insertAdjacentHTML("beforeend", dashCard("Calibration", "isotonic vs diagonal", `<div id="ch-cal" class="chart-wrap"></div>`));
  Charts.calibration($("ch-cal"), d.calibration);

  // Fixtures (wide)
  const fx = d.fixtures;
  let fxBody;
  if (fx.rows.length) {
    fxBody = `<table class="data-table"><tr><th>Match</th><th style="width:160px">Win / Draw / Loss</th><th class="num">BTTS</th><th>Likely</th></tr>` +
      fx.rows.map((r) => `<tr><td>${esc(r.match)}</td>
        <td>${Charts.probBar(r.p_home, r.p_draw, r.p_away)}</td>
        <td class="num">${r.p_btts != null ? pct(r.p_btts) : "—"}</td>
        <td>${esc(r.likely)}</td></tr>`).join("") + `</table>`;
  } else fxBody = `<div class="empty-mini">No upcoming fixtures predicted.</div>`;
  grid.insertAdjacentHTML("beforeend", dashCard("Fixtures", fx.day || "", fxBody, true));

  // Bet queue
  const q = d.queue;
  let qBody;
  if (q.rows.length) {
    qBody = `<table class="data-table"><tr><th>Match</th><th>Bet</th><th class="num">Odds</th><th class="num">Edge</th><th class="num">Stake</th></tr>` +
      q.rows.map((r) => `<tr><td>${esc(r.match)}</td><td>${esc(r.bet)}</td>
        <td class="num">${r.odds.toFixed(2)}</td>
        <td class="num pos">+${(r.edge * 100).toFixed(1)}%</td>
        <td class="num">${gbp(r.stake)}</td></tr>`).join("") + `</table>`;
  } else qBody = `<div class="empty-mini">No positive-edge picks queued.</div>`;
  grid.insertAdjacentHTML("beforeend", dashCard("Today's bet queue", q.adjustments || "", qBody, true));

  // Title odds
  const t = d.title;
  const titleBody = Charts.hbars(t.rows.map((r) => ({ label: r.team, value: r.champion, delta: r.delta })));
  grid.insertAdjacentHTML("beforeend", dashCard("Title odds", t.first_snapshot ? "first snapshot" : "▲▼ vs last", titleBody, true));
}

// ---------- shared suite-view helpers ----------
function setSuiteView(view, title) {
  state.view = view; state.current = null;
  $("engine-title").textContent = title;
  $("tabs").innerHTML = "";
  renderSidebar(); showPanel(view);
}

// ---------- FIXTURES (suite) ----------
let fxData = null;
async function openFixtures() {
  setSuiteView("fixtures", "Fixtures");
  $("fx-host").innerHTML = `<div class="dash-card"><div class="skeleton skeleton-line" style="width:30%"></div><div class="skeleton" style="height:120px;margin-top:12px"></div></div>`;
  try {
    fxData = await api("/api/fixtures");
    const days = fxData.days || [];
    $("fx-day").innerHTML = `<option value="">All upcoming</option>` + days.map((d) => `<option value="${esc(d.date)}">${esc(d.date)}</option>`).join("");
    $("fx-day").onchange = renderFixtures;
    $("fx-picks-only").onchange = renderFixtures;
    $("fx-search").oninput = renderFixtures;
    renderFixtures();
  } catch (e) { $("fx-host").innerHTML = `<div class="dash-card"><div class="empty-mini">Could not load fixtures: ${esc(e.message)}</div></div>`; }
}

function renderFixtures() {
  const days = (fxData && fxData.days) || [];
  const dayFilter = $("fx-day").value;
  const picksOnly = $("fx-picks-only").checked;
  const q = $("fx-search").value.trim().toLowerCase();
  const host = $("fx-host");
  host.innerHTML = "";
  let shown = 0;
  days.filter((d) => !dayFilter || d.date === dayFilter).forEach((day) => {
    let rows = day.rows;
    if (picksOnly) rows = rows.filter((r) => r.picks.length);
    if (q) rows = rows.filter((r) => r.match.toLowerCase().includes(q));
    if (!rows.length) return;
    shown += rows.length;
    const body = `<table class="data-table"><tr><th>Match</th><th style="width:170px">Win / Draw / Loss</th><th class="num">BTTS</th><th class="num">xG</th><th>Likely</th><th>Edge picks</th></tr>` +
      rows.map((r) => {
        const picks = r.picks.length
          ? r.picks.map((p) => `<span class="badge accent" title="edge +${(p.edge * 100).toFixed(1)}%">${esc(p.bet)} @ ${p.odds.toFixed(2)}</span>`).join(" ")
          : `<span class="muted">—</span>`;
        return `<tr><td><b>${esc(r.match)}</b></td>
          <td>${Charts.probBar(r.p_home, r.p_draw, r.p_away)}</td>
          <td class="num">${r.p_btts != null ? pct(r.p_btts) : "—"}</td>
          <td class="num">${r.xg_home.toFixed(1)}–${r.xg_away.toFixed(1)}</td>
          <td>${esc(r.likely)}</td><td>${picks}</td></tr>`;
      }).join("") + `</table>`;
    host.insertAdjacentHTML("beforeend", dashCard(day.date, `${rows.length} match${rows.length === 1 ? "" : "es"}`, body, true));
  });
  if (!shown) host.innerHTML = `<div class="dash-card"><div class="empty-mini">No fixtures match these filters.</div></div>`;
}

// ---------- OUTRIGHTS (suite) ----------
async function openOutrights() {
  setSuiteView("outrights", "Outrights");
  $("or-table").innerHTML = `<div class="skeleton" style="height:200px"></div>`;
  $("or-chart").innerHTML = "";
  try {
    const d = await api("/api/outrights");
    $("or-meta").textContent = d.first_snapshot ? "first snapshot" : (d.dates.length ? `${d.dates.length} snapshots` : "");
    const top = d.teams.slice(0, 8);
    const lines = top.map((t, i) => ({ label: t.team, color: Charts.PALETTE[i % Charts.PALETTE.length], points: t.series }));
    Charts.multiLine($("or-chart"), lines, d.dates, { fmt: (v) => v.toFixed(0) + "%" });
    $("or-legend").innerHTML = top.map((t, i) => `<span><i style="background:${Charts.PALETTE[i % Charts.PALETTE.length]}"></i>${esc(t.team)}</span>`).join("");
    $("or-table").innerHTML = Charts.hbars(d.teams.map((t) => ({ label: t.team, value: t.champion, delta: t.delta })));
  } catch (e) { $("or-table").innerHTML = `<div class="empty-mini">Could not load outrights: ${esc(e.message)}</div>`; }
}

// ---------- HISTORY (bet explorer) ----------
let histData = null;
async function openHistory() {
  setSuiteView("history", "Bet history");
  $("hist-stats").innerHTML = `<div class="stat"><div class="skeleton skeleton-line" style="width:50%"></div><div class="skeleton skeleton-line" style="width:40%;height:22px"></div></div>`.repeat(4);
  $("hist-table").innerHTML = "";
  try {
    histData = await api("/api/history");
    renderHistory(histData);
  } catch (e) { $("hist-table").innerHTML = `<div class="empty-mini">Could not load history: ${esc(e.message)}</div>`; }
}

function renderHistory(d) {
  const s = d.summary || {};
  if (!d.rows || !d.rows.length) {
    $("hist-stats").innerHTML = "";
    $("hist-table").innerHTML = `<div class="empty-mini">No bets recorded yet.</div>`;
    return;
  }
  const pnlCls = s.net_pnl > 0 ? "pos" : s.net_pnl < 0 ? "neg" : "";
  $("hist-stats").innerHTML =
    statTile("Net P&L", (s.net_pnl >= 0 ? "+" : "") + gbp(s.net_pnl), { cls: pnlCls, sub: `£${s.staked} staked` }) +
    statTile("ROI", (s.roi >= 0 ? "+" : "") + pct(s.roi), { cls: s.roi >= 0 ? "pos" : "neg", sub: "on settled stakes" }) +
    statTile("Hit rate", pct(s.hit_rate), { sub: `${s.won}/${s.settled} won` }) +
    statTile("Bets", `${s.total}`, { sub: `${s.open} open · ${s.settled} settled` });

  Charts.areaLine($("hist-curve"), (d.pnl_curve || []).map((p) => p.v), { color: Charts.cssVar(s.net_pnl >= 0 ? "--pos" : "--neg"), fmt: (v) => (v >= 0 ? "+" : "") + "£" + v.toFixed(0), baseline: 0 });

  $("hist-market").innerHTML = renderTable([
    { key: "market", label: "Market", fmt: "text" },
    { key: "settled", label: "Settled", fmt: "num" },
    { key: "net_pnl", label: "P&L", fmt: "gbp_signed" },
    { key: "roi", label: "ROI", fmt: "signed_pct" },
  ], d.by_market || []);
  $("hist-sport").innerHTML = renderTable([
    { key: "sport", label: "Sport", fmt: "text" },
    { key: "settled", label: "Settled", fmt: "num" },
    { key: "net_pnl", label: "P&L", fmt: "gbp_signed" },
    { key: "roi", label: "ROI", fmt: "signed_pct" },
  ], d.by_sport || []);

  const opt = (arr) => `<option value="">All</option>` + arr.map((x) => `<option>${esc(x)}</option>`).join("");
  $("hist-f-sport").innerHTML = opt(d.options.sports);
  $("hist-f-market").innerHTML = opt(d.options.markets);
  $("hist-f-status").innerHTML = opt(d.options.statuses);
  ["hist-f-sport", "hist-f-market", "hist-f-status"].forEach((id) => $(id).onchange = drawHistoryTable);
  $("hist-search").oninput = drawHistoryTable;
  drawHistoryTable();
}

function drawHistoryTable() {
  const fs = $("hist-f-sport").value, fm = $("hist-f-market").value, fst = $("hist-f-status").value;
  const q = $("hist-search").value.trim().toLowerCase();
  let rows = histData.rows;
  if (fs) rows = rows.filter((r) => r.sport === fs);
  if (fm) rows = rows.filter((r) => r.market === fm);
  if (fst) rows = rows.filter((r) => r.status === fst);
  if (q) rows = rows.filter((r) => (r.match + " " + r.bet).toLowerCase().includes(q));
  mountTable($("hist-table"), [
    { key: "match_date", label: "Date", fmt: "text" },
    { key: "match", label: "Match", fmt: "text" },
    { key: "bet", label: "Bet", fmt: "text" },
    { key: "market", label: "Market", fmt: "text" },
    { key: "sport", label: "Sport", fmt: "text" },
    { key: "odds", label: "Odds", fmt: "num" },
    { key: "stake", label: "Stake", fmt: "gbp" },
    { key: "status", label: "Result", fmt: "text", status: true },
    { key: "pnl", label: "P&L", fmt: "gbp_signed" },
  ], rows, { initialSort: { key: "match_date", dir: "desc" }, detail: kvDetail });
}

// ---------- BANKROLL (suite) ----------
async function openBankroll() {
  state.view = "bankroll"; state.current = null;
  $("engine-title").textContent = "Bankroll";
  $("tabs").innerHTML = "";
  renderSidebar(); showPanel("bankroll");
  $("bk-settle").onclick = () => withSpin($("bk-settle"), "Settle results", async () => {
    const r = await post("/api/bankroll", { action: "settle" });
    const res = r.result || {};
    renderBankroll(r); flash($("bk-bankroll")); flash($("bk-pnl"));
    toast(res.settled ? `Settled ${res.settled} bet(s) · ${res.still_open || 0} still open` : "No bets ready to settle yet", res.settled ? "pos" : "");
  });
  $("bk-reset").onclick = async () => {
    const amt = Number($("bk-reset-amt").value);
    if (!amt || amt <= 0) { toast("Enter a reset amount first", "neg"); return; }
    if (!confirm(`Reset the shared bankroll to ${gbp(amt)}? The current ledger is backed up.`)) return;
    const r = await post("/api/bankroll", { action: "reset", amount: amt });
    renderBankroll(r); flash($("bk-bankroll"));
    toast(`Bankroll reset to ${gbp(amt)} — ledger backed up`, "pos");
  };
  try { renderBankroll(await post("/api/bankroll", { action: "status" })); }
  catch (e) { $("bk-note").textContent = e.message; $("bk-note").hidden = false; }
}

function renderBankroll(d) {
  $("bk-bankroll").textContent = gbp(d.bankroll);
  $("bk-peak").textContent = gbp(d.peak);
  const pnl = d.totals.net_pnl;
  const pe = $("bk-pnl");
  pe.textContent = (pnl >= 0 ? "+" : "") + gbp(pnl).replace("£", "£");
  pe.className = "stat-value " + (pnl > 0 ? "pos" : pnl < 0 ? "neg" : "");
  $("bk-risk").textContent = gbp(d.totals.open_stake);

  mountTable($("bk-sport-host"), [
    { key: "sport", label: "Sport", fmt: "text" },
    { key: "open", label: "Open", fmt: "num" },
    { key: "settled", label: "Settled", fmt: "num" },
    { key: "net_pnl", label: "Net P&L", fmt: "gbp_signed" },
  ], d.by_sport || []);

  mountTable($("bk-open-host"), [
    { key: "sport", label: "Sport", fmt: "text" },
    { key: "match_date", label: "Date", fmt: "text" },
    { key: "match", label: "Match", fmt: "text" },
    { key: "bet", label: "Bet", fmt: "text" },
    { key: "odds", label: "Odds", fmt: "num" },
    { key: "stake", label: "Stake", fmt: "gbp" },
  ], d.open || [], { search: true, initialSort: { key: "match_date", dir: "asc" } });

  mountTable($("bk-settled-host"), [
    { key: "sport", label: "Sport", fmt: "text" },
    { key: "match_date", label: "Date", fmt: "text" },
    { key: "match", label: "Match", fmt: "text" },
    { key: "bet", label: "Bet", fmt: "text" },
    { key: "odds", label: "Odds", fmt: "num" },
    { key: "stake", label: "Stake", fmt: "gbp" },
    { key: "status", label: "Result", fmt: "text", status: true },
    { key: "pnl", label: "P&L", fmt: "gbp_signed" },
  ], d.settled || [], { search: true });
}

// ---------- SETTINGS (suite) ----------
async function openSettings() {
  state.view = "settings"; state.current = null;
  $("engine-title").textContent = "Settings";
  $("tabs").innerHTML = "";
  renderSidebar(); showPanel("settings");
  const s = await api("/api/settings");
  $("settings-keys").innerHTML = s.sources.map((src) => `
    <label class="field key-field">
      <span class="field-label">${esc(src.label)}</span>
      <input data-src="${esc(src.id)}" placeholder="${esc(s.odds_api_keys_set[src.id] ? "saved (" + s.odds_api_keys_masked[src.id] + ") — type to replace" : "not set")}" autocomplete="off" />
    </label>`).join("");
  $("set-model").value = s.default_model;
  $("set-kelly").value = s.default_kelly;
  $("settings-save").onclick = saveSettings;
}

async function saveSettings() {
  const keys = {};
  document.querySelectorAll("#settings-keys input").forEach((i) => {
    if (i.value.trim()) keys[i.dataset.src] = i.value.trim();
  });
  const patch = { default_model: $("set-model").value, default_kelly: Number($("set-kelly").value) };
  if (Object.keys(keys).length) patch.odds_api_keys = keys;
  await post("/api/settings", patch);
  const saved = $("settings-saved");
  saved.hidden = false;
  setTimeout(() => { saved.hidden = true; }, 1800);
  openSettings();
}

// ---------- helpers ----------
function fillSelect(id, opts) { $(id).innerHTML = opts.map((o) => `<option>${esc(o)}</option>`).join(""); }
function renderFilters(id, filters, prefix) {
  const wrap = $(id);
  if (!wrap) return;
  if (!filters || !filters.length) {
    wrap.innerHTML = "";
    wrap.style.display = "none";
    return;
  }
  wrap.innerHTML = filters.map((f) => {
    if (f.type === "date") {
      return `<label class="field filter-field">
        <span class="field-label">${esc(f.label || f.id)}</span>
        <input type="date" data-${prefix}-filter="${esc(f.id)}" />
      </label>`;
    }
    const options = (f.options || []).map((o) => {
      const value = typeof o === "object" ? o.value : o;
      const label = typeof o === "object" ? (o.label ?? o.value) : o;
      return `<option value="${esc(value)}">${esc(label)}</option>`;
    }).join("");
    return `<label class="field filter-field">
      <span class="field-label">${esc(f.label || f.id)}</span>
      <select data-${prefix}-filter="${esc(f.id)}">${options}</select>
    </label>`;
  }).join("");
  wrap.style.display = "";
}
function readFilters(id, prefix, params) {
  document.querySelectorAll(`#${id} [data-${prefix}-filter]`).forEach((s) => {
    const value = s.value;
    if (value !== "") params[s.dataset[`${prefix}Filter`]] = value;
  });
}
async function withSpin(btn, label, fn) {
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>';
  try { await fn(); } finally { btn.disabled = false; btn.textContent = label; }
}

init();
