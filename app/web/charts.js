"use strict";
/* Sports Predictor — tiny dependency-free SVG chart kit.
   Theme-aware (reads CSS variables), offline, reused by dashboard + panels. */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const cssVar = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  const num = (v) => (typeof v === "number" ? v : (v && v.v));

  // ---- area/line chart with hover tooltip -------------------------------
  // host: container element. series: [{i,v}] or [numbers]. opts: {color, fmt, baseline, height}
  function areaLine(host, series, opts = {}) {
    host.innerHTML = "";
    const vals = series.map(num);
    if (vals.length < 2) { host.innerHTML = `<div class="empty-mini">Not enough data yet.</div>`; return; }
    const w = 560, h = opts.height || 180, padL = 38, padR = 12, padT = 14, padB = 22;
    const color = opts.color || cssVar("--accent");
    const fmt = opts.fmt || ((v) => v.toFixed(2));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (opts.baseline != null) { lo = Math.min(lo, opts.baseline); hi = Math.max(hi, opts.baseline); }
    if (hi === lo) hi = lo + 1;
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = (i) => padL + (w - padL - padR) * (i / (vals.length - 1));
    const Y = (v) => h - padB - (h - padT - padB) * ((v - lo) / (hi - lo));

    let grid = "", axis = "";
    for (let g = 0; g <= 2; g++) {
      const frac = g / 2, y = padT + (h - padT - padB) * frac, val = hi - (hi - lo) * frac;
      grid += `<line class="chart-grid" x1="${padL}" y1="${y.toFixed(1)}" x2="${w - padR}" y2="${y.toFixed(1)}"/>`;
      axis += `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${fmt(val)}</text>`;
    }
    let base = "";
    if (opts.baseline != null && opts.baseline >= lo && opts.baseline <= hi) {
      const yb = Y(opts.baseline);
      base = `<line class="chart-baseline" x1="${padL}" y1="${yb.toFixed(1)}" x2="${w - padR}" y2="${yb.toFixed(1)}"/>`;
    }
    const linePts = vals.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    const areaPts = `${X(0).toFixed(1)},${(h - padB)} ${linePts} ${X(vals.length - 1).toFixed(1)},${h - padB}`;
    const gid = "g" + Math.random().toString(36).slice(2, 8);

    host.classList.add("chart-wrap");
    host.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" class="chart-axis" preserveAspectRatio="none">
        <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.28"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient></defs>
        ${grid}${base}
        <polygon points="${areaPts}" fill="url(#${gid})"/>
        <polyline points="${linePts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>
        <circle class="chart-dot" r="4" fill="${color}" stroke="var(--surface)" stroke-width="2" style="opacity:0"/>
        ${axis}
      </svg>
      <div class="chart-tip"></div>`;
    // hover
    const svg = host.querySelector("svg"), dot = host.querySelector(".chart-dot"), tip = host.querySelector(".chart-tip");
    svg.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const xRel = ((e.clientX - r.left) / r.width) * w;
      let i = Math.round(((xRel - padL) / (w - padL - padR)) * (vals.length - 1));
      i = Math.max(0, Math.min(vals.length - 1, i));
      const cx = X(i), cy = Y(vals[i]);
      dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); dot.style.opacity = 1;
      tip.style.left = (cx / w * 100) + "%"; tip.style.top = (cy / h * 100) + "%";
      tip.textContent = fmt(vals[i]); tip.classList.add("show");
    });
    svg.addEventListener("mouseleave", () => { dot.style.opacity = 0; tip.classList.remove("show"); });
  }

  // ---- sparkline (inline, no axes) --------------------------------------
  function sparkline(values, opts = {}) {
    const vals = values.map(num);
    const w = opts.w || 88, h = opts.h || 26, color = opts.color || cssVar("--accent");
    if (vals.length < 2) return "";
    let lo = Math.min(...vals), hi = Math.max(...vals); if (hi === lo) hi = lo + 1;
    const X = (i) => (w) * (i / (vals.length - 1));
    const Y = (v) => h - 2 - (h - 4) * ((v - lo) / (hi - lo));
    const pts = vals.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    return `<svg class="stat-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  }

  // ---- horizontal bars (title odds etc.) --------------------------------
  // rows: [{label, value, pct(0-1), delta?}]
  function hbars(rows, opts = {}) {
    if (!rows || !rows.length) return `<div class="empty-mini">No data.</div>`;
    const max = Math.max(...rows.map((r) => r.value), 1e-9);
    const color = opts.color || cssVar("--accent");
    const fmt = opts.fmt || ((v) => (v * 100).toFixed(1) + "%");
    return `<table class="data-table hbar-table">` + rows.map((r) => {
      const w = Math.max(2, (r.value / max) * 100);
      let d = "";
      if (r.delta != null) {
        const up = r.delta > 0;
        d = `<span class="badge ${up ? "pos" : "neg"}">${up ? "▲" : "▼"}${Math.abs(r.delta).toFixed(1)}</span>`;
      }
      return `<tr>
        <td style="width:30%">${esc(r.label)}</td>
        <td style="width:46%"><div class="hbar-track"><span style="width:${w.toFixed(0)}%;background:${color}"></span></div></td>
        <td class="num" style="width:14%">${fmt(r.value)}</td>
        <td class="num" style="width:10%">${d}</td></tr>`;
    }).join("") + `</table>`;
  }

  // ---- multi-line chart (title-race movement) ---------------------------
  // lines: [{label, color, points:[{date,v}]}]  dates: ordered x labels
  function multiLine(host, lines, dates, opts = {}) {
    host.innerHTML = "";
    const usable = (lines || []).filter((l) => l.points.some((p) => p.v != null));
    if (!usable.length || dates.length < 2) { host.innerHTML = `<div class="empty-mini">Not enough snapshots yet — movement appears after the next daily run.</div>`; return; }
    const w = 560, h = opts.height || 200, padL = 38, padR = 12, padT = 12, padB = 24;
    const fmt = opts.fmt || ((v) => v.toFixed(1) + "%");
    let lo = Infinity, hi = -Infinity;
    usable.forEach((l) => l.points.forEach((p) => { if (p.v != null) { lo = Math.min(lo, p.v); hi = Math.max(hi, p.v); } }));
    if (hi === lo) hi = lo + 1;
    const padv = (hi - lo) * 0.1; lo = Math.max(0, lo - padv); hi += padv;
    const X = (i) => padL + (w - padL - padR) * (i / (dates.length - 1));
    const Y = (v) => h - padB - (h - padT - padB) * ((v - lo) / (hi - lo));
    let grid = "", axis = "";
    for (let g = 0; g <= 2; g++) {
      const frac = g / 2, y = padT + (h - padT - padB) * frac, val = hi - (hi - lo) * frac;
      grid += `<line class="chart-grid" x1="${padL}" y1="${y.toFixed(1)}" x2="${w - padR}" y2="${y.toFixed(1)}"/>`;
      axis += `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${fmt(val)}</text>`;
    }
    let paths = "";
    usable.forEach((l) => {
      const pts = l.points.map((p, i) => p.v == null ? null : `${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`).filter(Boolean).join(" ");
      paths += `<polyline points="${pts}" fill="none" stroke="${l.color}" stroke-width="2" stroke-linejoin="round"/>`;
      const last = l.points[l.points.length - 1];
      if (last && last.v != null) paths += `<circle cx="${X(dates.length - 1).toFixed(1)}" cy="${Y(last.v).toFixed(1)}" r="3" fill="${l.color}"/>`;
    });
    host.classList.add("chart-wrap");
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" class="chart-axis" preserveAspectRatio="none">${grid}${paths}
      <text x="${padL}" y="${h - 4}">${esc(dates[0])}</text>
      <text x="${w - padR}" y="${h - 4}" text-anchor="end">${esc(dates[dates.length - 1])}</text></svg>`;
  }

  // ---- calibration scatter/line -----------------------------------------
  function calibration(host, sides) {
    if (!sides) { host.innerHTML = `<div class="empty-mini">Not fitted yet (validate.py --calibrate).</div>`; return; }
    const w = 240, h = 240, pad = 26;
    const X = (v) => pad + (w - 2 * pad) * v, Y = (v) => h - pad - (h - 2 * pad) * v;
    const colors = { home: cssVar("--win"), draw: cssVar("--warn"), away: cssVar("--pos") };
    let parts = `<line class="chart-baseline" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>`;
    for (const side of ["home", "draw", "away"]) {
      const m = sides[side]; if (!m) continue;
      const pts = m.map((p) => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
      parts += `<polyline points="${pts}" fill="none" stroke="${colors[side]}" stroke-width="2"/>`;
    }
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" class="chart-axis">${parts}
      <text x="${w / 2}" y="${h - 4}" text-anchor="middle">predicted →</text></svg>
      <div class="chart-legend">
        <span><i style="background:${colors.home}"></i>home</span>
        <span><i style="background:${colors.draw}"></i>draw</span>
        <span><i style="background:${colors.away}"></i>away</span></div>`;
  }

  // shared win/draw/loss probability bar
  function probBar(home, draw, away) {
    const seg = (p, cls) => `<div class="seg ${cls}" style="width:${(p * 100).toFixed(1)}%">${p > 0.1 ? (p * 100).toFixed(0) : ""}</div>`;
    return `<div class="prob-bar">${seg(home, "win")}${draw != null ? seg(draw, "draw") : ""}${seg(away, "loss")}</div>`;
  }

  function esc(v) {
    return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // palette for multi-series charts
  const PALETTE = ["#4f6bff", "#2fbf71", "#e0a83a", "#f4636b", "#9b6bff", "#21b8c4", "#ec6f4c", "#5a9bff"];

  return { areaLine, sparkline, hbars, calibration, probBar, multiLine, cssVar, PALETTE };
})();
window.Charts = Charts;
