// Bubbles: zoomable D3 circle packing of the occupation hierarchy.
// Data comes from data/hierarchy.json, built by src/build_hierarchy.py.

const SIZE = 1000;      // layout coordinate space (the SVG viewBox is SIZE x SIZE)
const LABEL_PX = 13;    // on-screen label size in pixels

// Brand palette, plus four blends of it so all nine groups get their own color
const PALETTE = { navy: "#23335f", sand: "#c3aa87", blue: "#2b5a87", sage: "#80a68a", terracotta: "#c48b71" };
const mix = (a, b, t) => d3.interpolateLab(a, b)(t);
const GROUP_COLORS = {
  Care: PALETTE.terracotta,
  Guide: PALETTE.sand,
  Protect: PALETTE.navy,
  Build: d3.color(PALETTE.sand).darker(1).formatHex(),
  Invent: PALETTE.blue,
  Discover: PALETTE.sage,
  Create: mix(PALETTE.terracotta, PALETTE.navy, 0.5),
  Connect: d3.color(PALETTE.terracotta).darker(0.7).formatHex(),
  Organize: mix(PALETTE.blue, PALETTE.sage, 0.5),
};
const FALLBACK = Object.values(PALETTE);
const TINT = { 1: 0.16, 2: 0.3, 3: 0.44 };  // how strongly cluster circles are tinted, by depth
const JOB_TINT = 0.9;

const JOB_ZONES = {
  1: "Little or no preparation needed",
  2: "Some preparation (e.g. high school)",
  3: "Medium preparation (e.g. vocational training)",
  4: "Considerable preparation (usually a bachelor's degree)",
  5: "Extensive preparation (usually graduate school)",
};
const INTERESTS = {
  Realistic: "practical, hands-on work",
  Investigative: "research, ideas and analysis",
  Artistic: "creativity and self-expression",
  Social: "helping, teaching and caring for people",
  Enterprising: "leading, persuading and business",
  Conventional: "organizing, data and detail work",
};

const chartEl = document.getElementById("chart");
const panel = document.getElementById("panel");
const crumbs = document.getElementById("breadcrumb");
const groupNav = document.getElementById("groups");
const darkMode = window.matchMedia("(prefers-color-scheme: dark)");

const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isJob = d => !d.children;
const countJobs = d => d.leaves().length;
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

fetch("data/hierarchy.json", { cache: "no-cache" })
  .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then(
    data => {
      try {
        init(data);
      } catch (err) {
        // The data loaded, but drawing failed (e.g. a stale cached page): say so instead of blaming the server
        console.error(err);
        panel.innerHTML = `<h2>Something went wrong drawing the bubbles</h2>
          <p class="meta">${esc(err.message)}</p>
          <p>Try a hard refresh (<b>Ctrl+F5</b>, or <b>Cmd+Shift+R</b> on Mac) to load the latest version of the page.</p>`;
      }
    },
    err => {
      console.error(err);
      panel.innerHTML = `<h2>Couldn't load the data</h2>
        <p class="meta">${esc(err.message)}</p>
        <p>Open this page through a local web server so it can read <code>data/hierarchy.json</code>:</p>
        <pre>cd web\npython -m http.server 8000</pre>
        <p>then visit <a href="http://localhost:8000">localhost:8000</a>.</p>`;
    });

function init(data) {
  const root = d3.pack()
    .size([SIZE, SIZE])
    .padding(d => (d.depth === 0 ? 16 : d.depth === 1 ? 6 : 3))(
      d3.hierarchy(data).sum(d => d.value || 0).sort((a, b) => b.value - a.value));

  root.children.forEach((group, i) => {
    const base = GROUP_COLORS[group.data.name] || FALLBACK[i % FALLBACK.length];
    group.descendants().forEach(d => (d.base = base));
  });
  const groupsInOrder = data.children.map(c => root.children.find(g => g.data === c));
  const jobsByName = new Map(root.leaves().map(d => [d.data.name.toLowerCase(), d]));

  // Colors adapt to the theme: in dark mode, deep colors are lifted so they stand out on navy
  const colorOf = d => {
    if (!darkMode.matches) return d.base;
    const c = d3.hsl(d.base);
    c.l = Math.max(c.l, 0.6);
    return c.formatHex();
  };
  const fillOf = d => mix(cssVar("--bg"), colorOf(d), isJob(d) ? JOB_TINT : TINT[d.depth] ?? 0.5);
  const inkOn = color => (d3.lab(color).l > 62 ? "#1c2747" : "#ffffff"); // readable text on a filled bubble

  // ---------- drawing ----------
  const svg = d3.select(chartEl).append("svg")
    .attr("viewBox", [-SIZE / 2, -SIZE / 2, SIZE, SIZE])
    .attr("preserveAspectRatio", "xMidYMid meet")
    .on("click", () => focus !== root && zoom(focus.parent));

  const node = svg.append("g").selectAll("circle")
    .data(root.descendants().slice(1))
    .join("circle")
    .attr("class", d => `node ${isJob(d) ? "job" : "cluster"}${d.data.niche && !isJob(d) ? " niche" : ""}`)
    .on("click", (event, d) => {
      event.stopPropagation();
      if (isJob(d)) focus !== d.parent ? zoom(d.parent, d) : showJob(d);
      else if (focus !== d) zoom(d);
    });
  node.append("title").text(d => (isJob(d) ? d.data.name : `${d.data.name} · ${countJobs(d)} jobs`));

  const label = svg.append("g").attr("class", "labels").selectAll("text")
    .data(root.descendants())
    .join("text")
    .attr("class", d => (isJob(d) ? "job" : d.depth === 1 ? "cluster group" : "cluster"))
    .style("display", d => (d.parent === root ? "inline" : "none"));

  function paint() {
    node.attr("fill", fillOf).style("--swatch", colorOf);
    label.filter(isJob).style("fill", d => inkOn(fillOf(d)));
    renderGroupNav();
  }

  let focus = root;
  let view;
  let lastSearch = null; // { query, hits, semantic } while search results are active

  // ---------- labels ----------
  const unitsPerPx = () => SIZE / Math.max(1, Math.min(chartEl.clientWidth, chartEl.clientHeight));
  const viewFor = d => [d.x, d.y, d.r * 2 + (d === root ? 0 : 10)];

  // Greedy word wrap into lines of at most maxChars; the last allowed line gets an ellipsis.
  function wrap(text, maxChars, maxLines) {
    const lines = [];
    let line = "";
    for (const word of text.split(/\s+/)) {
      const next = line ? `${line} ${word}` : word;
      if (next.length <= maxChars || !line) line = next;
      else { lines.push(line); line = word; }
    }
    if (line) lines.push(line);
    const out = lines.slice(0, maxLines).map(l => (l.length > maxChars ? l.slice(0, maxChars - 1) + "…" : l));
    if (lines.length > maxLines) out[maxLines - 1] = out[maxLines - 1].replace(/.?…?$/, "") + "…";
    return out;
  }

  // Fit each visible label inside its circle at the zoom level we're heading to.
  function setLabelText(targetView) {
    const k = SIZE / targetView[2];
    const unit = unitsPerPx();
    label.each(function (d) {
      if (d.parent !== focus) return;
      const isGroup = d.depth === 1;
      const px = isJob(d) ? LABEL_PX - 1.5 : isGroup ? LABEL_PX * 1.35 : LABEL_PX;
      const fontUnits = px * unit;
      const diameter = d.r * k * 2 * 0.84;
      const maxChars = Math.floor(diameter / (fontUnits * 0.56));
      const maxLines = Math.max(1, Math.min(3, Math.floor(diameter / (fontUnits * 1.25))));
      const lines = maxChars < 4 ? [] : wrap(d.data.name, maxChars, maxLines).map(t => ({ t, cls: "" }));
      if (isGroup && lines.length) lines.push({ t: `${countJobs(d)} jobs`, cls: "count", small: true });
      const total = lines.length;
      d3.select(this).attr("font-size", fontUnits)
        .selectAll("tspan").data(lines).join("tspan")
        .attr("class", l => l.cls)
        .attr("x", 0)
        .attr("font-size", l => (l.small ? fontUnits * 0.62 : null))
        .attr("dy", (l, i) => (i === 0 ? `${0.35 - (total - 1) * 0.6}em` : l.small ? "1.7em" : "1.2em"))
        .text(l => l.t);
    });
  }

  // ---------- zoom ----------
  function zoomTo(v) {
    const k = SIZE / v[2];
    view = v;
    label.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
    node.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`).attr("r", d => d.r * k);
  }

  function zoom(d, jobToShow = null, showPanel = null) {
    focus = d;
    svg.classed("overview", d === root);
    const target = viewFor(d);
    setLabelText(target);
    const t = svg.transition().duration(700).ease(d3.easeCubicInOut).tween("zoom", () => {
      const i = d3.interpolateZoom(view, target);
      return u => zoomTo(i(u));
    });
    label
      .filter(function (l) { return l.parent === focus || this.style.display === "inline"; })
      .transition(t)
      .style("fill-opacity", l => (l.parent === focus ? 1 : 0))
      .on("start", function (l) { if (l.parent === focus) this.style.display = "inline"; })
      .on("end", function (l) { if (l.parent !== focus) this.style.display = "none"; });
    renderBreadcrumb();
    renderGroupNav();
    if (showPanel) showPanel();
    else if (jobToShow) showJob(jobToShow);
    else showCluster(d);
  }

  // ---------- navigation ----------
  const groupOf = d => (d === root ? null : d.ancestors().find(a => a.depth === 1));

  function renderGroupNav() {
    const active = groupOf(focus);
    groupNav.innerHTML = groupsInOrder.map((g, i) => `
      <button class="chip${g === active ? " active" : ""}" data-i="${i}" style="--swatch:${colorOf(g)}">
        <span class="dot"></span>${esc(g.data.name)} <span class="n">${countJobs(g)}</span>
      </button>`).join("");
    groupNav.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      const g = groupsInOrder[+b.dataset.i];
      zoom(focus === g ? root : g);
    }));
  }

  function renderBreadcrumb() {
    const trail = focus.ancestors().reverse();
    const name = a => esc(a === root ? "All careers" : a.data.name);
    crumbs.innerHTML = trail.map((a, i) => i === trail.length - 1
      ? `<span class="here">${name(a)}</span>`
      : `<button data-i="${i}">${name(a)}</button><span class="sep">/</span>`).join("");
    crumbs.querySelectorAll("button").forEach(b => b.addEventListener("click", () => zoom(trail[+b.dataset.i])));
  }

  // ---------- side panel ----------
  const select = d => node.classed("selected", n => n === d);
  const pathOf = d => d.ancestors().reverse().slice(1, -1).map(a => esc(a.data.name)).join(" / ");
  const row = (k, i) => `
    <button class="row" data-i="${i}" style="--swatch:${colorOf(k)}">
      <span class="name"><span class="dot"></span><span>${esc(k.data.name)}${k.data.tagline ? `<small>${esc(k.data.tagline)}</small>` : ""}</span></span>
      ${isJob(k) ? "" : `<span class="n">${countJobs(k)}</span>`}
    </button>`;

  function showCluster(d) {
    select(null);
    if (d === root) {
      panel.style.removeProperty("--swatch");
      panel.innerHTML = `
        <p class="eyebrow">Start here</p>
        <h2>What kind of work do you want to do?</h2>
        <p class="meta">Pick a direction, or click any bubble to zoom in.</p>
        <div class="list" style="margin-top:14px">${groupsInOrder.map(row).join("")}</div>
        <h3>How it works</h3>
        <p class="meta">Each job is placed in the group whose example tasks are closest in meaning to its own
        (compared with a language model). Inside each group, similar jobs are clustered automatically.
        Dashed bubbles hold <em>unique roles</em> that didn't fit neatly into a cluster.</p>`;
      panel.querySelectorAll(".row").forEach(b => b.addEventListener("click", () => zoom(groupsInOrder[+b.dataset.i])));
      return;
    }
    const kids = d.children;
    const subclusters = kids.filter(k => !isJob(k)).length;
    panel.style.setProperty("--swatch", colorOf(d));
    panel.innerHTML = `
      <p class="eyebrow"><span class="dot"></span>${pathOf(d) || "Career group"}</p>
      <h2>${esc(d.data.name)}</h2>
      ${d.data.tagline ? `<p>${esc(d.data.tagline)}</p>` : ""}
      ${d.data.niche ? `<p><span class="badge">Unique roles</span></p><p class="meta">These jobs didn't cluster closely with others in this group.</p>` : ""}
      <div class="stats">
        <div class="stat"><b>${countJobs(d)}</b><span>jobs</span></div>
        ${subclusters ? `<div class="stat"><b>${subclusters}</b><span>clusters</span></div>` : ""}
      </div>
      ${d.data.keywords?.length && !d.data.niche ? `<h3>Key terms</h3><div class="tags">${d.data.keywords.map(k => `<span class="tag">${esc(k)}</span>`).join("")}</div>` : ""}
      <h3>${kids.every(isJob) ? "Jobs" : "Inside"}</h3>
      <div class="list">${kids.map(row).join("")}</div>`;
    panel.querySelectorAll(".row").forEach(b => b.addEventListener("click", () => {
      const k = kids[+b.dataset.i];
      isJob(k) ? showJob(k) : zoom(k);
    }));
    panel.scrollTop = 0;
  }

  function showJob(d) {
    select(d);
    const j = d.data;
    const siblings = d.parent.children.filter(s => s !== d && isJob(s)).slice(0, 8);
    panel.style.setProperty("--swatch", colorOf(d));
    panel.innerHTML = `
      <p class="eyebrow"><span class="dot"></span>${pathOf(d)}</p>
      <h2>${esc(j.name)}</h2>
      ${j.niche ? `<p><span class="badge">Unique role</span></p>` : ""}
      <p>${esc(j.description)}</p>
      <div class="stats">
        ${j.job_zone ? `<div class="stat"><b>Zone ${j.job_zone}</b><span>${esc(JOB_ZONES[j.job_zone])}</span></div>` : ""}
        ${j.interest ? `<div class="stat"><b>${esc(j.interest)}</b><span>${esc(INTERESTS[j.interest] || "")}</span></div>` : ""}
      </div>
      ${j.tasks?.length ? `<h3>Typical tasks</h3><ul>${j.tasks.map(t => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}
      <a class="link-out" href="${esc(j.url)}" target="_blank" rel="noopener">Full profile on O*NET OnLine ↗</a>
      ${siblings.length ? `<h3>Similar jobs</h3><div class="list">${siblings.map(row).join("")}</div>` : ""}
      <button class="back" id="back">← Back to ${esc(d.parent.data.name)}</button>
      ${lastSearch ? `<br><button class="back" id="back-results">← Back to results for “${esc(lastSearch.query)}”</button>` : ""}`;
    panel.querySelectorAll(".row").forEach(b => b.addEventListener("click", () => showJob(siblings[+b.dataset.i])));
    panel.querySelector("#back").addEventListener("click", () => showCluster(d.parent));
    panel.querySelector("#back-results")?.addEventListener("click", () => zoom(root, null, showResults));
    panel.scrollTop = 0;
  }

  // ---------- search & deep links ----------
  // Exact job titles open directly. Anything else goes to the semantic search API (src/serve.py),
  // which returns the jobs closest in meaning, or nothing if the query is out of scope.
  const search = document.getElementById("search");
  const jobsByCode = new Map(root.leaves().map(d => [d.data.code, d]));
  document.getElementById("job-list").innerHTML = root.leaves()
    .map(d => d.data.name).sort().map(n => `<option value="${esc(n)}">`).join("");

  const openJob = name => {
    const hit = jobsByName.get(name.trim().toLowerCase());
    if (hit) zoom(hit.parent, hit);
    return Boolean(hit);
  };

  const highlight = codes => {
    const set = new Set(codes);
    node.classed("match", d => isJob(d) && set.has(d.data.code));
    svg.classed("searching", set.size > 0);
  };
  const clearSearch = () => { lastSearch = null; highlight([]); };

  // In-browser semantic search, used when the Python search API isn't running (e.g. on GitHub Pages).
  // Loads a small sentence model (all-MiniLM-L6-v2, ~23 MB, cached by the browser after the first time)
  // and compares the query with job vectors exported by src/export_search_index.py.
  const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
  let browserEngine = null; // Promise of { meta, vectors, extractor }

  function loadBrowserEngine() {
    browserEngine ??= (async () => {
      const [meta, buffer] = await Promise.all([
        fetch("data/search_index.json").then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
        fetch("data/search_index.bin").then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
      ]);
      const { pipeline } = await import(TRANSFORMERS_URL);
      const extractor = await pipeline("feature-extraction", meta.model, { dtype: "q8" });
      return { meta, vectors: new Int8Array(buffer), extractor };
    })().catch(err => { browserEngine = null; throw err; }); // allow a retry later
    return browserEngine;
  }

  async function browserSearch(query) {
    const { meta, vectors, extractor } = await loadBrowserEngine();
    const { dim, count, scale, profileWeight, minScore, codes } = meta;
    const q = (await extractor(query, { pooling: "mean", normalize: true })).data;
    const scored = [];
    for (let i = 0; i < count; i++) {
      // Same blend as src/search.py: full job profile + title
      let profile = 0, title = 0;
      const p = i * dim, t = (count + i) * dim;
      for (let j = 0; j < dim; j++) { profile += vectors[p + j] * q[j]; title += vectors[t + j] * q[j]; }
      const score = (profileWeight * profile + (1 - profileWeight) * title) / scale;
      if (score >= minScore) scored.push({ code: codes[i], score });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, 10);
  }

  // Last resort if neither semantic search is available: match words in job titles
  function keywordSearch(query) {
    const words = query.toLowerCase().split(/\W+/).filter(w => w.length > 2);
    if (!words.length) return [];
    return root.leaves()
      .map(d => ({ d, hits: words.filter(w => d.data.name.toLowerCase().includes(w)).length }))
      .filter(r => r.hits > 0)
      .sort((a, b) => b.hits - a.hits)
      .slice(0, 10)
      .map(r => ({ code: r.d.data.code, score: null }));
  }

  let searchSeq = 0;
  let apiAvailable = null; // null = not tried yet; false once the Python API has failed

  function showSearchLoading(query) {
    select(null);
    panel.style.removeProperty("--swatch");
    panel.innerHTML = `
      <p class="eyebrow">Search</p>
      <h2>Searching for “${esc(query)}”…</h2>
      <p class="meta">Loading the search model (about 23 MB, first search only). Later searches are instant.</p>`;
  }

  async function runSearch(query) {
    query = query.trim();
    if (!query) { clearSearch(); if (focus === root) showCluster(root); return; }
    if (openJob(query)) return;
    const seq = ++searchSeq;
    let results = null;
    let semantic = true;
    if (apiAvailable !== false) {
      try {
        const r = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        if (!r.ok) throw new Error(r.status);
        results = (await r.json()).results;
        apiAvailable = true;
      } catch {
        apiAvailable = false;
      }
    }
    if (results === null) {
      try {
        if (!browserEngine && seq === searchSeq) showSearchLoading(query);
        results = await browserSearch(query);
      } catch (err) {
        console.warn("In-browser search unavailable, matching titles instead:", err);
        semantic = false;
        results = keywordSearch(query);
      }
    }
    if (seq !== searchSeq) return; // a newer search started while this one was running
    const hits = results.map(r => ({ ...r, d: jobsByCode.get(r.code) })).filter(r => r.d);
    lastSearch = { query, hits, semantic };
    highlight(hits.map(h => h.d.data.code));
    if (focus !== root) zoom(root, null, showResults);
    else showResults();
  }

  // Short queries score lower than long ones, so these bands are set from real searches (see src/search.py)
  const strength = s => (s == null ? "" : s >= 0.45 ? "Strong match" : s >= 0.34 ? "Good match" : "Possible match");

  function showResults() {
    if (!lastSearch) return;
    select(null);
    const { query, hits, semantic } = lastSearch;
    panel.style.removeProperty("--swatch");
    if (!hits.length) {
      panel.innerHTML = `
        <p class="eyebrow">Search</p>
        <h2>No results</h2>
        <p class="meta">Nothing is close enough to “${esc(query)}”. Try describing the kind of work, e.g.
        “working with animals”, “fixing cars” or “helping people”.</p>`;
      return;
    }
    panel.innerHTML = `
      <p class="eyebrow">${semantic ? "Closest matches" : "Title matches"}</p>
      <h2>Results for “${esc(query)}”</h2>
      <p class="meta">${hits.length} jobs${semantic ? ", ranked by how close their work is to your search" : ""}.
      Matches are outlined on the chart.</p>
      <div class="list" style="margin-top:12px">${hits.map((h, i) => `
        <button class="row" data-i="${i}" style="--swatch:${colorOf(h.d)}">
          <span class="name"><span class="dot"></span><span>${esc(h.d.data.name)}
            <small>${esc(groupOf(h.d).data.name)} · ${esc(h.d.parent.data.name)}</small></span></span>
          <span class="n">${esc(strength(h.score))}</span>
        </button>`).join("")}
      </div>`;
    panel.querySelectorAll(".row").forEach(b => b.addEventListener("click", () => {
      const d = hits[+b.dataset.i].d;
      zoom(d.parent, d);
    }));
    panel.scrollTop = 0;
  }

  let typingTimer;
  search.addEventListener("input", () => {
    clearTimeout(typingTimer);
    if (jobsByName.has(search.value.trim().toLowerCase())) return runSearch(search.value); // picked a suggestion
    typingTimer = setTimeout(() => {
      if (search.value.trim().length >= 3 || !search.value.trim()) runSearch(search.value);
    }, 450);
  });
  search.addEventListener("keydown", e => {
    if (e.key === "Enter") { clearTimeout(typingTimer); runSearch(search.value); }
    if (e.key === "Escape") { search.value = ""; runSearch(""); }
  });
  // index.html#Registered%20Nurses opens that job
  // index.html#search=I%20like%20animals runs a search
  const openFromHash = () => {
    const h = decodeURIComponent(location.hash.slice(1));
    if (h.startsWith("search=")) { search.value = h.slice(7); runSearch(search.value); }
    else openJob(h);
  };
  window.addEventListener("hashchange", openFromHash);

  // ---------- misc ----------
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && focus !== root && document.activeElement !== search) zoom(focus.parent);
  });
  window.addEventListener("resize", () => { setLabelText(view); zoomTo(view); });
  darkMode.addEventListener("change", () => { paint(); focus === root ? showCluster(root) : null; });

  paint();
  svg.classed("overview", true);
  zoomTo(viewFor(root));
  setLabelText(view);
  renderBreadcrumb();
  showCluster(root);
  openFromHash();
}
