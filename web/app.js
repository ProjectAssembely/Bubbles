// Bubbles: zoomable circle-packing view of the occupation hierarchy.
// Data comes from data/hierarchy.json, built by src/build_hierarchy.py.

const SIZE = 1000;            // layout coordinate space (the SVG viewBox is SIZE x SIZE)
const LABEL_PX = 13;          // on-screen label size in pixels
const PALETTE = ["#2f6fdf", "#e0702a", "#2a9d6f", "#c2417a", "#7a5af0", "#b8901c"];
const FILL_STRENGTH = [0, 0.16, 0.3, 0.44, 0.44]; // how strongly cluster circles are tinted, by depth

const JOB_ZONES = {
  1: "Little or no preparation needed",
  2: "Some preparation needed (e.g. high school)",
  3: "Medium preparation needed (e.g. vocational training)",
  4: "Considerable preparation needed (usually a bachelor's degree)",
  5: "Extensive preparation needed (usually graduate school)",
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
const darkMode = window.matchMedia("(prefers-color-scheme: dark)");

const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const countJobs = d => d.leaves().length;
const isJob = d => !d.children;

fetch("data/hierarchy.json")
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then(init)
  .catch(() => {
    panel.innerHTML = `<h2>Couldn't load the data</h2>
      <p>Open this page through a local web server so it can read <code>data/hierarchy.json</code>:</p>
      <pre>cd web\npython -m http.server 8000</pre><p>then visit <a href="http://localhost:8000">localhost:8000</a>.</p>`;
  });

function init(data) {
  const root = d3.pack()
    .size([SIZE, SIZE])
    .padding(d => (d.depth === 0 ? 14 : 3))(
      d3.hierarchy(data).sum(d => d.value || 0).sort((a, b) => b.value - a.value));

  root.children.forEach((group, i) => group.descendants().forEach(d => (d.color = PALETTE[i % PALETTE.length])));
  const jobsByName = new Map(root.leaves().map(d => [d.data.name.toLowerCase(), d]));

  const fill = d => {
    const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
    if (isJob(d)) return d3.interpolateRgb(bg, d.color)(0.8);
    return d3.interpolateRgb(bg, d.color)(FILL_STRENGTH[d.depth] ?? 0.44);
  };

  const svg = d3.select(chartEl).append("svg")
    .attr("viewBox", [-SIZE / 2, -SIZE / 2, SIZE, SIZE])
    .attr("preserveAspectRatio", "xMidYMid meet")
    .on("click", () => focus !== root && zoom(focus.parent));

  const node = svg.append("g").selectAll("circle")
    .data(root.descendants().slice(1))
    .join("circle")
    .attr("class", d => "node" + (d.data.niche && d.children ? " niche" : ""))
    .attr("fill", fill)
    .on("click", (event, d) => {
      event.stopPropagation();
      if (isJob(d)) {
        if (focus !== d.parent) zoom(d.parent, d);
        else showJob(d);
      } else if (focus !== d) {
        zoom(d);
      }
    });
  node.append("title").text(d => (isJob(d) ? d.data.name : `${d.data.name} (${countJobs(d)} jobs)`));

  const label = svg.append("g").attr("class", "labels").selectAll("text")
    .data(root.descendants())
    .join("text")
    .attr("class", d => (isJob(d) ? "job" : "cluster"))
    .style("display", d => (d.parent === root ? "inline" : "none"));

  let focus = root;
  let view;
  let selected = null;

  // --- layout helpers -------------------------------------------------------
  const unitsPerPx = () => SIZE / Math.max(1, Math.min(chartEl.clientWidth, chartEl.clientHeight));
  const viewFor = d => [d.x, d.y, d.r * 2 + (d === root ? 0 : 8)];

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
      const px = isJob(d) ? LABEL_PX - 1.5 : d.depth === 1 ? LABEL_PX * 1.2 : LABEL_PX;
      const fontUnits = px * unit;
      const diameter = d.r * k * 2 * 0.86;           // keep a little margin inside the circle
      const maxChars = Math.floor(diameter / (fontUnits * 0.55));
      const maxLines = Math.max(1, Math.min(3, Math.floor(diameter / (fontUnits * 1.2))));
      const lines = maxChars < 4 ? [] : wrap(d.data.name, maxChars, maxLines);
      const text = d3.select(this).attr("font-size", fontUnits);
      text.selectAll("tspan").data(lines).join("tspan")
        .attr("x", 0)
        .attr("dy", (_, i) => (i === 0 ? `${0.35 - (lines.length - 1) * 0.6}em` : "1.2em"))
        .text(l => l);
    });
  }

  function zoomTo(v) {
    const k = SIZE / v[2];
    view = v;
    label.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
    node.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`).attr("r", d => d.r * k);
  }

  function zoom(d, jobToShow = null) {
    focus = d;
    const target = viewFor(d);
    setLabelText(target);
    const t = svg.transition().duration(650).tween("zoom", () => {
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
    if (jobToShow) showJob(jobToShow);
    else showCluster(d);
  }

  // --- side panel ------------------------------------------------------------
  function select(d) {
    selected = d;
    node.classed("selected", n => n === d);
  }

  function pathOf(d) {
    return d.ancestors().reverse().slice(1, -1).map(a => esc(a.data.name)).join(" › ");
  }

  function showCluster(d) {
    select(null);
    if (d === root) {
      panel.innerHTML = `
        <h2>What kind of work appeals to you?</h2>
        <p class="meta">Pick a starting point, or click any bubble.</p>
        <ul class="choices">${root.children.map((g, i) => `
          <li><button data-i="${i}" style="--swatch:${g.color}">
            <strong>${esc(g.data.name)}</strong><small>${esc(g.data.tagline)} · ${countJobs(g)} jobs</small>
          </button></li>`).join("")}
        </ul>
        <h3>How this works</h3>
        <p class="meta">Jobs were grouped automatically: by what their tasks say (language model embeddings)
        and by the skills, knowledge and abilities they need. Bubbles with a dashed outline hold
        <em>unique roles</em> that didn't fit neatly into any group.</p>`;
      panel.querySelectorAll("button[data-i]").forEach(b =>
        b.addEventListener("click", () => zoom(root.children[+b.dataset.i])));
      return;
    }
    const kids = d.children;
    const jobKids = kids.every(isJob);
    panel.innerHTML = `
      <p class="meta">${pathOf(d) || "&nbsp;"}</p>
      <h2>${esc(d.data.name)}</h2>
      ${d.data.tagline ? `<p>${esc(d.data.tagline)}</p>` : ""}
      ${d.data.niche ? `<p><span class="badge niche">Unique roles</span> These jobs didn't cluster closely with others in this group.</p>` : ""}
      <p class="meta">${countJobs(d)} jobs</p>
      ${d.data.keywords?.length && !d.data.niche ? `<h3>Key terms</h3><div class="chips">${d.data.keywords.map(k => `<span class="chip">${esc(k)}</span>`).join("")}</div>` : ""}
      <h3>${jobKids ? "Jobs" : "Inside this group"}</h3>
      <div>${kids.map((k, i) => `<button class="job-link" data-i="${i}" style="--swatch:${k.color}">
        ${esc(k.data.name)}${isJob(k) ? "" : ` <span class="meta">· ${countJobs(k)}</span>`}</button>`).join("")}</div>`;
    panel.querySelectorAll("button[data-i]").forEach(b => b.addEventListener("click", () => {
      const k = kids[+b.dataset.i];
      isJob(k) ? showJob(k) : zoom(k);
    }));
  }

  function showJob(d) {
    select(d);
    const j = d.data;
    const siblings = d.parent.children.filter(s => s !== d && isJob(s)).slice(0, 8);
    panel.innerHTML = `
      <p class="meta">${pathOf(d)}</p>
      <h2>${esc(j.name)}</h2>
      ${j.niche ? `<p><span class="badge niche">Unique role</span></p>` : ""}
      <p>${esc(j.description)}</p>
      ${j.tasks?.length ? `<h3>Typical tasks</h3><ul>${j.tasks.map(t => `<li>${esc(t)}</li>`).join("")}</ul>` : ""}
      ${j.job_zone ? `<h3>Preparation</h3><p>${esc(JOB_ZONES[j.job_zone])}</p>` : ""}
      ${j.interest ? `<h3>Main interest</h3><p><strong>${esc(j.interest)}</strong>: ${esc(INTERESTS[j.interest] || "")}</p>` : ""}
      <p><a href="${esc(j.url)}" target="_blank" rel="noopener">Full profile on O*NET OnLine ↗</a></p>
      ${siblings.length ? `<h3>Similar jobs</h3><div>${siblings.map((s, i) =>
        `<button class="job-link" data-i="${i}" style="--swatch:${s.color}">${esc(s.data.name)}</button>`).join("")}</div>` : ""}
      <p><button class="job-link" id="back">← Back to ${esc(d.parent.data.name)}</button></p>`;
    panel.querySelectorAll("button[data-i]").forEach(b =>
      b.addEventListener("click", () => showJob(siblings[+b.dataset.i])));
    panel.querySelector("#back").addEventListener("click", () => showCluster(d.parent));
    panel.scrollTop = 0;
  }

  function renderBreadcrumb() {
    const trail = focus.ancestors().reverse();
    crumbs.innerHTML = trail.map((a, i) => i === trail.length - 1
      ? `<span>${esc(a === root ? "All careers" : a.data.name)}</span>`
      : `<button data-i="${i}">${esc(a === root ? "All careers" : a.data.name)}</button><span class="sep">›</span>`).join("");
    crumbs.querySelectorAll("button").forEach(b => b.addEventListener("click", () => zoom(trail[+b.dataset.i])));
  }

  // --- search ----------------------------------------------------------------
  const search = document.getElementById("search");
  document.getElementById("job-list").innerHTML = root.leaves()
    .map(d => d.data.name).sort().map(n => `<option value="${esc(n)}">`).join("");
  const runSearch = () => {
    const hit = jobsByName.get(search.value.trim().toLowerCase());
    if (hit) zoom(hit.parent, hit);
  };
  search.addEventListener("change", runSearch);
  search.addEventListener("keydown", e => e.key === "Enter" && runSearch());

  // --- misc ------------------------------------------------------------------
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && focus !== root && document.activeElement !== search) zoom(focus.parent);
  });
  window.addEventListener("resize", () => { setLabelText(view); zoomTo(view); });
  darkMode.addEventListener("change", () => node.attr("fill", fill));

  // Deep links: index.html#Registered%20Nurses opens that job
  const openFromHash = () => {
    const hit = jobsByName.get(decodeURIComponent(location.hash.slice(1)).toLowerCase());
    if (hit) zoom(hit.parent, hit);
  };
  window.addEventListener("hashchange", openFromHash);

  zoomTo(viewFor(root));
  setLabelText(view);
  renderBreadcrumb();
  showCluster(root);
  openFromHash();
}
