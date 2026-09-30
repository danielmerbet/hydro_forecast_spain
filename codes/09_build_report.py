#!/usr/bin/env python3
"""
09 — Build the phase-1 results page (static HTML + images) from outputs/.

Everything on the page is read from files produced by scripts 01-08, so the
page is rebuilt identically whenever the pipeline is re-run.

Writes
  outputs/report/index.html
  outputs/report/img/*.webp     figures re-encoded as WebP (smaller than PNG)
"""
from __future__ import annotations

import html
import json

import pandas as pd
from PIL import Image

from hydrocat.config import FIGURES, OUTPUTS, TABLES, P, load_settings

S = load_settings()
OUT = OUTPUTS / "report"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

LABEL = {"gr4j": "GR4J", "google_baseline": "Google, as released",
         "google_finetuned": "Google, fine-tuned", "gr4j_emo1": "GR4J, EMO-1 rain"}
ORDER = ["gr4j", "google_baseline", "google_finetuned", "gr4j_emo1"]


def webp(src, name, width=1600):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(IMG / name, "WEBP", quality=82, method=6)
    return f"img/{name}"


def fmt(v, pct=False):
    if pd.isna(v):
        return "–"
    return f"{v:+.0f} %" if pct else f"{v:.2f}"


def main():
    summ = pd.read_csv(TABLES / "metrics_summary.csv")
    met = pd.read_csv(P.metrics)
    st = pd.read_csv(P.stations).set_index("gauge_id")
    pr = pd.read_csv(TABLES / "03c_precip_era5land_vs_emo1.csv")
    t0, t1 = S["periods"]["test_start"], met.n_days.max()

    figs = {k: webp(FIGURES / f"{k}.png", f"{k}.webp") for k in [
        "02_catchments_map", "02_gauges_spain", "03c_precip_era5land_vs_emo1", "05b_gr4j_kge_by_gauge",
        "08_skill_cdf", "08_gr4j_vs_google_scatter", "08_skill_map"]}
    gauges = []
    for gid in sorted(met.gauge_id.unique()):
        m = met[met.gauge_id == gid].set_index("model")
        gauges.append(dict(
            id=gid, name=str(st.loc[gid, "name"]), river=str(st.loc[gid, "river"]),
            regulated=bool(m.regulated.iloc[0]), area=round(float(m.area_km2.iloc[0])),
            cmp=webp(FIGURES / "gauges" / f"{gid}.png", f"cmp_{gid}.webp", 1400),
            cal=webp(FIGURES / "gr4j" / f"{gid}.png", f"cal_{gid}.webp", 1400),
            kge={k: (None if pd.isna(m.KGE.get(k)) else round(float(m.KGE[k]), 2)) for k in ORDER if k in m.index}))

    def table(reg):
        rows = []
        for k in ORDER:
            r = summ[(summ.regulated == reg) & (summ.model == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f"<tr><th scope='row'><span class='sw sw-{k}'></span>{LABEL[k]}</th>"
                        f"<td>{fmt(r.KGE)}</td><td>{fmt(r.NSE)}</td><td>{fmt(r.logNSE)}</td>"
                        f"<td>{fmt(r.PBIAS, True)}</td><td>{fmt(r.peak_err_pct, True)}</td></tr>")
        return "\n".join(rows)

    w = met.pivot(index="gauge_id", columns="model", values="KGE")
    better_ft = int((w.google_finetuned > w.gr4j).sum()) if "google_finetuned" in w else 0
    page = TEMPLATE.format(
        t0=t0, n=len(gauges), better_ft=better_ft, n_nat=int((~met.drop_duplicates("gauge_id").regulated).sum()),
        tab_nat=table(False), tab_reg=table(True),
        p_ratio=f"{pr.ratio_era5l_emo1.median():.2f}", p_max=f"{pr.ratio_era5l_emo1.max():.2f}",
        imp_e=int(pr.impossible_era5l.sum()), imp_m=int(pr.impossible_emo1.sum()),
        gauges_json=json.dumps(gauges, ensure_ascii=False), **figs)
    (OUT / "index.html").write_text(page)
    print(f"-> {OUT / 'index.html'} ({len(gauges)} gauges, {len(list(IMG.iterdir()))} images)")


TEMPLATE = """<title>Catalan River Models</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single reading column (report), with a wide figure band and a gauge explorer. */
:root {{
  --paper: #f6f8f9; --ink: #13202b; --ink-2: #4a5a67; --rule: #d6dee4; --panel: #ffffff;
  --accent: #1e6aa8; --figbg: #ffffff;
  --m-gr4j: #2a78d6; --m-google_baseline: #eb6834; --m-google_finetuned: #1baf7a; --m-gr4j_emo1: #eda100;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #0f171d; --ink: #e6edf2; --ink-2: #9fb0bd; --rule: #26343f; --panel: #16212a; --accent: #6aaee8;
  --figbg: #f4f6f7; --m-gr4j: #3987e5; --m-google_baseline: #d95926; --m-google_finetuned: #199e70; --m-gr4j_emo1: #c98500;
  color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #0f171d; --ink: #e6edf2; --ink-2: #9fb0bd; --rule: #26343f; --panel: #16212a; --accent: #6aaee8;
  --figbg: #f4f6f7; --m-gr4j: #3987e5; --m-google_baseline: #d95926; --m-google_finetuned: #199e70; --m-gr4j_emo1: #c98500;
  color-scheme: dark; }}
body {{ background: var(--paper); color: var(--ink); font: 15px/1.6 var(--body); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 80px; display: grid; gap: 48px; }}
header {{ display: grid; gap: 10px; max-width: 70ch; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-2); }}
h1 {{ font: 650 clamp(30px, 5vw, 46px)/1.08 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 650 24px/1.2 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 15px/1.3 var(--body); margin: 0; }}
p {{ margin: 0; max-width: 68ch; }}
.lede {{ font-size: 17px; color: var(--ink-2); }}
section {{ display: grid; gap: 18px; }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.fact {{ border-top: 2px solid var(--ink); padding-top: 10px; display: grid; gap: 4px; }}
.fact b {{ font: 650 26px/1.1 var(--display); font-variant-numeric: tabular-nums; }}
.fact span {{ color: var(--ink-2); font-size: 13.5px; }}
.tables {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }}
.tscroll {{ overflow-x: auto; min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px; }}
caption {{ text-align: left; font: 600 13px/1.4 var(--body); color: var(--ink-2); padding-bottom: 6px; }}
th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
thead th {{ font: 500 11.5px/1.2 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--ink-2); }}
tbody th {{ text-align: left; font-weight: 500; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 8px; vertical-align: -1px; }}
.sw-gr4j {{ background: var(--m-gr4j); }} .sw-google_baseline {{ background: var(--m-google_baseline); }}
.sw-google_finetuned {{ background: var(--m-google_finetuned); }} .sw-gr4j_emo1 {{ background: var(--m-gr4j_emo1); }}
.note {{ font-size: 13px; color: var(--ink-2); }}
ul.findings {{ margin: 0; padding-left: 1.1em; display: grid; gap: 8px; max-width: 72ch; }}
figure {{ margin: 0; display: grid; gap: 8px; min-width: 0; }}
figure img {{ background: var(--figbg); border: 1px solid var(--rule); border-radius: 4px; width: 100%; height: auto; }}
figcaption {{ font-size: 13.5px; color: var(--ink-2); max-width: 75ch; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
.explorer {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px; display: grid; gap: 14px; }}
.pick {{ display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }}
select {{ font: 15px var(--body); padding: 6px 10px; border: 1px solid var(--rule); border-radius: 4px; background: var(--paper); color: var(--ink); max-width: 100%; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ font: 500 12.5px/1 var(--mono); padding: 6px 8px; border: 1px solid var(--rule); border-radius: 3px; font-variant-numeric: tabular-nums; }}
.tabs {{ display: flex; gap: 4px; }}
.tabs button {{ font: 500 13px var(--body); padding: 6px 12px; border: 1px solid var(--rule); background: transparent; color: var(--ink); border-radius: 3px; cursor: pointer; }}
.tabs button[aria-selected="true"] {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
button:focus-visible, select:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
dl.meth {{ display: grid; grid-template-columns: max-content 1fr; gap: 8px 18px; margin: 0; max-width: 90ch; }}
dl.meth dt {{ font-weight: 600; }} dl.meth dd {{ margin: 0; color: var(--ink-2); }}
@media (max-width: 560px) {{ dl.meth {{ grid-template-columns: 1fr; }} .tables {{ grid-template-columns: 1fr; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Phase 1 · calibration and model comparison · ACA gauges</div>
  <h1>Catalan river models: calibrated GR4J against Google's hydrology model</h1>
  <p class="lede">{n} gauges of Catalonia's internal basins, scored from {t0} to today — after the end of Google's training data — with every model driven by the same ERA5-Land weather.</p>
</header>

<section>
  <div class="facts">
    <div class="fact"><b>{n}</b><span>ACA gauges modelled ({n_nat} near-natural)</span></div>
    <div class="fact"><b>{better_ft} / {n}</b><span>gauges where fine-tuned Google beats GR4J (KGE)</span></div>
    <div class="fact"><b>×{p_ratio}</b><span>ERA5-Land rain vs gauge-based EMO-1, median catchment (up to ×{p_max})</span></div>
    <div class="fact"><b>{imp_e} → {imp_m}</b><span>catchments with an impossible water balance, ERA5-Land → EMO-1</span></div>
  </div>
</section>

<section>
  <h2>Test-period skill</h2>
  <div class="tables">
    <div class="tscroll"><table>
      <caption>Near-natural gauges — medians</caption>
      <thead><tr><th>Model</th><th>KGE</th><th>NSE</th><th>log NSE</th><th>Volume bias</th><th>Top 1 % flows</th></tr></thead>
      <tbody>{tab_nat}</tbody></table></div>
    <div class="tscroll"><table>
      <caption>Regulated gauges (reservoir upstream) — medians</caption>
      <thead><tr><th>Model</th><th>KGE</th><th>NSE</th><th>log NSE</th><th>Volume bias</th><th>Top 1 % flows</th></tr></thead>
      <tbody>{tab_reg}</tbody></table></div>
  </div>
  <p class="note">KGE, NSE: 1 is perfect, 0 is no better than the mean. log NSE weights low flows. Volume bias: simulated minus observed total. Top 1 %: error on the days with the highest observed flows. Only gauges whose delineated catchment is within 20 % of ACA's official area.</p>
  <ul class="findings">
    <li><b>Google's model as released gets flood timing and shape right but roughly doubles the water</b> in Catalan rivers. A global model cannot know that these catchments lose so much of their rain.</li>
    <li><b>Fine-tuned on ACA data (2008–2021), it has the best low flows and volumes of all models</b>, and is clearly the best below dams, where it has learned how regulated rivers behave. It under-predicts the largest peaks.</li>
    <li><b>Calibrated GR4J has the best median KGE at near-natural gauges</b> but its recessions drain too fast in summer.</li>
    <li><b>ERA5-Land's rain is too high here</b> (median +29 % against EMO-1). With EMO-1-scaled rain, GR4J's water balance becomes physically consistent; that run is shown as a sensitivity test.</li>
    <li><b>No model catches the January 2026 Onyar flood</b> (258 m³/s observed): ~9 km daily reanalysis rain cannot resolve that storm. Forecast skill for floods will depend on the rain forecast as much as on the hydrological model.</li>
  </ul>
</section>

<section>
  <h2>Skill across gauges</h2>
  <figure><img src="{08_skill_cdf}" alt="Cumulative distributions of KGE and NSE across gauges for each model" loading="lazy">
    <figcaption>Share of gauges at or below each score. Curves further right are better.</figcaption></figure>
  <figure><img src="{08_gr4j_vs_google_scatter}" alt="Per-gauge KGE of GR4J against each Google model" loading="lazy">
    <figcaption>Each dot is a gauge; above the diagonal the Google model is better. Squares are regulated rivers.</figcaption></figure>
  <div class="grid2">
    <figure><img src="{08_skill_map}" alt="Map of KGE difference between fine-tuned Google and GR4J" loading="lazy">
      <figcaption>Where each model does better: red dots favour fine-tuned Google, blue dots favour GR4J.</figcaption></figure>
    <figure><img src="{05b_gr4j_kge_by_gauge}" alt="GR4J KGE in calibration and test period per gauge" loading="lazy">
      <figcaption>GR4J at every gauge: calibration (filled) against test period (open).</figcaption></figure>
  </div>
</section>

<section>
  <h2>Gauge explorer</h2>
  <div class="explorer">
    <div class="pick">
      <label for="gauge">Gauge</label>
      <select id="gauge"></select>
      <div class="tabs" role="tablist">
        <button id="tab-cmp" role="tab" aria-selected="true">Model comparison (test)</button>
        <button id="tab-cal" role="tab" aria-selected="false">GR4J calibration (EMO-1-scaled rain)</button>
      </div>
    </div>
    <div class="chips" id="chips"></div>
    <figure><img id="gimg" src="" alt=""></figure>
  </div>
</section>

<section>
  <h2>Why the rain matters</h2>
  <figure><img src="{03c_precip_era5land_vs_emo1}" alt="ERA5-Land against EMO-1 precipitation per catchment, and the water balance each implies" loading="lazy">
    <figcaption>Left: mean annual rain per catchment, 2008–2023. Right: rain minus observed runoff, the water that must leave other than as river flow; with ERA5-Land it exceeds the FAO-56 reference evaporation (black ticks) in {imp_e} catchments, with EMO-1 in {imp_m}.</figcaption></figure>
</section>

<section>
  <h2>Gauges and catchments</h2>
  <div class="grid2">
    <figure><img src="{02_catchments_map}" alt="Delineated catchments of the ACA gauges" loading="lazy">
      <figcaption>ACA gauges and catchments from HydroSHEDS 90 m flow directions, checked against ACA's official drained areas (red: off by more than 20 %).</figcaption></figure>
    <figure><img src="{02_gauges_spain}" alt="All Spanish river gauges with daily discharge data" loading="lazy">
      <figcaption>Next step: all 1,238 Spanish gauges with daily data. CEDEX history to 2022; recent data from each basin authority's SAIH (Ebro and Júcar connected so far).</figcaption></figure>
  </div>
</section>

<section>
  <h2>Method in brief</h2>
  <dl class="meth">
    <dt>Observations</dt><dd>ACA daily discharge: 2007–2024 export plus the Catalan open-data feed (5-min, corrected for a litres-per-second unit error in part of its record), quality-controlled for infilled runs and sensor spikes.</dd>
    <dt>Weather</dt><dd>ERA5-Land catchment means from Earth Engine; FAO-56 Penman-Monteith evaporation from ERA5-Land radiation, humidity, pressure and wind.</dd>
    <dt>GR4J</dt><dd>GR4J + CemaNeige (port of INRAE's airGR, verified to 1e-6), 5-year spin-up, calibrated 2008–2023 on the mean of KGE(Q) and KGE(√Q); zero-flow threshold at intermittent near-natural rivers.</dd>
    <dt>Google</dt><dd>google-research/flood-forecasting (pinned commit): pretrained weights as released, and the same model fine-tuned on ACA gauges (train 2008–2021, pick epoch on 2021–2023).</dd>
    <dt>Test period</dt><dd>From {t0}: after the end of Google's training data (2023-09-30), so no model has seen it.</dd>
  </dl>
</section>
</div>

<script>
const GAUGES = {gauges_json};
const LABEL = {{"gr4j":"GR4J","google_baseline":"Google released","google_finetuned":"Google fine-tuned","gr4j_emo1":"GR4J EMO-1 rain"}};
const sel = document.getElementById("gauge"), img = document.getElementById("gimg"), chips = document.getElementById("chips");
const tCmp = document.getElementById("tab-cmp"), tCal = document.getElementById("tab-cal");
let view = "cmp";
for (const g of GAUGES) {{
  const o = document.createElement("option"); o.value = g.id;
  o.textContent = `${{g.id}} — ${{g.name}} (${{g.river}})${{g.regulated ? " · regulated" : ""}}`;
  sel.appendChild(o);
}}
function show() {{
  const g = GAUGES.find(x => x.id === sel.value) || GAUGES[0];
  img.src = view === "cmp" ? g.cmp : g.cal;
  img.alt = (view === "cmp" ? "Model comparison" : "GR4J calibration") + " for " + g.name;
  chips.replaceChildren(...Object.entries(g.kge).map(([k, v]) => {{
    const s = document.createElement("span"); s.className = "chip";
    s.innerHTML = `<span class="sw sw-${{k}}"></span>${{LABEL[k]}} KGE ${{v === null ? "–" : v.toFixed(2)}}`; return s; }}));
  tCmp.setAttribute("aria-selected", view === "cmp"); tCal.setAttribute("aria-selected", view === "cal");
  try {{ localStorage.setItem("gauge", sel.value); }} catch (e) {{}}
}}
let start = "EA020"; try {{ start = localStorage.getItem("gauge") || start; }} catch (e) {{}}
if (GAUGES.some(g => g.id === start)) sel.value = start;
sel.addEventListener("change", show);
tCmp.addEventListener("click", () => {{ view = "cmp"; show(); }});
tCal.addEventListener("click", () => {{ view = "cal"; show(); }});
show();
</script>
"""

if __name__ == "__main__":
    main()
