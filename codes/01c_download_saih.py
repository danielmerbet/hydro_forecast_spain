#!/usr/bin/env python3
"""
01c — Recent / live daily streamflow from the SAIH of each basin authority.

WHY
  CEDEX's Anuario (script 01b) ends on 2022-09-30. Google's model can only be
  scored after 2023-09-30 (its training end), and forecasts need data up to
  yesterday. Each Confederación Hidrográfica runs its own SAIH (Sistema
  Automático de Información Hidrológica) with its own web service; this script
  has one ADAPTER per SAIH, all producing the same table.

JOIN WITH CEDEX
  SAIH stations are linked to CEDEX gauges through CEDEX's `cod_saih` field
  (e.g. CEDEX 9001 "Miranda de Ebro" = SAIH Ebro A001). Over the overlap
  (2021-10-01 .. 2022-09-30) correlation and volume ratio are checked, like the
  ACA export vs open-data check in script 01; gauges where the sources
  disagree are flagged and not used for scoring.

ADAPTERS (status)
  ebro   SAIH Ebro (CHE), https://www.saihebro.com — public JSON/ZIP API used by
         its "Datos históricos" page: getEstaciones / getSenales /
         obtenerDatosHistoricos (tipoConsolidado=diario, signal type QRIO =
         river discharge; daily MEAN is used). No login needed.
         TLS: the server omits its intermediate certificate (FNMT "AC
         Componentes Informáticos"); it is shipped in config/certs/ and added
         to the CA bundle, so certificates are still fully verified.
  jucar  SAIH Júcar (CHJ), https://saih.chj.es — gauge list embedded in the
         public map page; 5-min discharge from the endpoint its public chart
         page uses; daily means on local days (>= 75 % coverage).
  guadalquivir  SAIH Guadalquivir (CHG): the public ASP.NET "Datos Históricos"
         form (daily mean of the river-discharge signal of each unit).
  others: see docs/spain_extension_plan.md (Segura needs terms-of-use acceptance,
         Cantábrico's download page is obfuscated -> formal data request,
         Tajo uses per-session encrypted URLs).

Writes
  data/processed/spain/q_obs_daily_saih_<basin>.parquet   daily mean m3/s, columns = CEDEX gauge ids
  outputs/tables/qc_cedex_vs_saih_<basin>.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import subprocess
import tempfile
import zipfile

import certifi
import numpy as np
import pandas as pd

from hydrocat.config import CONFIG, PROCESSED, ROOT, TABLES

OUT = PROCESSED / "spain"


def log(*a):
    print(*a, flush=True)


def ca_bundle() -> str:
    """certifi's roots + intermediates that some SAIH servers fail to send."""
    f = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
    f.write(open(certifi.where()).read())
    for pem in sorted((CONFIG / "certs").glob("*.pem")):
        f.write("\n" + pem.read_text())
    f.close()
    return f.name


def curl(url: str, params: dict | None = None, cafile: str | None = None, timeout: int = 300) -> bytes:
    cmd = ["curl", "-sS", "--fail", "--retry", "4", "--max-time", str(timeout), "-A", "Mozilla/5.0 (hydro-forecast)",
           "-G", url]
    if cafile:
        cmd += ["--cacert", cafile]
    for k, v in (params or {}).items():
        cmd += ["--data-urlencode", f"{k}={v}"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


# ---------------------------------------------------------------------------
class SaihEbro:
    name = "ebro"
    api = "https://www.saihebro.com/api/datos-historicos"

    def __init__(self):
        self.ca = ca_bundle()

    def _json(self, ep, **p):
        return json.loads(curl(f"{self.api}/{ep}", p, self.ca))

    def signals(self, stations: list[str]) -> dict[str, int]:
        """SAIH station code (A001) -> id of its daily discharge signal."""
        out = {}
        for i in range(0, len(stations), 50):
            chunk = ",".join(stations[i:i + 50])
            for s in self._json("getSenales", tipoConsolidado="diario", tiposSenal="QRIO", estaciones=chunk):
                code = s["text"].split(" ")[0]
                out.setdefault(code, s["id"])
        return out

    def daily(self, sig: dict[str, int], start: dt.date, end: dt.date) -> pd.DataFrame:
        ids = list(sig.values())
        by_id = {v: k for k, v in sig.items()}
        frames = []
        for i in range(0, len(ids), 50):                 # the web form allows <= 100 signals
            raw = curl(f"{self.api}/obtenerDatosHistoricos", {
                "tipoConsolidado": "diario", "senalesSeleccionadas": ",".join(map(str, ids[i:i + 50])),
                "fechaIni": start.strftime("%d/%m/%Y 00:00"), "fechaFin": end.strftime("%d/%m/%Y 00:00"),
                "formato": "csv"}, self.ca, timeout=600)
            z = zipfile.ZipFile(io.BytesIO(raw))
            for n in z.namelist():
                code = n.split("_")[-1][:4]               # DatosHistoricos_<date>_A001L65QRIO1.csv
                d = pd.read_csv(z.open(n), sep=";", decimal=".", encoding="utf-8-sig")
                mean_col = [c for c in d.columns if c.startswith("MEDIA (")]
                if "FECHA_GRUPO" not in d.columns or not mean_col:
                    log(f"    skipped {n}: {list(d.columns)[:1]} ({len(d)} rows)")   # e.g. 'No hay datos'
                    continue
                unit = mean_col[0][len("MEDIA ("):-1]
                scale = {"m3/s": 1.0, "l/s": 1e-3}.get(unit)   # units are read from the header, never assumed
                if scale is None:
                    raise ValueError(f"{n}: unknown unit {unit!r}")
                d = pd.DataFrame({"date": pd.to_datetime(d["FECHA_GRUPO"], format="%d/%m/%Y"),
                                  "code": code, "q": pd.to_numeric(d[mean_col[0]], errors="coerce") * scale})
                frames.append(d)
            log(f"    {min(i + 50, len(ids))}/{len(ids)} signals")
        d = pd.concat(frames)
        assert set(d.code) <= set(sig), "unexpected station code in SAIH file names"
        return d.pivot_table(index="date", columns="code", values="q").sort_index()


class SaihJucar:
    """SAIH Júcar (CHJ), https://saih.chj.es — the public map page embeds the
    gauge list (`let aforos = [...]`: idVariable, station code fldTCodigo), and
    the public chart page reads 5-min values from
    /admin/variables/valor/{idVariable}/{start}/{end} (UTC timestamps, m3/s).
    Daily means are computed here on local (Europe/Madrid) days, keeping only
    days with >= 75 % of the 288 five-minute values."""
    name = "jucar"
    base = "https://saih.chj.es"

    def signals(self, stations: list[str]) -> dict[str, str]:
        html = curl(f"{self.base}/mapa-aforos").decode("utf-8")
        i = html.index("let aforos = ") + len("let aforos = ")
        arr, _ = json.JSONDecoder().raw_decode(html[i:])
        by_code = {norm_code(a["fldTCodigo"]): a["idVariable"] for a in arr}
        return {c: by_code[norm_code(c)] for c in stations if norm_code(c) in by_code}

    def _chunk(self, var: str, t0: dt.date, t1: dt.date) -> pd.DataFrame:
        url = f"{self.base}/admin/variables/valor/{var}/{t0:%Y-%m-%d} 00:00:00/{t1:%Y-%m-%d} 00:00:00"
        rows = json.loads(curl(url.replace(" ", "%20"), timeout=300))
        if not rows:
            return pd.DataFrame(columns=["t", "v"])
        d = pd.DataFrame(rows)
        return pd.DataFrame({"t": pd.to_datetime(d["fecha"], utc=True), "v": pd.to_numeric(d["valor"], errors="coerce")})

    def daily(self, sig: dict[str, str], start: dt.date, end: dt.date) -> pd.DataFrame:
        from concurrent.futures import ThreadPoolExecutor
        edges = list(pd.date_range(start, end, freq="90D").date) + [end + dt.timedelta(days=1)]
        jobs = [(code, var, a, b) for code, var in sig.items() for a, b in zip(edges[:-1], edges[1:])]
        with ThreadPoolExecutor(max_workers=4) as ex:
            parts = list(ex.map(lambda j: (j[0], self._chunk(j[1], j[2], j[3])), jobs))
        out = {}
        for code in sig:
            d = pd.concat([p for c, p in parts if c == code and len(p)])
            if d.empty:
                continue
            d = d.drop_duplicates("t").set_index("t")["v"].tz_convert("Europe/Madrid")
            g = d.groupby(d.index.date)
            day = g.mean()[g.count() >= 0.75 * 288]
            day.index = pd.to_datetime(day.index)
            out[code] = day
        log(f"    {len(out)}/{len(sig)} stations with data")
        return pd.DataFrame(out).sort_index()


class SaihGuadalquivir:
    """SAIH Guadalquivir (CHG), public ASP.NET form "Datos Históricos"
    (https://www.chguadalquivir.es/saih/DatosHistoricos.aspx; data labelled
    "no contrastados"). Per station: select the remote unit (postback), add its
    river-discharge signal ("CAUDAL RIO/ARROYO ...", daily mean), request a
    daily list (Periodo=2) and parse the result table. Pumping, releases and
    turbine flows at the same unit are ignored."""
    name = "guadalquivir"
    url = "https://www.chguadalquivir.es/saih/DatosHistoricos.aspx"
    P = "ctl00$ContentPlaceHolder1$"
    SKIP = ("BOMBEADO", "DESEMBALSADO", "TOMA", "TURBINADO", "CANAL")

    def __init__(self):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (hydro-forecast)"

    @staticmethod
    def _hidden(t):
        import html as _h
        import re
        f = {}
        for m in re.finditer(r"<input[^>]*>", t):
            tag = m.group(0)
            if 'type="hidden"' in tag:
                n, v = re.search(r'name="([^"]+)"', tag), re.search(r'value="([^"]*)"', tag)
                if n:
                    f[n.group(1)] = _h.unescape(v.group(1)) if v else ""
        return f

    @staticmethod
    def _options(t, name):
        import html as _h
        import re
        m = re.search(r'name="ctl00\$ContentPlaceHolder1\$' + name + r'".*?</select>', t, re.S)
        return [(v, _h.unescape(l).strip()) for v, l in re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)', m.group(0))] if m else []

    def signals(self, stations):
        t = self.s.get(self.url, timeout=120).text
        self._page = t
        rem = {l.split("_")[0]: v for v, l in self._options(t, "ListaRemotas")}
        return {c: rem[norm_code(c)] for c in stations if norm_code(c) in rem}

    def _station(self, rid, start, end):
        import re
        P = self.P
        base = {P + "Periodo": "2", P + "Formato": "lista", P + "ListaRemotas": rid}
        f = self._hidden(self._page) | base | {"__EVENTTARGET": P + "ListaRemotas"}
        t = self.s.post(self.url, data=f, timeout=120).text
        sig = [v for v, l in self._options(t, "ListaSen")
               if l.upper().startswith("CAUDAL") and not any(k in l.upper() for k in self.SKIP)]
        if not sig:
            return None
        f = self._hidden(t) | base | {P + "ListaSen": sig[0], P + "AgregarSeñal": "Agregar"}
        t = self.s.post(self.url, data=f, timeout=120).text
        chosen = [v for v, _ in self._options(t, "ListBoxSeñales")]
        out = []
        for a, b in zip(pd.date_range(start, end, freq="365D"),
                        list(pd.date_range(start, end, freq="365D"))[1:] + [pd.Timestamp(end)]):
            f = self._hidden(t) | base | {P + "ListBoxSeñales": chosen, P + "Visualizar": "Visualizar",
                                          P + "FechaInicial": a.strftime("%d/%m/%Y"),
                                          P + "FechaFinal": b.strftime("%d/%m/%Y")}
            r = self.s.post(self.url, data=f, timeout=300).text
            m = re.search(r'<table[^>]*id="ContentPlaceHolder1_TableDat".*?</table>', r, re.S)
            if not m:
                continue
            for d, v in re.findall(r"<t[hd][^>]*>\s*(\d\d/\d\d/\d\d)\s*</t[hd]>\s*<td[^>]*>\s*([-\d.,]+)\s*</td>", m.group(0)):
                out.append((pd.to_datetime(d, format="%d/%m/%y"), float(v.replace(".", "").replace(",", "."))))
        if not out:
            return None
        return pd.Series(dict(out)).sort_index()

    def daily(self, sig, start, end):
        out = {}
        for i, (code, rid) in enumerate(sig.items(), 1):
            try:
                q = self._station(rid, start, end)
            except Exception as e:                       # one station must not stop the rest
                log(f"    {code}: {e}")
                q = None
            if q is not None and len(q):
                out[code] = q
            if i % 10 == 0:
                log(f"    {i}/{len(sig)} stations")
        log(f"    {len(out)}/{len(sig)} stations with river discharge")
        return pd.DataFrame(out).sort_index()


def norm_code(c: str) -> str:
    """CEDEX writes some SAIH codes with a letter O where SAIH uses a zero (OA01 vs 0A01)."""
    c = str(c).strip().upper()
    return ("0" + c[1:]) if c.startswith("O") else c


ADAPTERS = {"ebro": (SaihEbro, "Ebro"), "jucar": (SaihJucar, "Jucar"),
            "guadalquivir": (SaihGuadalquivir, "Guadalquivir")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="ebro", choices=list(ADAPTERS))
    ap.add_argument("--start", default="2021-10-01", help="overlap with CEDEX starts 2021-10-01")
    args = ap.parse_args()
    cls, dem = ADAPTERS[args.basin]
    ad = cls()

    st = pd.read_csv(OUT / "stations_cedex.csv", dtype={"cod_saih": str})
    st = st[(st.demarcacion == dem) & st.cod_saih.notna() & (st.cod_saih.str.strip() != "")]
    log(f"{dem}: {len(st)} CEDEX gauges with a SAIH code")
    sig = ad.signals(sorted(st.cod_saih.unique()))
    log(f"   {len(sig)} have a daily discharge signal in SAIH {dem}")
    q = ad.daily(sig, dt.date.fromisoformat(args.start), dt.date.today())
    if q.empty:
        raise RuntimeError(f"no data retrieved from SAIH {dem}")

    code2gauge = dict(zip(st.cod_saih, st.gauge_id))
    code2gauge |= {norm_code(k): v for k, v in code2gauge.items()}
    q = q.rename(columns=code2gauge)
    q.index.name = "date"
    q = q.clip(lower=0)
    q.to_parquet(OUT / f"q_obs_daily_saih_{args.basin}.parquet")

    cedex = pd.read_parquet(OUT / "q_obs_daily_cedex.parquet")
    qc = []
    for g in q.columns:
        if g not in cedex:
            continue
        a, b = cedex[g].align(q[g], join="inner")
        m = a.notna() & b.notna()
        if m.sum() > 60:
            qc.append(dict(gauge_id=g, n_days=int(m.sum()), corr=float(np.corrcoef(a[m], b[m])[0, 1]),
                           ratio_mean=float(b[m].mean() / a[m].mean()) if a[m].mean() > 0 else np.nan))
    qc = pd.DataFrame(qc)
    qc.to_csv(TABLES / f"qc_cedex_vs_saih_{args.basin}.csv", index=False)
    log(f"-> {len(q.columns)} gauges, {q.index.min().date()} .. {q.index.max().date()}")
    if len(qc):
        ok = (qc["corr"] >= 0.9) & ((qc.ratio_mean - 1).abs() <= 0.25)
        log(f"   overlap with CEDEX: {len(qc)} gauges, median corr {qc['corr'].median():.3f}, "
            f"median volume ratio {qc.ratio_mean.median():.3f}; {int((~ok).sum())} disagree")


if __name__ == "__main__":
    main()
