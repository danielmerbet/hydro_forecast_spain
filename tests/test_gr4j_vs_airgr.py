"""Check that hydrocat.gr4j reproduces airGR (the INRAE reference implementation).

Uses airGR's own example catchment (L0123001, "Blue River at Nourlangie Rock"
- daily P, PE, T, hypsometry). Skipped when R or airGR is not installed.

    pytest tests/test_gr4j_vs_airgr.py -v
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydrocat.gr4j import cemaneige, gr4j

R_SCRIPT = r"""
suppressMessages(library(airGR))
data(L0123002)   # has temperature + hypsometry -> CemaNeige possible
out <- commandArgs(trailingOnly = TRUE)[1]
Ind <- seq(which(format(BasinObs$DatesR, "%Y-%m-%d") == "1990-01-01"),
           which(format(BasinObs$DatesR, "%Y-%m-%d") == "1999-12-31"))
# --- GR4J alone
im <- CreateInputsModel(RunModel_GR4J, DatesR = BasinObs$DatesR, Precip = BasinObs$P, PotEvap = BasinObs$E)
ro <- suppressWarnings(CreateRunOptions(RunModel_GR4J, InputsModel = im, IndPeriod_Run = Ind, IndPeriod_WarmUp = 0L))
par <- c(X1 = 257.24, X2 = 1.012, X3 = 88.23, X4 = 2.208)
q <- RunModel_GR4J(im, ro, par)$Qsim
# --- CemaNeige + GR4J, 5 layers
im2 <- CreateInputsModel(RunModel_CemaNeigeGR4J, DatesR = BasinObs$DatesR, Precip = BasinObs$P,
                         PotEvap = BasinObs$E, TempMean = BasinObs$T, ZInputs = median(BasinInfo$HypsoData),
                         HypsoData = BasinInfo$HypsoData, NLayers = 5)
ro2 <- suppressWarnings(CreateRunOptions(RunModel_CemaNeigeGR4J, InputsModel = im2, IndPeriod_Run = Ind, IndPeriod_WarmUp = 0L))
par2 <- c(par, CTG = 0.35, Kf = 4.2)
r2 <- RunModel_CemaNeigeGR4J(im2, ro2, par2)
L <- function(x) do.call(cbind, lapply(x, function(v) v[Ind]))
write.csv(data.frame(P = BasinObs$P[Ind], E = BasinObs$E[Ind], q_gr4j = q, q_cn = r2$Qsim), file.path(out, "series.csv"), row.names = FALSE)
write.csv(L(im2$LayerPrecip), file.path(out, "lp.csv"), row.names = FALSE)
write.csv(L(im2$LayerFracSolidPrecip), file.path(out, "lf.csv"), row.names = FALSE)
write.csv(L(im2$LayerTempMean), file.path(out, "lt.csv"), row.names = FALSE)
writeLines(as.character(ro2$MeanAnSolidPrecip[1]), file.path(out, "masp.txt"))
"""


def _airgr_available():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "cat(requireNamespace('airGR', quietly=TRUE))"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "TRUE"


@pytest.fixture(scope="module")
def airgr_out():
    if not _airgr_available():
        pytest.skip("R/airGR not installed")
    d = Path(tempfile.mkdtemp())
    (d / "run.R").write_text(R_SCRIPT)
    subprocess.run(["Rscript", str(d / "run.R"), str(d)], check=True)
    return d


def test_gr4j_matches_airgr(airgr_out):
    s = pd.read_csv(airgr_out / "series.csv")
    q = gr4j(s.P.values.astype(float), s.E.values.astype(float), 257.24, 1.012, 88.23, 2.208)
    np.testing.assert_allclose(q, s.q_gr4j.values, atol=1e-6, rtol=1e-6)


def test_cemaneige_gr4j_matches_airgr(airgr_out):
    s = pd.read_csv(airgr_out / "series.csv")
    lp = pd.read_csv(airgr_out / "lp.csv").values.T.copy()
    lf = pd.read_csv(airgr_out / "lf.csv").values.T.copy()
    lt = pd.read_csv(airgr_out / "lt.csv").values.T.copy()
    masp = float((airgr_out / "masp.txt").read_text())
    liquid = cemaneige(lp, lf, lt, masp, 0.35, 4.2)
    q = gr4j(liquid, s.E.values.astype(float), 257.24, 1.012, 88.23, 2.208)
    np.testing.assert_allclose(q, s.q_cn.values, atol=1e-6, rtol=1e-6)
