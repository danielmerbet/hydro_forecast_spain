# ACA historical daily streamflow export (2007-01-01 → 2024-10-20)

**What:** Daily mean river discharge (m³/s) for the gauging stations (*aforaments*) and
reservoir outflows (*embassaments*) of Catalonia's internal river basins, operated by
the Agència Catalana de l'Aigua (ACA).

**Source:** ACA web service *"Consulta de dades del medi"*
(<https://aplicacions.aca.gencat.cat/sdim21/>), exported by hand in October 2024 as two
Excel/CSV files, because the Catalan open-data API (`3yr3-vq6y`, used by
`codes/01_download_aca_gauges.py`) only starts in 2020 and the calibration needs a
longer record. The files were later stored in `intoDBP/ACA_Catalunya/`.

**Changes made when copying here:** re-encoded from CP850 (DOS Latin) to UTF-8 and
gzip-compressed. No values were modified. Checksums are in `SHA256SUMS`.

| column | meaning |
|---|---|
| `date` | day (YYYY/MM/DD, local time) |
| `station` | station name |
| `basin` | river basin (conca) |
| `X`, `Y` | station coordinates, UTM zone 31N, ETRS89 (EPSG:25831) |
| `variable` | ACA series code + description, e.g. `EA010_Girona_Cabal riu Ter` |
| `value` | daily mean discharge |
| `units` | always m3/s |

**Licence / terms:** ACA data are public-sector information of the Generalitat de
Catalunya (reuse allowed with attribution: "Font: Agència Catalana de l'Aigua").
