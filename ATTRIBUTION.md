# Data attribution and licensing

## StatsBomb open data

This project uses **StatsBomb open data**, retrieved on demand from
<https://github.com/statsbomb/open-data>. The data are **not** redistributed in
this repository; the ingestion layer downloads and caches them locally at run
time (see `src/ffields/ingest/statsbomb.py`).

Use is governed by the **StatsBomb Public Data User Agreement**, *not* by a
Creative Commons licence. Obligations carried into every downstream output
(papers, figures, slides, the thesis itself):

1. **Attribute StatsBomb** as the data source in any published work.
2. **Display the StatsBomb logo** in published outputs, per their agreement.
3. **Register an email** with StatsBomb as required by the agreement.
4. **Do not re-host or redistribute** the raw data. Ship loaders + source
   links instead (this repository does so).

> Verification status: the existence and broad terms of the agreement are
> **[V]** (verified). The exact current wording can change; re-read the live
> agreement at submission time. Academic/thesis use is discussed in issue #47
> of the open-data repository.

### Required citation (data)

> StatsBomb. *StatsBomb Open Data.* GitHub repository,
> <https://github.com/statsbomb/open-data>. Commit pinned for this project in
> `data/.statsbomb_commit_sha.txt`.

## Other data sources (not yet used in code; listed for forward planning)

- **Wyscout / Pappalardo et al. (2019)**, *Scientific Data* 6:236 — openly
  downloadable via figshare with a citation requirement. Verify the exact CC
  variant before any redistribution. **[V]**
- **Metrica Sports sample tracking data** — open for research; cite. **Used** (`src/ffields/ingest/metrica.py`) as the continuous-tracking source
  for the kinematic-control fidelity-gap analysis, which open event/360 data
  cannot support. Downloaded on demand and cached (gitignored); not
  redistributed. Source:
  <https://github.com/metrica-sports/sample-data>. **[V]**

## Software and method reuse

The threat field T reuses **expected threat (xT)**:

- **Singh, K. (2019).** *Introducing Expected Threat (xT).* Method origin.
- **socceraction** (Van Roy, Robberechts, et al.), the reference open-source
  implementation used to fit the xT surface (`scripts/fit_xt.py`). MIT-licensed.
  Cite the socceraction papers (e.g. Decroos et al., 2019, *Actions Speak Louder
  than Goals*, KDD) where appropriate.

The fitted artefact `data/models/xt_grid.json` holds **aggregate model
parameters** (a 16x12 value grid) derived from StatsBomb open data, not the raw
data; it is gitignored and regenerable. Reusing socceraction rather than
reimplementing xT is the project's anti-decoration discipline and keeps the
threat field comparable with the published literature.

## Provenance

Every ingested match is written to the local cache with provenance metadata
(source URL, retrieval timestamp, open-data commit SHA, library versions).
See `src/ffields/provenance.py`. The xT artefact additionally records the
training competitions, game/action counts, and socceraction version.
