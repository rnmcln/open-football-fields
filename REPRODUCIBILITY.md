# Reproducibility

This document describes how to reproduce every result in the thesis. Each script
stamps provenance (a global seed, the pinned StatsBomb open-data commit, and
library versions) into the artefacts it writes, and every reported number is
traceable to an artefact listed in `results_verification.csv`.

## Environment

* Python 3.11 or newer (a CPython 3.12 runtime was used for the recorded results).
* Core libraries: numpy, scipy, pandas, pyarrow, requests, shapely, matplotlib,
  mplsoccer, scikit-learn, pyyaml.
* Expected-threat fitting only: socceraction and statsbombpy, installed via the
  optional `[xt]` extra in a separate environment because socceraction pins
  pandas < 3. The runtime does not depend on socceraction; it consumes the
  fitted artefact `data/models/xt_grid.json` only.
* Additional statistical analyses (count models, mixed models): statsmodels, via
  the optional `[stats]` extra.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Offline tests

```bash
pytest -m "not network"
```

These cover geometry and coordinate transforms, the observation mask, control
fields (positional and kinematic), density, threat, flow operators, the
robustness gate, the information layer, and the validation statistics.

## Network or live-data tests

```bash
pytest -m network
```

These exercise real StatsBomb ingestion and orientation, 360 attachment, batch
ingestion, and Metrica tracking ingestion, and require internet access.

## Reproduction scripts and expected outputs

Run from the package root. The first run of each script downloads and caches the
data it needs; subsequent runs are fast. Each writes a figure and a
machine-readable summary to `figures/`, with a provenance record alongside.

| Command | Writes | Key reported quantities |
|---|---|---|
| `python scripts/demo1.py` | `figures/demo1_*` | observation ceiling; control and density entropies |
| `python scripts/demo2.py` | `figures/demo2_*` | threat, flow, robustness-gate outcomes |
| `python scripts/demo3.py` | `figures/demo3_*` | control fidelity gap by speed |
| `python scripts/demo4.py` | `figures/demo4_*` | discriminant, incremental, FDR (tournaments) |
| `python scripts/demo5.py` | `figures/demo5_*` | league reliability; team-vs-opponent |
| `python scripts/demo6.py` | `figures/demo6_*` | second-season generalisation |

## Supporting analysis scripts

These produce the additional figures and JSON summaries cited in the thesis. Each
writes to `figures/` and a provenance file alongside.

| Command | Writes | Thesis location |
|---|---|---|
| `python scripts/analysis_framework_diagram.py` | `figures/framework_schematic.*` | Ch1, Fig 1.1 |
| `python scripts/analysis_voronoi_schematic.py` | `figures/voronoi_schematic.png` | Ch2, Fig 2.1 |
| `python scripts/analysis_obs_ceiling.py` | `figures/analysis_obs_ceiling.*` | Ch4, Table 4.1 |
| `python scripts/analysis_obs_ceiling_wc2022.py` | `figures/analysis_obs_ceiling_wc2022.*`, `figures/analysis_obs_ceiling_comparison.*` | Ch4, Table 4.2 |
| `python scripts/analysis_gate_sensitivity.py` | `figures/analysis_gate_sensitivity.*` | Ch4, Tables 4.3-4.4 |
| `python scripts/analysis_fidelity_extended.py` | `figures/analysis_fidelity_extended.*` | Ch5, Table 5.1 |
| `python scripts/analysis_fidelity_sportec.py` | `figures/analysis_fidelity_sportec.*` | Ch5, Table 5.2 |
| `python scripts/analysis_fidelity_speedstat.py` | `figures/analysis_fidelity_speedstat.*` | Ch5, section 5.5 |
| `python scripts/analysis_control_constants.py` | `figures/analysis_control_constants.*` | Ch5, Table 5.3 |
| `python scripts/analysis_descriptor_stats.py` | `figures/analysis_descriptor_*` | Ch6, Tables 6.6, 6.10 |
| `python scripts/analysis_mens_league.py` | `figures/analysis_mens_league.*` | Ch6, Table 6.7 |
| `python scripts/analysis_mixed_models.py` | `figures/analysis_mixed_models.*` | Ch6, Table 6.8 |
| `python scripts/analysis_face_validity.py` | `figures/analysis_face_validity.*` | Ch6, Fig 6.4 |
| `python scripts/analysis_directness_passonly.py` | `figures/analysis_directness_passonly.*` | Ch6, section 6.5.5 |
| `python scripts/analysis_xprovider_wyscout.py` | `figures/analysis_xprovider_wyscout.*` | Ch6, Table 6.9 |
| `python scripts/analysis_xg_outcome.py` | `figures/analysis_xg_outcome.*` | Ch6, Table 6.11 |
| `python scripts/analysis_xg_robustness.py` | `figures/analysis_xg_robustness.*` | Ch6, section 6.6.2 |
| `python scripts/analysis_coverage_impact.py` | `figures/analysis_coverage_impact.*` | Ch4, Table 4.5, Fig 4.5 |
| `python scripts/analysis_dataset_inventory.py` | `figures/analysis_dataset_inventory.json` | Appendix E, Table E.1 |

The mixed-model, count-model, and dataset-inventory scripts use statsmodels; run
`pip install -e ".[stats]"` first. The men's-league and women's-season analyses
read per-match event parquet produced by batch ingestion; run the relevant
`demo*.py` or the batch ingestion once to populate the cache.

## Second open tracking provider (Sportec/IDSSE)

The Sportec fidelity replication (`analysis_fidelity_sportec.py`) reads the
Sportec/IDSSE open tracking files (Bassek et al., 2025; CC-BY 4.0). Because the
position XML is large (about 2.6 GB across seven matches) it is not committed;
fetch it in one command:

```bash
python scripts/fetch_sportec.py        # all 7 matches (resumable; skips complete files)
python scripts/analysis_fidelity_sportec.py
```

The data come from the PySport mirror used by kloppy; each match has a
`meta_<id>.xml` and a `track_<id>.xml`. The seven identifiers
(`J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9`) are coded in both
scripts; the analysis runs on whichever complete files are present. The raw
coordinates and per-player speeds were checked to be identical to
`kloppy.sportec.load_open_tracking_data(..., coordinates="secondspectrum")`.

## Cross-provider check (Wyscout open)

The cross-provider analysis (`analysis_xprovider_wyscout.py`) reads the Wyscout
open event data (Pappalardo et al., 2019; CC-BY) for the Premier League 2017/18
from `data/cache/wyscout/`. The match files come from the kloppy-compatible
mirror (`koenvo/wyscout-soccer-match-event-dataset`,
`processed-v2/files/<id>.json`); the 380 England match identifiers are listed in
that repository. The script parses each match through kloppy, caches one
descriptor row-file per match under `data/cache/wyscout_rows/`, and is resumable.

## Re-fitting the expected-threat surface (optional)

```bash
pip install -e ".[xt]"      # in a pandas<3 environment
python scripts/fit_xt.py    # writes data/models/xt_grid.json
```

## Verifying the reported numbers

Each reproduction script prints a JSON summary and writes it to
`figures/demoN_summary.json`. Compare these, and the `analysis_*` JSON outputs,
against the consolidated `results_verification.csv`, which records each reported
quantity, its value, the source artefact, and the generating script.

## Future data and extensions

Two scripts are provided for extending the analyses to open data released after the core results were fixed (Section 8.2 of the thesis). They require their respective datasets and internet access, and are not part of the pinned reproduction set.

| Command | Purpose |
|---|---|
| `python scripts/analysis_obs_ceiling_any.py <comp> <season> "<name>"` | Observation ceiling for any StatsBomb 360 competition (e.g. UEFA Euro 2024), extending the Section 4.5.2 replication. Identifiers are in the StatsBomb `competitions.json`. |
| `python scripts/analysis_obs_ceiling_compare.py` | Combines all per-competition observation-ceiling summaries (Euro 2020, WC 2022, and any `analysis_obs_ceiling_any.py` runs) into one comparison table and figure, extending the Section 4.5.2 replication across competitions and sexes. |
| `python scripts/analysis_pff_paired.py` | Scaffold for the external validation of the coverage-bias result (Section 4.5.7) against PFF FC 2022 broadcast tracking on the same World Cup matches. Requires the PFF data (via kloppy) and a provider-matching step; prints guidance when run without them. |

The coverage-impact result itself (`analysis_coverage_impact.py`) runs on the pinned StatsBomb 360 cache and needs no additional data.

## Determinism

A single global seed (in `config/default.yaml`) controls all stochastic steps
(kernel-density resampling, bootstrap, permutation tests, k-means). The seed, the
pinned StatsBomb commit, and the library versions are stamped into every
generated artefact.

