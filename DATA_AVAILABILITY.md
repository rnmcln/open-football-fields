# Data availability

This repository does not contain any third-party match data. All data are openly
available from their original providers and are downloaded on demand into a local
cache that is excluded from version control.

## Datasets used

**StatsBomb open data** (events and broadcast-derived 360 freeze-frames). Source:
https://github.com/statsbomb/open-data. Used for the field framework (Chapter 4)
and the team-descriptor studies (Chapter 6). The open-data commit is pinned in
`data/.statsbomb_commit_sha.txt` so that event and 360 content are fixed for
reproduction. Competitions used include UEFA Euro 2020, FIFA World Cup 2022, and
the FA Women's Super League 2018/19 and 2019/20. Use is governed by the StatsBomb
Public Data User Agreement; the data must not be redistributed and must be
attributed (see `ATTRIBUTION.md`).

**Metrica Sports sample tracking data** (continuous tracking, derivable
velocities). Source: https://github.com/metrica-sports/sample-data. Used only in
Chapter 5 to measure the fidelity ceiling for pitch control. Provided as an open
sample for research; not redistributed here.

## What is cached locally

On first run, the ingestion modules download the required files and store them
under `data/cache/` (events, matches, 360 frames, batch event parquet, SPADL
actions, and Metrica tracking). This directory is gitignored. The fitted
expected-threat surface is written to `data/models/xt_grid.json`; it is an
aggregate grid of model parameters derived from open data, not raw data, and is
also gitignored and regenerable.

## What is excluded from version control

Raw and cached data, the local virtual environment, and regenerable figures are
excluded (see `.gitignore`). The repository ships code, configuration, the pinned
commit identifier, verification tables, and the thesis manuscript only.

## How to obtain the data

No manual download is required: running the tests or the study scripts retrieves
the data automatically, subject to internet access and the providers' terms.
Users must accept the StatsBomb Public Data User Agreement before use.


## Further open sources (extensions; not used in the core analyses)

These support the extensions described in Section 8.2 of the thesis and the
`analysis_obs_ceiling_any.py` and `analysis_pff_paired.py` scripts. They are not
required to reproduce the thesis results and are not redistributed here.

* Additional StatsBomb 360 competitions (e.g. UEFA Euro 2024, FIFA Women's World
  Cup 2023, UEFA Women's Euros), under the StatsBomb Public Data User Agreement.
* PFF FC 2022 World Cup broadcast tracking plus events (free, form-gated from PFF;
  loadable via kloppy). Fuller observation of the same matches on which the
  observation ceiling is measured; used for the coverage-bias validation.
* SkillCorner open data (ten matches; broadcast tracking; GitHub SkillCorner/opendata).
* SoccerNet Game State Reconstruction benchmark (broadcast clips with reference
  on-pitch positions), for quantifying vision-tracker error.
* Google Research Football and RoboCup simulation logs (full latent state), for
  benchmarking masking against imputation under a known missingness mechanism.
