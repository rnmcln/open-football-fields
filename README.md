# ffields

**Spatial fields and team descriptors from open football data, estimated under
partial observation.**

This repository accompanies the doctoral thesis *Estimation under partial
observation in association football: spatial fields and team descriptors from
open data* (Aaron Lawson McLean; monograph format). It implements, tests, and
reproduces every empirical result in the thesis from openly licensed football
data. The compiled monograph is included as `THESIS_MONOGRAPH.pdf`.

All results are associational and descriptive; none is causal or prescriptive.
Third-party match data are used under their respective licences and are not
redistributed by this repository.

## Summary of the work

Modern football analytics represents the game as scalar and vector fields over
the pitch: pitch control (a probability that a team would win possession at each
location), expected threat (a value surface), event density (a spatial
intensity), and possession flow (a vector field). These representations are
informative but demanding of data, because a value defined over the whole pitch
is only as good as the observation of the player configuration behind it. Open,
reproducible research rarely has access to the complete continuous tracking used
in the professional game; it works instead with event records, broadcast-derived
freeze frames, and small tracking samples, all of which observe only part of the
match state.

The thesis separates four levels throughout: the latent match state, the partial
observation actually recorded, the estimate computed from that observation, and
the empirical claim made from the estimate. A quantity is interpreted only when
the observation and the validation evidence support it. Within this frame the
work makes three measured contributions, one per empirical chapter.

1. **The observation ceiling (Chapter 4).** An explicit observation-operator
   formalism for open data, with player-configuration fields reported inside the
   camera-visible region rather than extrapolated beyond it. The ceiling is
   quantified across two full competitions (UEFA Euro 2020, 166,871 frames; FIFA
   World Cup 2022, 203,882 frames): a typical broadcast frame observes about a
   quarter to a third of the pitch and roughly two thirds of the players, and
   least of all at shots. A pre-specified robustness gate admits stable field
   operators (the possession-flow magnitude) and rejects unstable ones (the curl
   of the flow field; the divergence only at coarse resolution). An internal proxy
   on the same data shows the systematically-missing far-from-ball players are
   consequential for pitch control, not merely numerous: removing about five of
   them flips the controlling team in roughly a fifth of reported cells, a larger
   effect than omitting player velocities.

2. **The fidelity ceiling (Chapter 5).** A speed-stratified measurement of the
   error introduced by the missing player velocities that open broadcast data do
   not record. On identical frames from two independent open tracking providers
   (Metrica, two matches; Sportec/IDSSE, seven matches), position-only pitch
   control agrees closely with a velocity-bearing reference in settled play and
   diverges progressively as players move faster, with the controlling-team
   cell-flip fraction rising from about one cell in eighty to about one in
   thirteen. The pattern is robust to the control constants.

3. **The validation battery (Chapter 6).** A measurement-validity treatment of
   field-derived team descriptors, combining discriminant validity,
   cross-competition and cross-season replication, bootstrap split-half
   reliability, a confounder-controlled team-versus-opponent decomposition, and
   incremental value for shot counts and expected goals. Directness of ball
   progression behaves as a stable, largely opponent-independent team property
   across two women's league seasons, a men's league season, and an independent
   event provider; threat-based descriptors are more context-dependent, and an
   entropy-based dispersion descriptor performs poorly. Quantities that fail
   their tests are reported as results in their own right.

## Relevance

The contribution is a measurement-and-validation discipline for open football
analytics rather than a new field model or value metric. It gives practitioners
a workflow: make the observation operator explicit, mask rather than
extrapolate, gate noise-amplifying operators before use, treat position-only
control as reliable mainly in settled play, and validate any team-level
descriptor for discriminability, reliability, and opponent-specificity before
interpreting it as a team property. None of this requires proprietary data, and
the same discipline applies to any measurement system with a limited field of
view and an error model.

## Repository layout

```
src/ffields/        core package: geometry, observation operator, fields
                    (control, threat, density, flow), robustness gate,
                    information layer, validation statistics, data ingestion
scripts/            demo1..demo6 reproduction scripts, fit_xt, and the
                    analysis_* scripts behind the appendix and results figures
tests/              offline unit tests and network-gated integration tests
config/             default.yaml: every numerical default and the global seed
figures/            derived, aggregate result artefacts (PNGs and JSON
                    summaries with provenance stamps)
data/models/        the fitted expected-threat surface (aggregate parameters)
references.bib                 bibliography
references_verification.csv    per-reference verification record
results_verification.csv       per-number verification record (quantity,
                               value, source artefact, generating script)
THESIS_MONOGRAPH.pdf           the compiled monograph
```

The raw match-data cache is not part of this repository; see **Data access**.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or newer is recommended. The optional `[xt]` extra installs
`socceraction` and `statsbombpy` and is needed only to re-fit the
expected-threat surface; it is not required at runtime. The optional `[stats]`
extra installs `statsmodels` for the count-model and mixed-model analyses.

## Tests

```bash
pytest -m "not network"   # offline unit tests
pytest -m network         # live-data integration tests (require internet)
```

## Reproducing the figures and results

```bash
python scripts/demo1.py   # observation ceiling, positional control, density
python scripts/demo2.py   # threat field, possession flow, robustness gate
python scripts/demo3.py   # control fidelity gap (Metrica), information layer
python scripts/demo4.py   # validation battery (tournaments)
python scripts/demo5.py   # league-season reliability and confounders
python scripts/demo6.py   # second-season generalisation
```

Each script writes figures and a machine-readable summary to `figures/`, with a
provenance record (seed, pinned data commit, library versions) alongside. The
`analysis_*` scripts reproduce the appendix and supporting-analysis figures. Full
instructions, including the one-command Sportec and Wyscout fetchers, are in
`REPRODUCIBILITY.md`. Every number in the thesis is traceable to an artefact
listed in `results_verification.csv`; none is entered by hand.

## Data access and licensing

This repository contains no third-party match data. StatsBomb open data, the
Metrica Sports sample, the Sportec/IDSSE open dataset (CC-BY 4.0), and the
Wyscout open data are downloaded on demand into a local cache that is excluded
from version control. See `DATA_AVAILABILITY.md` for how to obtain each source
and `ATTRIBUTION.md` for attribution requirements. StatsBomb data are governed by
the StatsBomb Public Data User Agreement and must not be redistributed.

## Citation

If you use this software, please cite it using `CITATION.cff`, and cite the
thesis and attribute the data providers as described in `ATTRIBUTION.md`.

## Contact

Aaron Lawson McLean. ORCID: 0000-0001-5528-6905.

## Licence

Code is released under the MIT Licence (`LICENCE`). Third-party data are not
covered by that licence and are not redistributed.
