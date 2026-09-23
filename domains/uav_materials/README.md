# UAV materials domain

Represents *what a UAV design wants* (`TargetMaterialProfile`) and *what real
materials have* (`MaterialRecord`). Physical units, test conditions, explicit
missing data and **measurement-level provenance** are kept throughout. The
domain depends only on JevPilot's public API, and JevPilot never imports it.

## Pipeline (iteration 2)

```text
TargetMaterialProfile + real records (MIL-HDBK-5J, NRL AD0609618)
  → harmonize (canonical units; originals kept)       harmonize.py
  → condition compatibility (exact/compatible/...)    compatibility.py
  → measurement selection + not_applicable policy     selection.py
  → hard constraints: FEASIBLE / UNDETERMINED / INFEASIBLE, per-constraint reasons
  → soft preferences: direction-aware penalties, coverage   ranking.py
  → deterministic ranking with explanations            search.py
```

Run: `python examples/uav_material_search.py` (real data) and
`python examples/uav_target_profile.py` (synthetic fixtures).

## Real data

| Source | Reuse | Used for | Materials |
|---|---|---|---|
| MIL-HDBK-5J (DoD, 2003), Internet Archive `milhdbk-5-j` | "Distribution Statement A. Approved for public release; distribution is unlimited." US Government work | room-temperature design allowables: Ftu, Fty, Fcy, Fsu, elongation, E, G, density | 43 (aluminum 23, titanium 7, heat-resistant 4, other 9) from 53 of 209 tables |
| NRL, *Corrosion of Metals in Tropical Environments, Part 6* (DTIC AD0609618) | "Unlimited availability"; US Government work | 5 quote-verified corrosion statements (1–16 years of tropical exposure) | 6061 (linked at alloy level) + 3 NRL-only records |

Layout: `data/uav_materials/{sources,raw,processed}`.
- `sources/*.json` hold the license statement, URL, retrieval date, checksums, and what is and isn't used.
- `raw/` holds the unmodified originals (large PDFs are git-ignored and checksummed) and the committed text extracts that parsers read.
- `processed/` is rebuilt with `python -m domains.uav_materials.ingest.build`; add `--from-pdf` to re-extract, which needs the `[data]` extra.

The MIL-HDBK-5J parser is **strict**. A table is rejected, with a reason in
`processed/build_report.json`, when any of these holds:
- its layout is ambiguous;
- its value count doesn't equal rows × columns;
- a row label is duplicated;
- a column violates physics (Fty > Ftu, Fsu ≥ Ftu, elongation outside 0–100 %).

NRL table images are not transcribed.

## Policies

- **Identity** (`identity.py`): materials are matched on parsed designation parts (system, designation, cladding, temper), never on display names. A query without a temper matches only at *alloy level*, and the link is recorded on every attached measurement. Different designations (AZ31X vs AZ31B) are never merged.
- **Compatibility**: temperature, exposure environment class, exposure duration and humidity each have explicit equivalence rules. A condition the requirement states but the data doesn't report makes that measurement unusable for the requirement. It is never assumed.
- **Multiple measurements**: nothing is averaged. The selection order is:
  1. best compatibility tier;
  2. exact identity over alloy-level identity;
  3. statistical basis A > B > S > typical;
  4. the conservative value for the requirement's direction.

  Alternatives are kept. Bounds ("≤ x") only answer upper limits.
- **not_applicable**: satisfies only `water_absorption <=` (no uptake mechanism). Everything else stays **undetermined**.
- **Ranking**: only numerically weighted soft preferences are scored; qualitative importance is listed as unscored. The penalties are:
  - maximize / minimize: relative gap to the best viable value;
  - between: 0 inside the range, otherwise distance to the nearest bound / range width;
  - target: |v − t| / |t|.

  The distance is a weighted RMS over the covered preferences, and coverage is reported. The ordering uses a *pessimistic* distance that counts each missing preference at max(1, worst observed penalty), so missing data never improves a rank. Normalization parameters are returned with every result.

## Limitations

- **Real-data coverage:** strength, deformation and density are strong (42–43 of 46 materials). Corrosion is weak (5), and water absorption and temperature effects have none. No polymers, composites, steels or magnesium alloys come from the real sources yet.
- **MIL-HDBK-5J is superseded** by MMPDS. Its values are historical design allowables.
- **Conservative envelope:** the selection takes the envelope across thickness ranges and grain directions of a table, which is conservative for thick products.
