# Report skeleton (max 6 pages) - fill the [brackets], export to PDF as docs/report.pdf

1. **Problem and approach (0.5 p)** - 3 questions; pipeline diagram (copy from README); educational prototype.
2. **Data (0.5 p)** - S1 GRD IW (same track, [dates, track number]), S2 L2A [clear fraction], Copernicus DEM GLO-30, OSM snapshot 2026-07-27 via ohsome, training data. State the data rules and that EMSR927 was used only for checking.
3. **Flood and debris mapping (1 p)** - preprocessing (calibration, terrain correction, boxcar [5x5]); baseline log-ratio thresholds [-3 dB, -17 dB]; slope mask [30 deg]; debris heuristic; minimum area.
4. **AI component (1.25 p)** - U-Net resnet34, 4 input channels, BCE+Dice, data split, **held-out Himalayan scenes** definition, table of IoU/F1/precision/recall vs baseline, failure cases (shadow, wet snow, debris).
5. **Damage and cut-off analysis (1 p)** - overlay logic (likely hit / possibly hit with 40 m buffer), graph construction, confirmed vs conservative scenarios, results for Trishuli [numbers from summary.json]. Bonus flow path [if done].
6. **Case study vs EMSR927 (0.75 p)** - IoU/precision/recall/area comparison, map figure, discussion of disagreement (EMS uses optical + analyst judgement; our radar sees different things; evaluation region = reference bbox).
7. **Limitations (0.75 p)** - copy and extend:
   - **No early warning.** Sentinel satellites revisit every few days (~12 days per single track); a glacier collapse and flood happen in minutes to hours. This is a response tool, not a warning system.
   - **Temporal gap.** The after-image may be days old: water may have receded, and new debris/landslides after the image are missed.
   - **Radar geometry in mountains.** Layover/shadow/steep slopes are masked or unreliable; flooding in deep gorges may be invisible. Wet snow looks like water in radar (false positives).
   - **Resolution.** 10-20 m pixels: narrow channels, single houses, footbridges, foot trails are missed. "Hit" means overlap with detected extent, not structural damage level.
   - **Debris is hard.** Debris flows are not mostly water; the training data is mostly water floods; the debris class is heuristic and weakly validated.
   - **OSM completeness.** Cut-off result depends on pre-event OSM roads/places in a remote area; unmapped roads, trails, helicopter access, landslide blockages, and repairs are not modelled. Settlements without a mapped road are reported as "unmapped".
   - **Optical limits.** Monsoon clouds; Sentinel-2 only confirms, never the sole source.
   - **Same-track requirement.** If no same-track before/after pair exists, the system cannot run change detection reliably.
   - **Not operational.** Needs ground verification; outputs must not drive life-safety decisions on their own.
8. **Ethics and responsibility (0.25 p)** - no images of victims; respectful framing; educational prototype.
9. **Attributions and citations** - see docs/ATTRIBUTIONS.md.
