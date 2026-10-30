# Leaderboard submission

These results follow the [FP-DeErr contribution format](https://github.com/mogroupumd/FP-DeErr/blob/0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815/ADDING_A_POTENTIAL.md). The guide requests email submission to Prof. Yifei Mo; the pull request supplies an optional reviewed patch for integration.

[Complete standardized results and checksums](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/tag/audited-mace-2026-10) · [Metric tables](reports/additional-models/tables) · [Audits and execution methods](reports/additional-models/README.md) · [Proposed leaderboard patch](https://github.com/mogroupumd/FP-DeErr/pull/1)

| Model | Exact head | Parameter count in saved checkpoint |
| --- | --- | ---: |
| MACE-OMAT-0 Small | omat_pbe | 8,222,244 |
| MACE-OMAT-0 Medium | default | 9,063,204 |
| MACE-MH-1 | omat_pbe | 6,439,878 (whole six-head checkpoint) |

Exact release assets, SHA-256 hashes and training-data notes are in the [checkpoint record](reports/additional-models/README.md). Counts above were measured from the loaded official checkpoint parameters; they include all checkpoint heads and are not architecture-size estimates.

Code: [MACE 0.3.16](https://github.com/ACEsuit/mace/tree/v0.3.16), [MIT license](https://github.com/ACEsuit/mace/blob/main/LICENSE.md). Checkpoints: the official [foundation-model roster](https://github.com/ACEsuit/mace-foundations#models) identifies MACE-OMAT-0 and MACE-MH-1 as ASL-licensed. Consult their checkpoint-specific license terms; model weights are linked rather than redistributed here. Model citations are provided in the official foundation-model repository and model cards.

## Complete results files

The release contains one full gzipped force-results JSON per dataset with all three MACE models, static and relaxed phase results per checkpoint, full standardized NEB exports, a table/audit ZIP, and SHA256SUMS.txt. Force entries preserve all original atoms, aligned structure IDs and atom indices, signed magnitude errors, vector errors and undefined angles as NaN (the upstream Python JSON convention). Decoded arrays are checked byte-for-byte against the audited float64 shards. No filtering, rounding or scientific recalculation is substituted in packaging.

Static and relaxed phase files retain their original bytes after decompression. NEB files retain the audited standardized exports. No checkpoint weights or original source datasets are included. These downloadable outputs supplement the summaries; retained raw Cartesian force shards remain local.

## Environments and settings

Python 3.10; mace-torch 0.3.16; float64; 6 Angstrom cutoff; no added dispersion. Complete per-stage dependency versions and checkpoint identity are retained in the audit/provenance records. CPU/CUDA phase relaxation used the same ASE 3.23.0 FIRE optimizer, fixed cells, fmax 0.01 eV/Angstrom and a 10,000-step limit. NEB used ASE 3.26.0 with recorded settings, differing from the original upstream environment as disclosed. The supplied DFT reference data were reused.

The [original workflow instructions](WORKFLOW.md) describe dataset setup and the first completed evaluation. The additional-model command adapters are in [validation/round2/adapters_v2](validation/round2/adapters_v2); the manifests pin exact scientific sources. The full raw-source audits require access to the complete local retained outputs and supplied references; portable summaries alone cannot rerun them.

The public round2_scope.json omits the private email message identifier; the audit records retain the hash of the original local scope. Scientific configuration fields are unchanged. Published source/adapters are exact copies, so recorded source hashes remain meaningful. Paths and production queue controls must be configured for a new execution environment.
