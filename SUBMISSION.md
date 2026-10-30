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

## Download index

| Complete result file | Size (MiB) | SHA-256 |
| --- | ---: | --- |
| [matpes_pbe_force_results_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/matpes_pbe_force_results_standardized.json.gz) | 481.9 | `fa6f5cb304119ae380aabde2aeea4596bbdc9efd147e68c0ac030474c39d6c3f` |
| [omat24_force_results_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/omat24_force_results_standardized.json.gz) | 228.7 | `bc14e2095a57e868bac0b70a2bb853a2801ed2a02718624df8396b1f36647fd8` |
| [mace_omat_0_small_static_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_small_static_phase_standardized.json.gz) | 4.2 | `5a2ed7844f42b4afefacdc1cbc467ca1c386d857491890cde94c3b364dfb942c` |
| [mace_omat_0_small_relaxed_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_small_relaxed_phase_standardized.json.gz) | 19.7 | `0972f0b56ab4d33b63a5cf46c0cca20240820b20102de10af3ffbd25e7ba1d48` |
| [mace_omat_0_medium_static_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_medium_static_phase_standardized.json.gz) | 4.2 | `15089ce7c05168fc55ed9f4905abe89e5ed3290f5906c96dfc4b32674375ae30` |
| [mace_omat_0_medium_relaxed_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_medium_relaxed_phase_standardized.json.gz) | 19.7 | `35998a7a2611a1d0e866e0a6dcfcd321251d8213808122536274ccba32384b60` |
| [mace_mh_1_omat_pbe_static_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_mh_1_omat_pbe_static_phase_standardized.json.gz) | 4.2 | `d9d4228a24ca315181783c1980cf6b787de450abdd097213de4803b4460d5f45` |
| [mace_mh_1_omat_pbe_relaxed_phase_standardized.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_mh_1_omat_pbe_relaxed_phase_standardized.json.gz) | 19.7 | `91771e70078ac5c8c63dbc3952d94702068e8c25fcf2b2f0ddfc16e942c17fcb` |
| [mace_omat_0_small_neb_results.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_small_neb_results.json.gz) | 16.8 | `5823ce99c07f738d4ee7ec8c603e52c1072dc885d286a9db3112f3459281d300` |
| [mace_omat_0_medium_neb_results.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_omat_0_medium_neb_results.json.gz) | 16.8 | `e1e1236d29216162ac00c26fa513aaf2397acc2f1f36513644e98add4bb0ec34` |
| [mace_mh_1_omat_pbe_neb_results.json.gz](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/mace_mh_1_omat_pbe_neb_results.json.gz) | 16.8 | `7c98976b99c9a1fbf016f8eb61eeb4e66a93d1bc16e96d420e213284b1af3fc1` |
| [review_bundle.zip](https://github.com/michael-wang0605/fp-deerr-mace-evaluation/releases/download/audited-mace-2026-10/review_bundle.zip) | 0.1 | `735e0da10feb285e0841de975f908ccfb96dcf138504c5641ac78cdcc5a32eac` |

[Export validation and asset manifest](validation/round2/submission_assets). General model citation: [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096); consult the linked official release/model-card citations for each checkpoint.
