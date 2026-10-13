# FP-DeErr reproduction and MACE-OMAT-0 Small evaluation

VERIFIED FINAL REPORT

2026-10-12 23:04 UTC

Prof. Mo requested running FP-DeErr with an included potential to check reproduction, then applying the same workflow to additional potentials. This study uses CHGNet 0.3.0 as the baseline and the official MACE-OMAT-0 Small checkpoint as an additional potential.

Fresh CHGNet static phase/order calculations reproduce 11/11 displayed metrics; Both full MatPES-PBE and OMat24 force benchmarks reproduce all 8 displayed cells each. Static migration-barrier metrics also match. Full-relaxation metrics differ and are reported without claiming exact reproduction. MACE has lower energy MAE in both the static and full-relaxation benchmarks. Full MatPES-PBE force-magnitude MAE is 0.20 versus 0.31 eV/A for CHGNet. Its completed static NEB barrier MAE is 0.0707 eV versus 0.1124 eV for CHGNet. Full NEB barrier MAE is 0.0558 versus 0.1163 eV for CHGNet.

## Calculation coverage

| Model | Workflow | Coverage | Status |
| --- | --- | --- | --- |
| CHGNet | Static phase stability and ordering | 6,697/6,697 structures | Complete |
| CHGNet | Phase-stability relaxation | 597/597 structures | Complete |
| CHGNet | Ordering relaxation | 6,100/6,100 structures | Complete |
| CHGNet | MatPES-PBE forces | 434,712/434,712 structures | Complete |
| CHGNet | OMat24 rattled-1000 forces | 117,004/117,004 structures | Complete |
| CHGNet | Full NEB (+1,078 static images) | 154/154 paths | Complete; 152 converged |
| MACE-OMAT-0 Small | Static phase stability and ordering | 6,697/6,697 structures | Complete |
| MACE-OMAT-0 Small | Phase stability and ordering relaxation | 6,697/6,697 structures | Complete; 6697 converged |
| MACE-OMAT-0 Small | MatPES-PBE forces | 434,712/434,712 structures | Complete |
| MACE-OMAT-0 Small | OMat24 rattled-1000 forces | 117,004/117,004 structures | Complete |
| MACE-OMAT-0 Small | Full NEB (+1,078 static images) | 154/154 paths | Complete; 152 converged |

## Static phase stability and ordering

| Metric | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- |
| Phase stability energy MAE (meV/atom) | 246 | 62 |
| Ground-state agreement (%) | 55.9 | 60.9 |
| Within-phase agreement (%) | 48.8 | 66.9 |
| Global hull-minimum agreement (%) | 22.7 | 22.7 |
| Ordering energy MAE (meV/atom) | 270 | 51 |
| Top-1 accuracy (%) | 20.3 | 42.0 |
| Recall@3 (%) | 29.7 | 52.3 |
| Recall@10 (%) | 59.0 | 71.3 |
| Spearman correlation | 0.24 | 0.54 |
| Ranking-error rate (%) | 40.6 | 27.9 |
| Mean / max DFT energy penalty (meV/atom) | 10 / 29 | 6 / 16 |

The model documentation identifies OMat24 as MACE-OMAT-0 training data. Exact structure-level training overlap across these benchmarks has not been audited. These results describe performance against the supplied reference labels, not independently established out-of-distribution performance. MACE-OMAT-0 targets PBE/PBE+U labels; MatPES-PBE is a PBE reference, so reference-level differences can affect comparisons.

## Full phase stability and ordering

Static predictions use supplied DFT-relaxed geometries. Full relaxations start from the supplied initial structures, using the unchanged fixed-cell ASE FIRE routine (maximum force 0.01 eV/A, 10,000 maximum steps). Metrics and geometry mapping use the public analysis functions. All candidates, failures and non-convergence records are retained.

| Metric | Published CHGNet | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- | --- |
| Energy MAE (meV/atom) | 242 | 241 | 97 |
| Ground-state agreement (%) | 53.4 | 53.4 | 60.2 |
| Within-phase hull-minimum agreement (%) | 40.5 | 39.7 | 59.5 |
| Global hull-minimum agreement (%) | 18.2 | 18.2 | 27.3 |
| Structure mapping success (%) | 87.3 | 87.4 | 89.4 |
| Mean / max RMSD (A) | 0.37 / 2.28 | 0.37 / 2.28 | 0.27 / 2.45 |
| RMSD < 0.05 A (%) | 11.9 | 11.9 | 22.5 |
| RMSD < 0.1 A (%) | 26.5 | 26.4 | 43.3 |
| RMSD < 0.2 A (%) | 49.5 | 49.8 | 64.4 |

| Metric | Published CHGNet | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- | --- |
| Ordering energy MAE (meV/atom) | 272 | 271 | 76 |
| Top-1 accuracy (%) | 8.2000 | 7.5 | 24.9 |
| Recall@3 (%) | 20.1000 | 20.3 | 34.0 |
| Recall@10 (%) | 52.8000 | 52.6 | 60.3 |
| Spearman correlation | 0.0900 | 0.08 | 0.28 |
| Ranking-error rate (%) | 46.8000 | 46.8 | 39.3 |
| Mean / max DFT energy penalty (meV/atom) | 13 / 39 | 13 / 39 | 12 / 35 |

Fresh CHGNet hull relaxation matches 4/9 displayed relaxation/geometry values; ordering relaxation matches 2/7. Full NEB matches 9/18 leaderboard fields, including coverage counts and static-barrier fields. Full barrier MAE is 0.1163 eV versus 0.1161 eV published. The original full generation environment is unavailable. Hardware, dependency and trajectory effects are possible explanations, not established causes. A largest hull-energy discrepancy remained stable across repeated 1/2-thread calculations.

## Force prediction

The force benchmark populations are the full MatPES-PBE-2025.1 and OMat24 validation rattled-1000 datasets; the coverage table records completion. Canonical Cartesian conversion and notebook summaries are unchanged. Undefined angles are excluded as in the public workflow; the general analysis uses the strict DFT-force threshold >0.01 eV/A. FE atoms have DFT-force magnitude >1 eV/A. Published tables retain their own displayed precision.

### MatPES-PBE force comparison

| Metric | Published CHGNet | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- | --- |
| Force-magnitude MAE / RMSE (eV/A) | 0.31 / 2.19 | 0.31 / 2.19 | 0.20 / 0.98 |
| Force-angle MAE / RMSE (degrees) | 45 / 66 | 45 / 66 | 34 / 56 |
| Magnitude MAE / RMSE, magnitude error <1 eV/A | 0.22 / 0.32 | 0.22 / 0.32 | 0.16 / 0.25 |
| Joint accuracy, angles <1 / <20 degrees (%) | 0.66 / 2.2 | 0.66 / 2.2 | 1.2 / 5.7 |
| Magnitude error <0.01 eV/A (%) | 5.6 | 5.6 | 8.8 |
| Magnitude error >1 eV/A (%) | 5.71 | 5.71 | 2.45 |
| Magnitude MAE / RMSE on FE atoms (eV/A) | 0.70 / 4.14 | 0.70 / 4.14 | 0.38 / 1.83 |
| Angle MAE / RMSE on FE atoms (degrees) | 24 / 36 | 24 / 36 | 15 / 25 |

### OMat24 force comparison

| Metric | Published CHGNet | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- | --- |
| Force-magnitude MAE / RMSE (eV/A) | 0.76 / 4.00 | 0.76 / 4.00 | 0.14 / 0.32 |
| Force-angle MAE / RMSE (degrees) | 13 / 22 | 13 / 22 | 5 / 10 |
| Magnitude MAE / RMSE, magnitude error <1 eV/A | 0.34 / 0.42 | 0.34 / 0.42 | 0.12 / 0.19 |
| Joint accuracy, angles <1 / <20 degrees (%) | 0.029 / 1.3 | 0.029 / 1.3 | 1.8 / 8.1 |
| Magnitude error <0.01 eV/A (%) | 1.8 | 1.8 | 8.4 |
| Magnitude error >1 eV/A (%) | 18.50 | 18.50 | 0.76 |
| Magnitude MAE / RMSE on FE atoms (eV/A) | 0.90 / 4.46 | 0.90 / 4.46 | 0.15 / 0.35 |
| Angle MAE / RMSE on FE atoms (degrees) | 9 / 15 | 9 / 15 | 4 / 6 |

## Ion migration

All 154 reference Li/Na pathways are evaluated with seven images per pathway. Full model NEB uses fixed-cell BFGS endpoints (0.002 eV/A, 500 steps), five intermediate climbing images (0.05 eV/A, 1,000 steps), and upstream periodic prealignment/autosort rules. Static model calculations use the original 1,078 DFT images. Barrier errors pool forward and backward directions. Non-converged paths are retained and handled by the upstream metric functions.

| Metric | Published CHGNet | Fresh CHGNet | MACE-OMAT-0 Small |
| --- | --- | --- | --- |
| Non-converged paths | 2 | 2 | 2 |
| Total paths | 154 | 154 | 154 |
| Full NEB barrier MAE (eV) | 0.1161 | 0.1163 | 0.0558 |
| Full NEB barrier RMSE (eV) | 0.157 | 0.1574 | 0.0978 |
| Static barrier MAE (eV) | 0.1124 | 0.1124 | 0.0707 |
| Static barrier RMSE (eV) | 0.1523 | 0.1523 | 0.1277 |
| Endpoint-energy difference MAE (eV) | 0.0358 | 0.0368 | 0.0274 |
| Endpoint-energy difference RMSE (eV) | 0.069 | 0.0707 | 0.0595 |
| Endpoint ranking agreement (%) | 78.2895 | 78.9474 | 83.5526 |
| Energy-profile shape agreement (%) | 93.4211 | 94.0789 | 92.7632 |
| Converged paths used for endpoint RMSD | 152 | 152 | 152 |
| Endpoint comparisons | 304 | 304 | 304 |
| Endpoint mapping success (%) | 100.0 | 100.0 | 100.0 |
| Mean endpoint RMSD (A) | 0.0783 | 0.0792 | 0.0554 |
| Maximum endpoint RMSD (A) | 0.6584 | 0.8009 | 0.7397 |
| Endpoint RMSD <0.05 A (%) | 54.6053 | 54.6053 | 72.3684 |
| Endpoint RMSD <0.10 A (%) | 78.2895 | 77.9605 | 87.5 |
| Endpoint RMSD <0.20 A (%) | 91.4474 | 91.4474 | 96.0526 |

Additional DFT calculations on model-generated NEB paths were not run. The repository explicitly marks this diagnostic optional. The supplied DFT reference data support the requested reproduction and new-potential comparisons; VASP is not required for this request.

## Reproducibility and limitations

CHGNet weights, float32 precision, legacy 6/3 A graphs and convergence criteria were held fixed. GPU inference retains original derivative math. CPU NEB disables unused higher-derivative graph construction after exact raw-output checks and one full trajectory comparison. An empty-neighbor CHGNet adjacency schema was reshaped from (0,) to (0,2), without adding edges; the actual failing input and ordinary predictions were validated. GPU allocator caps/cache cleanup and bounded graph prefetch alter resource management only. A singleton GPU-memory failure uses disk storage for large autograd saved tensors, retaining CUDA arithmetic. Bitwise checks passed under deterministic CUDA; default CUDA repeat variability was recorded separately. The actual failing 216-atom input returned finite outputs with this storage route.

MACE uses the official checkpoint, omat_pbe head, 6 A cutoff, float64 and no added dispersion. Phase/ordering, static NEB and OMat24 forces use CPU; MatPES forces use CUDA. Full NEB uses CPU and validated CUDA, with actual device/package provenance retained per path. Two complete CPU/CUDA NEB comparisons matched all convergence flags and step counts; maximum energy/force differences were below 3e-12 eV / 4e-10 eV/A. This validation covers two paths. Relaxation/NEB use direct ASE calculator calls. Force batches reuse official graph conversion and model arguments; a 33-structure comparison with direct ASE calls found force differences below 6e-14 eV/A. CPU/CUDA force checks covered 65 pilot structures and the actual failed 128-structure shard; energy/force/stress differences were below 3e-13. Direct or TorchScript-wrapped CUDA allocation failures split only failed batches, retaining existing predictions. Scientific precision and thresholds are unchanged.

The model documentation identifies OMat24 as MACE-OMAT-0 training data. Exact structure-level training overlap across these benchmarks has not been audited. These results describe performance against the supplied reference labels, not independently established out-of-distribution performance. MACE-OMAT-0 targets PBE/PBE+U labels; MatPES-PBE is a PBE reference, so reference-level differences can affect comparisons.

Source repository: https://github.com/mogroupumd/FP-DeErr, pinned at 0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815. Additional-model source: https://github.com/ACEsuit/mace-foundations (MACE-OMAT-0 Small). Source dataset and checkpoint SHA-256 hashes, stage package versions, standardized outputs and raw result locations are listed in the packet manifest. The original repository remains unmodified.
