# Additional-model leaderboard contribution

All twelve additional scientific stages and both fresh deep-source model audits have passed.

The tables compare the same reference populations and unchanged public metrics. Static and full-relaxation protocols remain separate. Non-converged paths remain in the saved results and counts.

Training exposure is documented per checkpoint. These results do not establish unseen-data performance.

The selected material heads target PBE/PBE+U labels, while MatPES-PBE supplies PBE labels. Reference-functional differences can affect the comparisons; r2SCAN heads/results are not mixed into these tables.

**Status: verified results prepared for review; not yet merged into the upstream leaderboard.**

## static comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Phase stability energy MAE (meV/atom) | 246 | 62 | 57 | 48 |
| Ground-state agreement (%) | 55.9 | 60.9 | 64.6 | 75.8 |
| Within-phase agreement (%) | 48.8 | 66.9 | 70.2 | 75.2 |
| Global hull-minimum agreement (%) | 22.7 | 22.7 | 27.3 | 50.0 |
| Ordering energy MAE (meV/atom) | 270 | 51 | 46 | 38 |
| Top-1 accuracy (%) | 20.3 | 42.0 | 46.9 | 56.1 |
| Recall@3 (%) | 29.7 | 52.3 | 55.1 | 58.0 |
| Recall@10 (%) | 59.0 | 71.3 | 74.2 | 75.8 |
| Spearman correlation | 0.24 | 0.54 | 0.59 | 0.63 |
| Ranking-error rate (%) | 40.6 | 27.9 | 25.5 | 23.3 |
| Mean / max DFT energy penalty (meV/atom) | 10 / 29 | 6 / 16 | 5 / 15 | 5 / 13 |

## relaxed phase comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Energy MAE (meV/atom) | 241 | 97 | 95 | 87 |
| Ground-state agreement (%) | 53.4 | 60.2 | 59.0 | 67.1 |
| Within-phase hull-minimum agreement (%) | 39.7 | 59.5 | 54.5 | 60.3 |
| Global hull-minimum agreement (%) | 18.2 | 27.3 | 27.3 | 40.9 |
| Structure mapping success (%) | 87.4 | 89.4 | 89.6 | 89.9 |
| Mean / max RMSD (A) | 0.37 / 2.28 | 0.27 / 2.45 | 0.27 / 2.61 | 0.27 / 2.46 |
| RMSD < 0.05 A (%) | 11.9 | 22.5 | 20.4 | 25.3 |
| RMSD < 0.1 A (%) | 26.4 | 43.3 | 44.3 | 45.8 |
| RMSD < 0.2 A (%) | 49.8 | 64.4 | 66.7 | 64.4 |

## relaxed ordering comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Ordering energy MAE (meV/atom) | 271 | 76 | 76 | 68 |
| Top-1 accuracy (%) | 7.5 | 24.9 | 22.6 | 22.3 |
| Recall@3 (%) | 20.3 | 34.0 | 33.4 | 34.3 |
| Recall@10 (%) | 52.6 | 60.3 | 61.5 | 62.0 |
| Spearman correlation | 0.08 | 0.28 | 0.3 | 0.31 |
| Ranking-error rate (%) | 46.8 | 39.3 | 38.4 | 37.8 |
| Mean / max DFT energy penalty (meV/atom) | 13 / 39 | 12 / 35 | 12 / 35 | 11 / 35 |

## matpes pbe force comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Force-magnitude MAE / RMSE (eV/A) | 0.31 / 2.19 | 0.20 / 0.98 | 0.22 / 1.11 | 0.15 / 1.10 |
| Force-angle MAE / RMSE (degrees) | 45 / 66 | 34 / 56 | 33 / 55 | 28 / 50 |
| Magnitude MAE / RMSE, magnitude error <1 eV/A | 0.22 / 0.32 | 0.16 / 0.25 | 0.15 / 0.24 | 0.13 / 0.21 |
| Joint accuracy, angles <1 / <20 degrees (%) | 0.66 / 2.2 | 1.2 / 5.7 | 1.4 / 6.7 | 2.2 / 9.3 |
| Magnitude error <0.01 eV/A (%) | 5.6 | 8.8 | 9.9 | 13.0 |
| Magnitude error >1 eV/A (%) | 5.71 | 2.45 | 2.57 | 1.52 |
| Magnitude MAE / RMSE on FE atoms (eV/A) | 0.70 / 4.14 | 0.38 / 1.83 | 0.45 / 2.07 | 0.29 / 2.07 |
| Angle MAE / RMSE on FE atoms (degrees) | 24 / 36 | 15 / 25 | 14 / 26 | 11 / 21 |

## omat24 force comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Force-magnitude MAE / RMSE (eV/A) | 0.76 / 4.00 | 0.14 / 0.32 | 0.12 / 0.30 | 0.20 / 0.83 |
| Force-angle MAE / RMSE (degrees) | 13 / 22 | 5 / 10 | 5 / 9 | 5 / 10 |
| Magnitude MAE / RMSE, magnitude error <1 eV/A | 0.34 / 0.42 | 0.12 / 0.19 | 0.11 / 0.17 | 0.14 / 0.22 |
| Joint accuracy, angles <1 / <20 degrees (%) | 0.029 / 1.3 | 1.8 / 8.1 | 2.3 / 9.0 | 1.4 / 7.3 |
| Magnitude error <0.01 eV/A (%) | 1.8 | 8.4 | 9.3 | 7.5 |
| Magnitude error >1 eV/A (%) | 18.50 | 0.76 | 0.65 | 2.56 |
| Magnitude MAE / RMSE on FE atoms (eV/A) | 0.90 / 4.46 | 0.15 / 0.35 | 0.14 / 0.33 | 0.23 / 0.92 |
| Angle MAE / RMSE on FE atoms (degrees) | 9 / 15 | 4 / 6 | 3 / 6 | 4 / 7 |

## neb comparison

| Metric | Fresh CHGNet | MACE-OMAT-0 Small | MACE-OMAT-0 Medium | MACE-MH-1 (omat_pbe) |
| --- | --- | --- | --- | --- |
| Non-converged paths | 2 | 2 | 2 | 1 |
| Total paths | 154 | 154 | 154 | 154 |
| Full NEB barrier MAE (eV) | 0.1163 | 0.0558 | 0.0541 | 0.0542 |
| Full NEB barrier RMSE (eV) | 0.1574 | 0.0978 | 0.0981 | 0.0938 |
| Static barrier MAE (eV) | 0.1124 | 0.0707 | 0.0685 | 0.066 |
| Static barrier RMSE (eV) | 0.1523 | 0.1277 | 0.1257 | 0.116 |
| Endpoint-energy difference MAE (eV) | 0.0368 | 0.0274 | 0.0283 | 0.0271 |
| Endpoint-energy difference RMSE (eV) | 0.0707 | 0.0595 | 0.0697 | 0.0649 |
| Endpoint ranking agreement (%) | 78.9474 | 83.5526 | 87.5 | 86.9281 |
| Energy-profile shape agreement (%) | 94.0789 | 92.7632 | 94.0789 | 95.4248 |
| Converged paths used for endpoint RMSD | 152 | 152 | 152 | 153 |
| Endpoint comparisons | 304 | 304 | 304 | 306 |
| Endpoint mapping success (%) | 100.0 | 100.0 | 100.0 | 100.0 |
| Mean endpoint RMSD (A) | 0.0792 | 0.0554 | 0.0542 | 0.0479 |
| Maximum endpoint RMSD (A) | 0.8009 | 0.7397 | 0.9631 | 0.738 |
| Endpoint RMSD <0.05 A (%) | 54.6053 | 72.3684 | 80.5921 | 80.3922 |
| Endpoint RMSD <0.10 A (%) | 77.9605 | 87.5 | 89.4737 | 91.1765 |
| Endpoint RMSD <0.20 A (%) | 91.4474 | 96.0526 | 96.7105 | 96.732 |

## Exact checkpoints

- MACE-OMAT-0 Medium: [mace-omat-0-medium.model](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model), head `default`, SHA-256 `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.
  OMat24 training data; structure-level overlap is not audited.
  Full relaxation execution counts: {'cpu': 5821, 'cuda': 876}. CPU and CUDA used float64, the same ASE FIRE version, fixed cells, 0.01 eV/Angstrom force tolerance and 10,000-step limit. Complete validation trajectories matched optimization step counts; actual execution profiles are included.
- MACE-MH-1 (omat_pbe): [mace-mh-1.model](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model), head `omat_pbe`, SHA-256 `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`.
  OMat pretraining and cross-domain fine-tuning including MatPES-related data. No unseen-data claim; exact structure-level overlap remains unaudited.
  Full relaxation execution counts: {'cpu': 5082, 'cuda': 1615}. CPU and CUDA used float64, the same ASE FIRE version, fixed cells, 0.01 eV/Angstrom force tolerance and 10,000-step limit. Complete validation trajectories matched optimization step counts; actual execution profiles are included.
  matpes_pbe: 1 singleton CUDA memory failures used validated saved-tensor disk storage; all arithmetic remained float64 CUDA. Structure indices and storage statistics are retained in the audit and raw shard metadata.

Original CHGNet and MACE-Small benchmark outputs are retained unchanged. See the original report for reproduction discrepancies. The upstream contribution guide requests exact model/checkpoint identity and official source links. Existing rows must remain unchanged when these entries are integrated.
