# Reproduction setup

The results in this repository are completed snapshots. Running the calculation scripts regenerates predictions in `outputs/`; the full source audit needs these raw predictions, not just the portable summary files in `results/`.

## Source and environments

Use Python 3.10. Clone the upstream repository into `FP-DeErr/` and pin its commit:

```powershell
git clone https://github.com/mogroupumd/FP-DeErr.git FP-DeErr
git -C FP-DeErr checkout 0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815
```

Create separate environments and install each matching requirements file:

| Environment | Requirements | Purpose |
| --- | --- | --- |
| `.venv` | `requirements-reproduction.txt` | CHGNet static/relaxation and analysis |
| `.venv-force` | `requirements-force.txt` | CHGNet CUDA forces |
| `.venv-neb` | `requirements-neb.txt` + `requirements-mace-cuda.txt` | CHGNet NEB and recorded MACE CUDA environment |
| `.venv-mace` | `requirements-mace.txt` | MACE CPU static/relaxation/forces |
| `.venv-mace-neb` | `requirements-mace-neb.txt` | MACE CPU NEB |

For example, `py -3.10 -m venv .venv`, followed by `.\.venv\Scripts\python.exe -m pip install -r requirements-reproduction.txt`. CUDA requirements pin Torch 2.7.1+cu126; choose a compatible GPU/driver. Package versions actually used per stage are retained in `results/stages/`.

## Checkpoints and force inputs

Download the official [CHGNet 0.3.0 checkpoint](https://raw.githubusercontent.com/CederGroupHub/chgnet/v0.3.0/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) into each CHGNet environment's `Lib/site-packages/chgnet/pretrained/0.3.0/` if the wheel omits it. Expected SHA-256: `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.

Download [MACE-OMAT-0 Small](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model) to `models/mace-omat-0-small.model`. Expected SHA-256: `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`.

Download [MatPES-PBE-2025.1.jsonl](https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-PBE-2025.1.jsonl) to `datasets/MatPES-PBE-2025.1.jsonl`, and the [OMat24 validation rattled-1000 archive](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz) to `datasets/OMat24-val-rattled-1000.tar.gz`. `prepare_force_datasets.py` verifies the MatPES checksum, extracts the selected LMDB population and records source checksums and counts.

## Execute and compare

Follow [REPRODUCING.md](REPRODUCING.md). Static, full-relaxation and full-NEB protocols are separate. The fresh reproduction route uses CPU for MACE NEB; the recorded run additionally used validated CUDA on 71 paths and CPU on 83. The mixed-run process-control configuration belongs to the retained original run and is not needed for a fresh CPU reproduction. Do not start multiple writers against the same output folder.

The supplied reference geometries are in the pinned upstream repository. All force populations and all 154 NEB paths must complete before comparing aggregate results. Preserve non-converged paths and inspect failures. VASP is not required.

See the [report methods](reports/report.md) for graph handling, exact convergence criteria, float32 CHGNet/float64 MACE precision, execution adaptations and the limits of their validation. The [evidence index](results/evidence_index.json) preserves original audited source hashes and portable review-copy hashes. It does not replace access to the retained raw results for a new deep-source audit.
