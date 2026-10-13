# Reproduction commands

Run from the workspace root with Python 3.10. Environments and checkpoints are isolated; existing outputs are verified before resume. Do not start duplicate writers.

```powershell
.\.venv\Scripts\python.exe reproduce.py analysis
.\.venv\Scripts\python.exe reproduce.py static --scope all
.\.venv\Scripts\python.exe relaxation_runner.py --scope all --workers 3 --threads 1
.\.venv\Scripts\python.exe relaxation_analysis.py --scope hull
.\.venv\Scripts\python.exe relaxation_analysis.py --scope ordering
.\.venv\Scripts\python.exe prepare_force_datasets.py
.\.venv-neb\Scripts\python.exe validate_inference_optimization.py --device cpu
.\.venv\Scripts\python.exe neb_static_runner.py --workers 8
.\.venv-neb\Scripts\python.exe neb_full_runner.py --device cpu --workers 4 --inference-optimized
.\.venv\Scripts\python.exe neb_full_analysis.py --device cpu
.\.venv-force\Scripts\python.exe validate_gpu_memory_cleanup.py --memory-fraction 0.9
.\.venv-force\Scripts\python.exe validate_cuda_disk_offload.py
.\.venv-force\Scripts\python.exe verify_chgnet_offload_retry.py
.\.venv-force\Scripts\python.exe force_benchmark_runner.py --dataset matpes_pbe --device cuda --graph-workers 2 --prefetch-shards 2 --cuda-memory-fraction 0.9 --singleton-disk-offload
.\.venv-force\Scripts\python.exe force_benchmark_runner.py --dataset omat24 --device cuda --graph-workers 2
.\.venv\Scripts\python.exe force_benchmark_analysis.py --dataset matpes_pbe --device cuda
.\.venv\Scripts\python.exe force_benchmark_analysis.py --dataset omat24 --device cuda
.\.venv-mace\Scripts\python.exe new_potential_static.py --workers 3
.\.venv-mace\Scripts\python.exe new_potential_relaxation.py --workers 8
.\.venv-mace\Scripts\python.exe validate_mace_batches.py
.\.venv-mace\Scripts\python.exe new_potential_force.py --dataset omat24 --workers 4
.\.venv-mace\Scripts\python.exe profile_mace_device.py --device cpu
.\.venv-neb\Scripts\python.exe profile_mace_device.py --device cuda
.\.venv-neb\Scripts\python.exe validate_mace_cuda_partitions.py
.\.venv-neb\Scripts\python.exe verify_mace_retry_controls.py
.\.venv-neb\Scripts\python.exe new_potential_force.py --dataset matpes_pbe --device cuda --workers 1
.\.venv-mace-neb\Scripts\python.exe new_potential_neb.py --mode static --workers 2
.\.venv-mace-neb\Scripts\python.exe new_potential_neb.py --mode full --workers 8 --active-paths 8
.\.venv-mace-neb\Scripts\python.exe analyze_mace_neb.py
```

The commands above reproduce the scientific protocol in a fresh workspace using CPU for full MACE NEB. The recorded run also uses validated float64 CUDA. Its disjoint pending/external pathway lists are in `outputs/mace_neb_mixed_config.json`; per-path actual execution profiles are in the full run summary. Transition launch scripts record process handles from this run and must not be replayed on another host. Never launch the CPU reproduction queue against the active mixed production folder.

For the retained mixed outputs, after all 154 paths finish, use this metadata-preserving analysis command:

```powershell
.\.venv-mace-neb\Scripts\python.exe new_potential_neb_mixed.py --mode full --config outputs/mace_neb_mixed_config.json --analyze
.\.venv\Scripts\python.exe audit_research_results.py --deep-source
```

See the pinned requirements files and the workspace README for source downloads, checkpoint compatibility, graph controls and validation commands. Optional VASP diagnostics are not required.
