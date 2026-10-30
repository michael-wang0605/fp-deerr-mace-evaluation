# Generated isolated adapter; see adapter_manifest.json.
"""Validate smaller float64 CUDA batches before permitting OOM partitioning."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import profile_mace_device as profile
import hashlib
import itertools
import json
import numpy as np
import torch
from mace.calculators import MACECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace_batch_inference import predict
from relaxation_runner import atomic_json

def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.9)
    root = profile.mace.reproduce.ROOT
    manifest = json.loads((root / 'datasets/manifest.json').read_text())
    rows = []
    for dataset in ['matpes_pbe', 'omat24']:
        rows += list(itertools.islice(profile.sources.structures(dataset, manifest[dataset]['path']), 32))
    rows += list(itertools.islice(profile.sources.structures('matpes_pbe', manifest['matpes_pbe']['path'], start=201762), 1))
    inputs_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)).encode()).hexdigest()
    original = json.loads((root / 'outputs/round2/mace_omat_0_medium/validation/mace_device_profile_cpu.json').read_text())
    assert original['inputs_sha256'] == inputs_sha and original['weights_sha256'] == profile.mace.MODEL_SHA
    atoms = [AseAtomsAdaptor.get_atoms(Structure.from_dict(row['structure'])) for row in rows]
    calc = MACECalculator(model_paths=str(profile.mace.MODEL), device='cuda', default_dtype='float64', head='default')
    trials = []
    for size in [16, 8, 4, 2, 1]:
        results = predict(calc, atoms, batch_size=size)
        differences = {key: max((float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))) for (a, b) in zip(results, original['predictions']))) for key in ['energy_total', 'forces', 'stress']}
        assert len(results) == len(atoms) and all((np.isfinite(v) and v < 1e-09 for v in differences.values()))
        trials.append({'batch_size': size, 'maximum_absolute_differences': differences})
        print(json.dumps(trials[-1]), flush=True)
    atomic_json(root / 'outputs/round2/mace_omat_0_medium/validation/mace_cuda_partition_validation.json', {'validation_pass': True, 'dtype': 'float64', 'structures': len(atoms), 'inputs_sha256': inputs_sha, 'weights_sha256': profile.mace.MODEL_SHA, 'trials': trials, 'implementation_sha256': profile.mace.reproduce.sha(root / 'outputs/round2/adapters_v2/mace_omat_0_medium/mace_batch_inference.py'), 'scope': 'Contiguous CUDA partitions versus the identical saved float64 CPU predictions; no input skipped or precision changed.'})
if __name__ == '__main__':
    main()
