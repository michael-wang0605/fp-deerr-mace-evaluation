# Generated isolated adapter; see adapter_manifest.json.
"""Compare identical float64 MACE inputs across CPU/CUDA and record timings."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import importlib.metadata
import itertools
import json
import time
import numpy as np
import torch
from mace.calculators import MACECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import force_benchmark_runner as sources
import new_potential_static as mace
from mace_batch_inference import predict
from relaxation_runner import atomic_json

def main(device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((mace.reproduce.ROOT / 'datasets/manifest.json').read_text())
    rows = []
    for dataset in ['matpes_pbe', 'omat24']:
        rows += list(itertools.islice(sources.structures(dataset, manifest[dataset]['path']), 32))
    rows += list(itertools.islice(sources.structures('matpes_pbe', manifest['matpes_pbe']['path'], start=201762), 1))
    inputs_sha = __import__('hashlib').sha256(json.dumps(rows, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value)).encode()).hexdigest()
    atoms = [AseAtomsAdaptor.get_atoms(Structure.from_dict(row['structure'])) for row in rows]
    assert mace.reproduce.sha(mace.MODEL) == mace.MODEL_SHA
    calc = MACECalculator(model_paths=str(mace.MODEL), device=device, default_dtype='float64', head='omat_pbe')
    predict(calc, atoms[:2], batch_size=2)
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = predict(calc, atoms, batch_size=16)
    if device == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    report = {'device': device, 'dtype': 'float64', 'structures': len(rows), 'seconds': seconds, 'structures_per_second': len(rows) / seconds, 'inputs_sha256': inputs_sha, 'weights_sha256': mace.MODEL_SHA, 'batch_size': 16, 'implementation_sha256': mace.reproduce.sha(mace.reproduce.ROOT / 'outputs/round2/adapters_v2/mace_mh_1_omat_pbe/mace_batch_inference.py'), 'versions': {name: importlib.metadata.version(name) for name in ['torch', 'mace-torch', 'e3nn', 'ase', 'pymatgen', 'numpy']}, 'predictions': [{key: np.asarray(value).tolist() for (key, value) in row.items()} for row in result]}
    if device == 'cuda':
        report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
        report['gpu'] = torch.cuda.get_device_name()
    atomic_json(mace.reproduce.OUT / f'round2/mace_mh_1_omat_pbe/validation/mace_device_profile_{device}.json', report)
    other_path = mace.reproduce.OUT / f"round2/mace_mh_1_omat_pbe/validation/mace_device_profile_{('cpu' if device == 'cuda' else 'cuda')}.json"
    if other_path.exists():
        other = json.loads(other_path.read_text())
        assert other['inputs_sha256'] == inputs_sha and other['weights_sha256'] == mace.MODEL_SHA
        differences = {}
        for key in ['energy_total', 'forces', 'stress']:
            differences[key] = max((float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))) for (a, b) in zip(report['predictions'], other['predictions'])))
        assert all((value < 1e-09 for value in differences.values())), differences
        atomic_json(mace.reproduce.OUT / 'round2/mace_mh_1_omat_pbe/validation/mace_cpu_cuda_validation.json', {'validation_pass': True, 'dtype': 'float64', 'structures': len(rows), 'inputs_sha256': inputs_sha, 'weights_sha256': mace.MODEL_SHA, 'implementation_sha256': report['implementation_sha256'], 'batch_size': 16, 'maximum_absolute_differences': differences, 'timings_seconds': {device: seconds, other['device']: other['seconds']}, 'versions': {device: report['versions'], other['device']: other['versions']}, 'scope': 'Identical supplied structures, including the empty-edge fixture; unchanged model, precision, cutoff, head, conversion and output definitions.'})
        print(json.dumps({'differences': differences, 'timings': {device: seconds, other['device']: other['seconds']}}), flush=True)
    print(json.dumps({key: value for (key, value) in report.items() if key != 'predictions'}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    main(parser.parse_args().device)
