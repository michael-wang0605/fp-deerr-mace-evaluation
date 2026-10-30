# Generated isolated adapter; see adapter_manifest.json.
"""Validate float64 force batching against direct official ASE calculations."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import itertools
import json
import os
import time
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress
from pymatgen.core import Structure
import reproduce
import force_benchmark_runner as sources
import new_potential_static as mace
from mace_batch_inference import predict
from relaxation_runner import atomic_json

def main():
    manifest = json.loads((reproduce.ROOT / 'datasets/manifest.json').read_text())
    mace.initialize()
    rows = list(itertools.islice(sources.structures('matpes_pbe', manifest['matpes_pbe']['path']), 16))
    rows += list(itertools.islice(sources.structures('omat24', manifest['omat24']['path']), 16))
    rows += list(itertools.islice(sources.structures('matpes_pbe', manifest['matpes_pbe']['path'], start=201762), 1))
    atoms = [mace.ADAPTOR.get_atoms(Structure.from_dict(row['structure'])) for row in rows]
    original = []
    begun = time.perf_counter()
    for item in atoms:
        item.calc = mace.CALC
        original.append({'energy_total': float(item.get_potential_energy()), 'forces': item.get_forces().copy(), 'stress': voigt_6_to_full_3x3_stress(item.get_stress())})
    single_seconds = time.perf_counter() - begun
    trials = []
    for size in [8, 16, 32]:
        begun = time.perf_counter()
        batched = predict(mace.CALC, atoms, batch_size=size)
        seconds = time.perf_counter() - begun
        differences = {name: max((float(np.max(np.abs(np.asarray(a[name]) - np.asarray(b[name])))) for (a, b) in zip(original, batched))) for name in ['energy_total', 'forces', 'stress']}
        assert differences['energy_total'] < 1e-09 and differences['forces'] < 1e-09 and (differences['stress'] < 1e-10)
        trials.append({'batch_size': size, 'seconds': seconds, 'maximum_absolute_differences': differences})
    report = {'validation_pass': True, 'weights_sha256': mace.MODEL_SHA, 'implementation_sha256': reproduce.sha(reproduce.ROOT / 'outputs/round2/adapters_v2/mace_mh_1_omat_pbe/mace_batch_inference.py'), 'dtype': 'float64', 'device': 'cpu', 'structures': len(rows), 'includes_actual_empty_neighbor_structure': True, 'reference': 'MACECalculator ASE energies, forces and stresses', 'single_seconds': single_seconds, 'trials': trials, 'tolerances': {'energy_total_eV': 1e-09, 'forces_eV_per_angstrom': 1e-09, 'stress_eV_per_angstrom3': 1e-10}}
    atomic_json(reproduce.OUT / 'round2/mace_mh_1_omat_pbe/validation/mace_batch_validation.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
