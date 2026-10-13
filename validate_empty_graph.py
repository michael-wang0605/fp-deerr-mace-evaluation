"""Check the actual failing MatPES input and unchanged ordinary GPU predictions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import itertools
import json
import numpy as np
import torch
import force_benchmark_runner as runner
import reproduce
from chgnet.model.model import CHGNet
from chgnet.graph.crystalgraph import CrystalGraph
from graph_shape_adapter import normalize_empty_atom_graph
from relaxation_runner import atomic_json


def restore(packed):
    return CrystalGraph(**{key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                           for key, value in packed.items()})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.init_graph_worker()
    manifest = json.loads((reproduce.ROOT/'datasets/manifest.json').read_text())
    rows = list(itertools.islice(runner.structures('matpes_pbe', manifest['matpes_pbe']['path'], start=201728), 128))
    # Build directly so this test sees the original upstream malformed shape.
    from pymatgen.core import Structure
    graphs = [runner.CONVERTER(runner.ADAPTOR.get_structure(runner.ADAPTOR.get_atoms(Structure.from_dict(row['structure'])))) for row in rows]
    model = CHGNet.from_dict(torch.load(runner.WEIGHTS, map_location='cpu', weights_only=False)['model']).eval().to('cuda')
    ordinary = [g for g in graphs if g.atom_graph.numel()][:16]
    before = model.predict_graph(ordinary, task='efsm', return_crystal_feas=True, batch_size=16)
    changed = [normalize_empty_atom_graph(g) for g in ordinary]
    after = model.predict_graph(ordinary, task='efsm', return_crystal_feas=True, batch_size=16)
    fields = ['e', 'f', 's', 'm', 'crystal_fea']
    checks = {field: all(np.array_equal(a[field], b[field]) for a, b in zip(before, after)) for field in fields}
    assert not any(changed) and all(checks.values())
    exceptional = []
    for row, graph in zip(rows, graphs):
        if graph.atom_graph.numel():
            continue
        old_shape = list(graph.atom_graph.shape)
        assert normalize_empty_atom_graph(graph)
        prediction = model.predict_graph(graph, task='efsm', return_crystal_feas=True)
        assert np.isfinite(prediction['e']) and np.isfinite(prediction['f']).all()
        assert prediction['f'].shape == (len(row['forces']), 3)
        assert np.array_equal(prediction['f'], np.zeros_like(prediction['f']))
        assert np.array_equal(prediction['s'], np.zeros_like(prediction['s']))
        # Independently rebuild small +/- Cartesian displacements. There are
        # still no edges, and the energy must remain coordinate independent.
        perturbations = []
        original = runner.ADAPTOR.get_atoms(Structure.from_dict(row['structure']))
        for displacement in [-0.001, 0.001]:
            atoms = original.copy()
            atoms.positions[0, 0] += displacement
            displaced = runner.CONVERTER(runner.ADAPTOR.get_structure(atoms))
            assert displaced.atom_graph.numel() == 0
            normalize_empty_atom_graph(displaced)
            value = model.predict_graph(displaced, task='efsm', return_crystal_feas=True)
            assert np.array_equal(value['e'], prediction['e'])
            perturbations.append(float(value['e']))
        exceptional.append({'index': row['index'], 'source_id': row['source_id'], 'atoms': len(row['forces']),
            'original_edge_shape': old_shape, 'corrected_edge_shape': list(graph.atom_graph.shape),
            'energy_eV_per_atom': float(prediction['e']), 'all_forces_zero': True, 'all_stresses_zero': True,
            'displaced_energies_eV_per_atom': perturbations})
    assert len(exceptional) == 1 and exceptional[0]['index'] == 201762
    report = {'validation_pass': True, 'weights_sha256': reproduce.sha(runner.WEIGHTS),
        'adapter_sha256': reproduce.sha(reproduce.ROOT/'graph_shape_adapter.py'),
        'ordinary_graphs': len(ordinary), 'ordinary_predictions_exact': checks, 'exceptional_graphs': exceptional,
        'scope': 'Empty adjacency schema only; no edges, cutoff, model parameters, precision, or metric changes.'}
    atomic_json(reproduce.OUT/'empty_graph_validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
