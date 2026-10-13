"""Run the unchanged notebook summary on complete, checksum-verified force shards."""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
import reproduce
from relaxation_runner import atomic_json


def summary_from_arrays(dataset, arrays, model_label):
    """Run the original notebook cell without writing any output files."""
    finite = np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    standard = {key:value[finite] for key,value in arrays.items()}
    notebook_path = reproduce.REPO/'Force_error/analysis'/(
        'force_error_analysis_matpes_pbe.ipynb' if dataset=='matpes_pbe' else 'force_error_analysis_omat24_rattled_1000.ipynb')
    nb = json.loads(notebook_path.read_text(encoding='utf-8'))
    cell_index,cell = next((i,c) for i,c in enumerate(nb['cells'])
        if c['cell_type']=='code' and 'COL_FMT =' in ''.join(c['source']) and 'rows[model]' in ''.join(c['source']))
    sys.path.insert(0,str(reproduce.REPO/'Force_error/scripts'))
    compatible = {'F_dft':standard['dft_force_magnitude'],'F_fp':standard['fp_force_magnitude'],
        'deltaF':np.abs(standard['force_magnitude_error']),'deltaTheta':standard['force_angle_error'],
        'e_vec':standard['force_vector_error']}
    namespace = {'all_results_filtered':{model_label:compatible},'FDFT_MIN':.01}
    exec(compile(''.join(cell['source']),str(notebook_path),'exec'),namespace)
    return namespace['df_summary'],nb,cell_index,notebook_path,finite,standard


def analyze(dataset, device, allow_partial=False, run_name=None, model_label='CHGNet', compare_published=True):
    folder = reproduce.OUT / 'force_benchmarks' / dataset / (run_name or device)
    progress = json.loads((folder / 'progress.json').read_text(encoding='utf-8'))
    if not allow_partial:
        assert progress['full_dataset_complete'], 'Full dataset has not finished'
    shard_paths = sorted((folder / 'shards').glob('*.npz'))
    fields = ['dft_force_magnitude', 'fp_force_magnitude', 'force_magnitude_error', 'force_angle_error', 'force_vector_error']
    arrays = {key: [] for key in fields}
    expected_index = 0
    atoms = 0
    for path in shard_paths:
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert metadata['sha256'] == reproduce.sha(path)
        assert metadata['first'] == expected_index
        with np.load(path, allow_pickle=False) as shard:
            assert np.array_equal(shard['structure_indices'], np.arange(metadata['first'], metadata['last'] + 1))
            assert np.isfinite(shard['dft_cartesian']).all() and np.isfinite(shard['fp_cartesian']).all()
            assert shard['dft_cartesian'].shape == shard['fp_cartesian'].shape == (metadata['atoms'], 3)
            for key in fields:
                assert len(shard[key]) == metadata['atoms']
                arrays[key].append(shard[key].copy())
        expected_index = metadata['last'] + 1
        atoms += metadata['atoms']
    assert expected_index == progress['completed_structures'] and atoms == progress['completed_atoms']
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
    table,nb,cell_index,notebook_path,finite,standard = summary_from_arrays(dataset,arrays,model_label)
    table.to_csv(folder / 'summary_metrics.csv')
    comparison = reproduce.compare_saved(dataset + ' force summary', table, cell_index, nb) if progress['full_dataset_complete'] and compare_published else None
    result = {'dataset': dataset, 'device': device, 'full_dataset_complete': progress['full_dataset_complete'],
        'structures': expected_index, 'atoms': atoms, 'finite_angle_atoms': int(finite.sum()),
        'atoms_above_DFT_force_0_01': int((standard['dft_force_magnitude'] > 0.01).sum()),
        'undefined_angle_atoms': int((~finite).sum()), 'comparison': comparison,
        'notebook': str(notebook_path), 'summary_cell': cell_index,
        'provenance': json.loads((folder / 'provenance.json').read_text(encoding='utf-8')),
        'model': model_label, 'summary_metrics': table.loc[model_label].to_dict()}
    if (folder/'execution_history.json').exists():
        result['execution_history']=json.loads((folder/'execution_history.json').read_text())
    atomic_json(folder / 'summary.json', result)
    text = ['# '+model_label+' force benchmark: ' + dataset, '',
        f"Fresh {device} inference on {expected_index:,} structures and {atoms:,} atoms.", '',
        'Full official dataset evaluated.' if progress['full_dataset_complete'] else '**Partial diagnostic only; this does not reproduce the full benchmark.**', '',
        'The upstream summary cell and Cartesian force conversion were run unchanged. Undefined force angles were excluded, as in the official generator; the analysis uses the strict DFT-force cutoff > 0.01 eV/Å.', '',
        '| Metric | Fresh result |', '|---|---|']
    text += [f'| {key} | {value} |' for key, value in table.loc[model_label].items()]
    if comparison:
        text += ['', f"Published displayed cells reproduced: {comparison['matching_cells']}/{comparison['total_cells']}."]
    text += ['', 'Raw Cartesian forces, structure IDs, atom offsets, energies and unfiltered atom metrics are retained in checksum-verified NPZ shards. This report does not establish dataset independence from model training data.']
    (folder / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['matpes_pbe', 'omat24'])
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    result = analyze(args.dataset, args.device, args.allow_partial, args.run_name)
    print(json.dumps({key: result[key] for key in ['dataset', 'structures', 'atoms', 'full_dataset_complete', 'comparison']}, indent=2))
