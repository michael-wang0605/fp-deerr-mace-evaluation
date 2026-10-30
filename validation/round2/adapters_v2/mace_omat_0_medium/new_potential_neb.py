# Generated isolated adapter; see adapter_manifest.json.
"""Execute both public model-only NEB protocols with MACE-OMAT-0 Medium."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import reproduce
import relaxation_runner as relaxation
import neb_full_runner as full
import neb_static_runner as static
import new_potential_static as mace
KEY = 'MACE_OMAT_0_MEDIUM'
ROOT = mace.OUT / 'neb'
REFERENCE = reproduce.REPO / 'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz'

def folder(mode):
    return ROOT / mode

def initialize(mode):
    mace.initialize()
    (relaxation.CALC, relaxation.ADAPTOR) = (mace.CALC, mace.ADAPTOR)
    (full.CALC, full.OUT) = (mace.CALC, folder('full'))
    full.EXECUTION = {'device': 'cpu', 'dtype': 'float64', 'inference_optimized': False}
    static.OUT = folder('static')

def record_path(mode, key):
    if mode == 'full':
        return folder(mode) / 'records' / (key.replace('|', '_') + '.json')
    return folder(mode) / (hashlib.sha256(key.encode()).hexdigest() + '.json')

def provenance(mode):
    names = ['mace-torch', 'torch', 'e3nn', 'numpy', 'ase', 'pymatgen', 'matcalc', 'phonopy', 'phono3py']
    return {'reference_sha256': reproduce.sha(REFERENCE), 'weights_sha256': mace.MODEL_SHA, 'repository_commit': '0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815', 'fp_key': KEY, 'registry_key': mace.KEY, 'model': 'MACE-OMAT-0 Medium', 'protocol': 'full_fp_neb' if mode == 'full' else 'fp_static_on_dft_neb', 'device': 'cpu', 'dtype': 'float64', 'head': 'default', 'cutoff_angstrom': 6.0, 'dispersion_correction': False, 'threads': 1, 'template_sha256': hashlib.sha256(full.template().encode()).hexdigest() if mode == 'full' else None, 'versions': {name: importlib.metadata.version(name) for name in names}}

def run(args):
    assert reproduce.sha(mace.MODEL) == mace.MODEL_SHA
    reference = reproduce.load(REFERENCE)
    expected = provenance(args.mode)
    jobs = list(reference['pathways'].items())
    if args.limit:
        jobs = jobs[:args.limit]
    (completed, pending) = ({}, [])
    for job in jobs:
        path = record_path(args.mode, job[0])
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['provenance'] == expected
            completed[job[0]] = saved['result']
        else:
            pending.append(job)
    relaxation.atomic_json(folder(args.mode) / 'plan.json', {'provenance': expected, 'pathways': [k for (k, _) in jobs], 'workers': args.workers})
    print(f'MACE NEB {args.mode}: {len(jobs)} planned, {len(completed)} cached, {len(pending)} pending', flush=True)
    begun = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.mode,)) as pool:
        function = full.compute if args.mode == 'full' else static.compute
        futures = {}
        pending = iter(pending)
        exhausted = False
        while futures or not exhausted:
            cap_path = folder(args.mode) / 'active_paths.json'
            cap = args.active_paths or args.workers
            if cap_path.exists():
                cap = json.loads(cap_path.read_text())['active_paths']
            assert 1 <= cap <= args.workers
            while len(futures) < cap and (not exhausted):
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                futures[pool.submit(function, job, expected)] = job[0]
            if not futures:
                break
            (finished, _) = wait(futures, return_when=FIRST_COMPLETED, timeout=10)
            for future in finished:
                collect(future, futures, completed, args.mode, len(jobs), begun)
    relaxation.atomic_json(folder(args.mode) / 'run_summary.json', {'provenance': expected, 'completed': len(completed), 'planned': len(jobs)})
    if not args.limit:
        export(args.mode)

def collect(future, futures, completed, mode, planned, begun):
    future.result()
    key = futures.pop(future)
    completed[key] = json.loads(record_path(mode, key).read_text())['result']
    progress = {'completed': len(completed), 'planned': planned, 'elapsed_seconds': time.perf_counter() - begun, 'converged': sum((r['neb_status']['neb_converged'] is True for r in completed.values())) if mode == 'full' else None, 'failed': sum((r['neb_status']['calculation_status'] == 'failed' for r in completed.values())) if mode == 'full' else sum((bool(r['image_errors']) for r in completed.values())), 'last_pathway': key}
    relaxation.atomic_json(folder(mode) / 'progress.json', progress)
    print(json.dumps(progress), flush=True)

def export(mode):
    reference = reproduce.load(REFERENCE)
    expected = provenance(mode)
    (successful, failed) = ({}, {})
    for (key, pathway) in reference['pathways'].items():
        saved = json.loads(record_path(mode, key).read_text())
        assert saved['provenance'] == expected
        result = saved['result']
        if mode == 'full':
            if result['neb_status']['calculation_status'] == 'failed':
                failed[key] = {'identifiers': pathway['identifiers'], 'calculation_status': 'failed', 'error': result['neb_status']['error']}
            else:
                successful[key] = {name: result[name] for name in ['identifiers', 'final_fp_neb_images', 'neb_status', 'calculation_metadata']}
        else:
            assert not result['image_errors'], (key, result['image_errors'])
            successful[key] = result
    assert len(successful) + len(failed) == 154
    protocol = 'full_fp_neb' if mode == 'full' else 'fp_static_on_dft_neb'
    output = {name: reference[name] for name in ['schema_version', 'component', 'units']}
    output.update(dataset_name='ion_migration_neb_results', provenance=expected, models={KEY: {protocol: {'pathways': successful, 'unsuccessful_pathways': failed}}})
    relaxation.atomic_json(folder(mode) / 'results.json', output)
    if mode == 'static':
        analyze_static()
    if (folder('full') / 'results.json').exists() and (folder('static') / 'results.json').exists():
        analyze()

def analyze_static():
    """Analyze the complete static protocol without claiming full NEB completion."""
    import numpy as np
    import sys
    sys.path.insert(0, str(reproduce.REPO / 'Ion_migration_NEB/scripts'))
    import neb_analysis as na
    reference = reproduce.load(REFERENCE)
    results = json.loads((folder('static') / 'results.json').read_text())
    records = results['models'][KEY]['fp_static_on_dft_neb']['pathways']
    assert set(records) == set(reference['pathways']) and len(records) == 154
    images = 0
    for (identifier, pathway) in reference['pathways'].items():
        saved = json.loads(record_path('static', identifier).read_text())
        assert saved['provenance'] == provenance('static')
        assert not records[identifier]['image_errors']
        for source in pathway['dft_neb_reference']['images']:
            image = records[identifier]['images'][str(source['image_index'])]
            forces = np.asarray(image['fp_forces_eV_per_angstrom'])
            assert forces.shape == (len(source['structure']['sites']), 3) and np.isfinite(forces).all()
            assert np.isfinite(image['fp_energy_total_eV'])
            images += 1
    assert images == 1078
    built = na.build_neb_analysis_results(reference, results, expected_pathways=154)
    assert built.validation_report.ok and built.coverage_mismatch_df.empty
    protocol = 'fp_static_on_dft_neb'
    table = built.barrier_error_summary_by_protocol[protocol]
    row = table.loc[KEY]
    errors = np.concatenate([row['_errors_energy_forward_barrier'], row['_errors_energy_backward_barrier']])
    assert len(errors) == 308 and np.isfinite(errors).all()
    metrics = {'barrier_mae_static_eV': na.round_half_away_from_zero(float(np.abs(errors).mean()), 4), 'barrier_rmse_static_eV': na.round_half_away_from_zero(float(np.sqrt(np.mean(errors ** 2))), 4)}
    table.to_csv(folder('static') / 'barrier_metrics.csv')
    built.protocol_coverage_df.to_csv(folder('static') / 'coverage.csv', index=False)
    built.fp_path_metrics_by_protocol[KEY][protocol].to_csv(folder('static') / 'path_metrics.csv', index=False)
    built.dft_neb_path_force_errors_df.to_csv(folder('static') / 'force_errors.csv', index=False)
    summary = {'pathways': 154, 'images': 1078, 'validation_pass': True, 'failed': 0, 'pooled_barrier_errors': 308, 'metrics': metrics, 'provenance': provenance('static'), 'meaning': 'Complete static-on-DFT-NEB protocol only; full model NEB is separate.'}
    relaxation.atomic_json(folder('static') / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

def analyze():
    import numpy as np
    import sys
    full_results = json.loads((folder('full') / 'results.json').read_text())
    static_results = json.loads((folder('static') / 'results.json').read_text())
    full_results['models'][KEY].update(static_results['models'][KEY])
    assert not full_results['models'][KEY]['full_fp_neb']['unsuccessful_pathways']
    reference = reproduce.load(REFERENCE)
    for (key, record) in full_results['models'][KEY]['full_fp_neb']['pathways'].items():
        assert set(record['final_fp_neb_images']) == {str(i) for i in range(7)}
        source_cell = reference['pathways'][key]['full_fp_neb_input']['source_start_endpoint_structure']['lattice']['matrix']
        for image in record['final_fp_neb_images'].values():
            force = np.asarray(image['fp_forces_eV_per_angstrom'])
            assert force.shape == (len(image['structure']['sites']), 3) and np.isfinite(force).all()
            assert np.isfinite(image['fp_energy_total_eV'])
            assert np.allclose(image['structure']['lattice']['matrix'], source_cell, rtol=0, atol=1e-08)
    for (key, record) in full_results['models'][KEY]['fp_static_on_dft_neb']['pathways'].items():
        for image in reference['pathways'][key]['dft_neb_reference']['images']:
            result = record['images'][str(image['image_index'])]
            forces = np.asarray(result['fp_forces_eV_per_angstrom'])
            assert forces.shape == (len(image['structure']['sites']), 3) and np.isfinite(forces).all()
            assert np.isfinite(result['fp_energy_total_eV'])
    relaxation.atomic_json(ROOT / 'results.json', full_results)
    sys.path.insert(0, str(reproduce.REPO / 'Ion_migration_NEB/scripts'))
    import neb_analysis as na
    from neb_plots import area_between_curves, simplify_class
    from export_neb_leaderboard import build_leaderboard
    import export_neb_leaderboard as display_registry
    display_registry.FP_DISPLAY_NAMES[KEY] = 'MACE-OMAT-0 Medium'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fresh = build_leaderboard(REFERENCE, ROOT / 'results.json', na, area_between_curves, simplify_class, fp_keys=[KEY])
    relaxation.atomic_json(ROOT / 'leaderboard.json', fresh)
    row = fresh['models'][0]
    summary = {'pathways': 154, 'static_images': 1078, 'full_images': 1078, 'validation_pass': True, 'converged': row['n_total'] - row['n_nonconverged'], 'non_converged': row['n_nonconverged'], 'failed': 0, 'metrics': row, 'provenance': {'full': provenance('full'), 'static': provenance('static')}, 'optional_dft_on_model_paths': 'not run; optional in public workflow'}
    relaxation.atomic_json(ROOT / 'summary.json', summary)
    baseline = json.loads((reproduce.OUT / 'neb_full_cpu/leaderboard.json').read_text())['models'][0]
    lines = ['# MACE-OMAT-0 Medium: full and static NEB evaluation', '', f"All 154 paths completed: {summary['converged']} converged, {summary['non_converged']} non-converged, no calculation failures.", '', 'Same public fixed-cell BFGS/MatCalc template and convergence settings as CHGNet; float64 CPU MACE, default head, cutoff 6 Å, no added dispersion.', '', '| Metric | Fresh CHGNet | MACE-OMAT-0 Medium |', '|---|---:|---:|']
    lines += [f'| {key} | {baseline.get(key)} | {value} |' for (key, value) in row.items() if key not in ['fp_key', 'display_name']]
    lines += ['', 'DFT reference paths are supplied by FP-DeErr. Optional new DFT evaluations on model-generated paths are not included. Training overlap is not fully audited.', '']
    (ROOT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'static'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--active-paths', type=int)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--analyze-static', action='store_true')
    args = parser.parse_args()
    analyze_static() if args.analyze_static else analyze() if args.analyze else run(args)
