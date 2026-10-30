# Generated isolated adapter; see adapter_manifest.json.
"""Run the unchanged FP-DeErr fixed-cell FIRE workflow with MACE-MH-1 (omat_pbe)."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import time
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import reproduce
import relaxation_runner as base
import new_potential_static as mace
OUT = mace.OUT / 'relaxation'

def initialize():
    base.OUT = OUT
    base.RECORDS = OUT / 'records'
    mace.initialize()
    (base.CALC, base.ADAPTOR) = (mace.CALC, mace.ADAPTOR)
    namespace = {'calc': base.CALC, 'adaptor': base.ADAPTOR}
    exec(base.upstream_energy_source(), namespace)
    base.ENERGY = namespace['get_energy']

def planned():
    (jobs, reference_provenance) = base.plan_jobs('all')
    static_provenance = json.loads((mace.OUT / 'run_summary.json').read_text())['provenance']
    provenance = {**static_provenance, 'protocol': 'full fixed-cell relaxation from initial_structure', 'energy_func_sha256': hashlib.sha256(base.upstream_energy_source().encode()).hexdigest(), 'optimizer': 'ASE FIRE', 'fmax': 0.01, 'max_steps': 10000, 'fixed_cell': True}
    return ([(key, source, {}) for (key, source, _) in jobs], provenance)

def run(args):
    base.OUT = OUT
    base.RECORDS = OUT / 'records'
    (jobs, provenance) = planned()
    if args.limit:
        jobs = jobs[:args.limit]
    (complete, pending) = ({}, [])
    for job in jobs:
        path = base.record_path(job[0])
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['id'] == job[0] and saved['provenance'] == provenance
            complete[job[0]] = saved['result']
        else:
            pending.append(job)
    base.atomic_json(OUT / 'plan.json', {'provenance': provenance, 'jobs': [job[0] for job in jobs], 'workers': args.workers, 'threads_per_worker': 1})
    print(f'MACE RELAX: {len(jobs)} planned, {len(complete)} cached, {len(pending)} pending', flush=True)
    begun = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        futures = [pool.submit(base.compute, job, provenance) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            complete[row['id']] = json.loads(base.record_path(row['id']).read_text())['result']
            progress = {'completed': len(complete), 'planned': len(jobs), 'success': sum((r['status'] == 'success' for r in complete.values())), 'non_converged': sum((r['status'] == 'non_converged' for r in complete.values())), 'failed': sum((r['status'] == 'failed' for r in complete.values())), 'elapsed_seconds': time.perf_counter() - begun, 'last': row}
            base.atomic_json(OUT / 'progress.json', progress)
            if len(complete) <= 3 or len(complete) % 50 == 0 or row['status'] != 'success':
                print(json.dumps(progress), flush=True)
    base.atomic_json(OUT / 'run_summary.json', {'provenance': provenance, 'planned': len(jobs), 'completed': len(complete), 'workers': args.workers, 'threads_per_worker': 1})
    if not args.limit:
        analyze()

def analyze():
    base.OUT = OUT
    base.RECORDS = OUT / 'records'
    (jobs, provenance) = planned()
    records = {}
    for (key, source, _) in jobs:
        saved = json.loads(base.record_path(key).read_text())
        assert saved['id'] == key and saved['provenance'] == provenance
        records[key] = saved['result']
        result = records[key]
        if result['status'] != 'failed':
            import numpy as np
            from pymatgen.core import Structure
            force = np.asarray(result['forces'])
            assert force.shape == (len(source['initial_structure']['sites']), 3) and np.isfinite(force).all()
            assert np.isfinite(result['energy_total'])
            assert np.allclose(Structure.from_dict(result['relaxed_structure']).lattice.matrix, Structure.from_dict(source['initial_structure']).lattice.matrix, rtol=0, atol=1e-10)
    assert len(records) == 6697
    (reference, _, _) = reproduce.data()
    ref = reference['reference_data']
    fp = json.loads((mace.OUT / 'standardized_results.json').read_text())['models'][mace.KEY]
    for (system, candidates) in ref['hull'].items():
        fp['hull']['relax'][system] = {cid: records['hull||' + (system if rec['role'] == 'interior' else '__endpoint__') + '||' + cid] for (cid, rec) in candidates.items()}
    for (group, row) in ref['ordering'].items():
        fp['ordering']['relax'][group] = {cid: records['ordering||' + group + '||' + cid] for cid in row['orderings']}
    base.atomic_json(OUT / 'standardized_results.json', {'models': {mace.KEY: fp}})
    import convexhull_analysis_utils as u
    from finalize_report import clean
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        built = u.build_phase_stability_ordering_results(ref, {mace.KEY: fp})
        (hull, hull_details) = u.build_combined_hull_table(built['dft_hull'], built['fp_hull'], [mace.KEY], {mace.KEY: 'MACE-MH-1 (omat_pbe)'})
        (ordering, ordering_details) = u.build_combined_ordering_table(built['dft_ordering'], built['fp_ordering'], [mace.KEY], {mace.KEY: 'MACE-MH-1 (omat_pbe)'})
        (rmsd, rmsd_frames, rmsd_details) = u.build_rmsd_table(built['dft_hull'], built['fp_hull'], [mace.KEY], {mace.KEY: 'MACE-MH-1 (omat_pbe)'})
    for (name, table) in [('hull_metrics', hull), ('ordering_metrics', ordering), ('rmsd_metrics', rmsd)]:
        table.to_csv(OUT / (name + '.csv'))
    for (key, frame) in rmsd_frames.items():
        frame.to_csv(OUT / ('rmsd_records_' + key + '.csv'), index=False)
    summary = {'completed': len(records), 'planned': 6697, 'provenance': provenance, 'success': sum((r['status'] == 'success' for r in records.values())), 'non_converged': sum((r['status'] == 'non_converged' for r in records.values())), 'failed': sum((r['status'] == 'failed' for r in records.values())), 'hull_details': hull_details, 'ordering_details': ordering_details, 'rmsd_details': rmsd_details}
    (OUT / 'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False))
    lines = ['# MACE-MH-1 (omat_pbe): full phase stability and ordering evaluation', '', f"All 6,697 initial structures evaluated: {summary['success']} converged, {summary['non_converged']} non-converged, {summary['failed']} failed.", '', 'The exact upstream fixed-cell FIRE routine was used with fmax 0.01 eV/Å and 10,000 steps. Static and relaxed results remain separate. All metrics below use unchanged public analysis functions.', '']
    for (name, table) in [('Phase stability', hull), ('Ordering', ordering), ('Geometry', rmsd)]:
        lines += ['## ' + name, '', table.to_markdown(), '']
    lines += ['Training-data overlap has not been fully audited; benchmark performance does not establish out-of-distribution superiority.', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['completed', 'success', 'non_converged', 'failed']}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    analyze() if args.analyze else run(args)
