"""Compare fresh full-NEB results through FP-DeErr's unchanged exporter."""
import argparse
import json
import os
import sys
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
import reproduce
from relaxation_runner import atomic_json


def analyze(device):
    folder = reproduce.OUT / ('neb_full_cuda' if device == 'cuda' else 'neb_full_cpu')
    run = json.loads((folder / 'run_summary.json').read_text(encoding='utf-8'))
    assert run['completed'] == run['planned'] == 154
    sys.path.insert(0, str(reproduce.REPO / 'Ion_migration_NEB/scripts'))
    import neb_analysis as na
    from neb_plots import area_between_curves, simplify_class
    from export_neb_leaderboard import build_leaderboard
    reference = reproduce.REPO / 'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz'
    result_path = folder / 'results.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    branch = result['models']['CHGNET']['full_fp_neb']
    # Accept outputs from a calculation process started before this export-key
    # correction; change only the schema key, never scientific values.
    corrected=False
    for pathway in branch['pathways'].values():
        if 'images' in pathway and 'final_fp_neb_images' not in pathway:
            pathway['final_fp_neb_images']=pathway.pop('images')
            corrected=True
    if corrected:
        atomic_json(result_path,result)
    failed = branch['unsuccessful_pathways']
    assert len(branch['pathways']) + len(failed) == 154
    assert not failed, f'{len(failed)} failed pathways require investigation'
    raw_reference=reproduce.load(reference)
    image_count=0
    for key,pathway in branch['pathways'].items():
        images=pathway['final_fp_neb_images']
        assert set(images)=={str(i) for i in range(7)}, key
        source_cell=np.asarray(raw_reference['pathways'][key]['full_fp_neb_input']['source_start_endpoint_structure']['lattice']['matrix'])
        for image in images.values():
            force=np.asarray(image['fp_forces_eV_per_angstrom'])
            assert force.shape==(len(image['structure']['sites']),3),key
            assert np.isfinite(force).all() and np.isfinite(image['fp_energy_total_eV']),key
            assert np.allclose(image['structure']['lattice']['matrix'],source_cell,rtol=0,atol=1e-8),key
            image_count+=1
    assert image_count==1078
    # This calls the exact public validation, metric aggregation and rounding
    # used to generate the original compact benchmark leaderboard.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fresh = build_leaderboard(reference, result_path, na, area_between_curves, simplify_class, fp_keys=['CHGNET'])
    atomic_json(folder / 'leaderboard.json', fresh)
    published = json.loads((reproduce.REPO / 'Ion_migration_NEB/data/ion_migration_neb_leaderboard_summary.json').read_text(encoding='utf-8'))
    expected = next(row for row in published['models'] if row['fp_key'] == 'CHGNET')
    actual = fresh['models'][0]
    comparisons = [{'metric': key, 'published': value, 'fresh': actual[key], 'match': value == actual[key]}
                   for key, value in expected.items() if key not in ('fp_key', 'display_name')]
    summary = {'pathways': 154, 'images':image_count, 'all_image_shapes_and_values_valid':True, 'converged': actual['n_total'] - actual['n_nonconverged'],
        'non_converged': actual['n_nonconverged'], 'failed': len(failed), 'validation_pass': True,
        'matching_metrics': sum(row['match'] for row in comparisons), 'total_metrics': len(comparisons),
        'comparisons': comparisons, 'provenance': run['provenance'],'execution':run.get('execution',{}),
        'dft_static_on_fresh_fp_neb': 'not run; requires VASP, licensed pseudopotentials and DFT compute access'}
    atomic_json(folder / 'summary.json', summary)
    lines = ['# CHGNet full ion-migration NEB reproduction', '',
        f"Fresh {device} calculations completed for all 154 pathways: {summary['converged']} converged and {summary['non_converged']} did not converge. No pathway was dropped. Input validation passed.", '',
        'The exact upstream template was executed using fixed-cell BFGS endpoint relaxation (fmax 0.002 eV/Å, 500 steps), five intermediate climbing NEB images (fmax 0.05 eV/Å, 1,000 steps) and the original periodic endpoint prealignment and autosort tolerances. Only calculator initialization and file persistence were adapted for this Windows environment.', '',
        '| Metric | Published | Fresh | Match |', '|---|---:|---:|:---:|']
    lines += [f"| {row['metric']} | {row['published']} | {row['fresh']} | {'Yes' if row['match'] else 'No'} |" for row in comparisons]
    lines += ['', 'Comparisons use the unchanged public leaderboard exporter and pooled forward/backward barrier errors. Optimization trajectories and their convergence can differ with hardware and dependency versions; any differences are retained above.', '',
        'Execution settings and per-path inference provenance are retained in the summary and individual records. The CPU inference optimization disables unused higher derivative graph construction; exact raw-output and full-trajectory checks are documented in ../optimization_report.md. Model weights, float32 precision, graph construction and convergence settings are unchanged.', '',
        'DFT static evaluations on these newly generated model paths are not included. That separate protocol requires licensed VASP pseudopotentials and DFT execution access. Static model evaluations on all original DFT paths were already completed and are included in results.json.', '',
        'The full raw images, forces, energies, optimization logs and runtime checkpoints are retained alongside this report.']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    analyze(parser.parse_args().device)
