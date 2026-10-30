"""Build reviewable four-model leaderboard tables only after both fresh audits."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/round2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as source:
        return list(csv.reader(source))


def selected_row(path, prefix, skip=1):
    return next(row[skip:] for row in rows(path) if row and row[0].startswith(prefix))


def main():
    scope = read(ROOT / 'round2_scope.json')
    models = scope['models']
    audits = {}
    for key, specs in scope['required_evidence'].items():
        audit_path = OUT / key / 'scientific_audit.json'
        audit = read(audit_path)
        assert audit['pass'] and audit['deep_source_verification']
        assert audit['scope_sha256'] == sha(ROOT / 'round2_scope.json')
        assert audit['auditor_sha256'] == sha(ROOT / 'audit_round2_model.py')
        for stage, spec in specs.items():
            assert audit['checks'][stage]['complete']
            assert audit['checks'][stage]['summary_sha256'] == sha(ROOT / spec['summary'])
        audits[key] = audit
    destination = OUT / 'review_bundle'
    tables = destination / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'outputs/prof_mo_review/tables'
    keys = list(models)
    headings = ['Metric', 'Fresh CHGNet', 'MACE-OMAT-0 Small'] + [models[key]['display_name'] for key in keys]
    built = {}
    def write(name, data):
        with (tables / (name + '.csv')).open('w', encoding='utf-8', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(headings)
            writer.writerows(data)
        built[name] = data
    static_old = rows(base / 'static_comparison.csv')[1:]
    new_static = {key: read(ROOT / 'outputs' / key / 'summary.json')['metrics'] for key in keys}
    write('static_comparison', [row + [new_static[key][i][key] for key in keys] for i, row in enumerate(static_old)])
    phase_new = {}
    order_new = {}
    for key in keys:
        folder = ROOT / 'outputs' / key / 'relaxation'
        display = models[key]['display_name']
        phase_new[key] = selected_row(folder / 'hull_metrics.csv', display)[:4] + selected_row(folder / 'rmsd_metrics.csv', display)
        order_new[key] = selected_row(folder / 'ordering_metrics.csv', 'Full', skip=2)
    for name, values in [('relaxed_phase_comparison', phase_new), ('relaxed_ordering_comparison', order_new)]:
        old = rows(base / (name + '.csv'))[1:]
        write(name, [[row[0], row[2], row[3]] + [values[key][i] for key in keys] for i, row in enumerate(old)])
    for dataset in ['matpes_pbe', 'omat24']:
        old = rows(base / (dataset + '_force_comparison.csv'))[1:]
        new = {key: list(read(ROOT / scope['required_evidence'][key][dataset]['summary'])['summary_metrics'].values()) for key in keys}
        write(dataset + '_force_comparison', [[row[0], row[2], row[3]] + [new[key][i] for key in keys] for i, row in enumerate(old)])
    baseline_neb = read(ROOT / 'outputs/neb_full_cpu/summary.json')['comparisons']
    old = rows(base / 'neb_comparison.csv')[1:]
    new = {key: read(ROOT / 'outputs' / key / 'neb/summary.json')['metrics'] for key in keys}
    write('neb_comparison', [[row[0], row[2], row[3]] + [new[key][baseline_neb[i]['metric']] for key in keys] for i, row in enumerate(old)])
    report = ['# Additional-model leaderboard contribution', '',
              'All twelve additional scientific stages and both fresh deep-source model audits have passed.', '',
              'The tables compare the same reference populations and unchanged public metrics. Static and full-relaxation protocols remain separate. Non-converged paths remain in the saved results and counts.', '',
              'Training exposure is documented per checkpoint. These results do not establish unseen-data performance.', '',
              'The selected material heads target PBE/PBE+U labels, while MatPES-PBE supplies PBE labels. Reference-functional differences can affect the comparisons; r2SCAN heads/results are not mixed into these tables.', '',
              '**Status: verified results prepared for review; not yet merged into the upstream leaderboard.**', '']
    for name, data in built.items():
        report += ['## ' + name.replace('_', ' '), '', '| ' + ' | '.join(headings) + ' |',
                   '| ' + ' | '.join(['---'] * len(headings)) + ' |']
        report += ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in data]
        report += ['']
    report += ['## Exact checkpoints', '']
    for key, model in models.items():
        report += [f"- {model['display_name']}: [{model['checkpoint']}]({model['url']}), head `{model['head']}`, SHA-256 `{model['sha256']}`.",
                   '  ' + model['training_overlap']]
        shutil.copy2(OUT / key / 'scientific_audit.json', destination / (key + '_audit.json'))
        relaxation = audits[key]['checks']['full_relaxation']
        if relaxation.get('cpu_cuda_profile_validation'):
            execution = read(ROOT / 'outputs' / key / 'relaxation/summary.json')
            report += [f"  Full relaxation execution counts: {relaxation['execution_counts']}. CPU and CUDA used float64, the same ASE FIRE version, fixed cells, 0.01 eV/Angstrom force tolerance and 10,000-step limit. Complete validation trajectories matched optimization step counts; actual execution profiles are included."]
            shutil.copy2(ROOT / 'outputs' / key / 'relaxation/summary.json', destination / (key + '_relaxation_summary.json'))
            acceleration = destination / 'relaxation_acceleration'
            acceleration.mkdir(exist_ok=True)
            for name in ['validation.json', 'calibration_cpu.json', 'calibration_cuda.json']:
                shutil.copy2(OUT / 'relaxation_acceleration' / name, acceleration / name)
        for dataset in ['matpes_pbe', 'omat24']:
            storage = audits[key]['checks'][dataset].get('saved_tensor_disk_offload', {})
            if storage.get('count', 0):
                report += [f"  {dataset}: {storage['count']} singleton CUDA memory failures used validated saved-tensor disk storage; all arithmetic remained float64 CUDA. Structure indices and storage statistics are retained in the audit and raw shard metadata."]
    report += ['', 'Original CHGNet and MACE-Small benchmark outputs are retained unchanged. See the original report for reproduction discrepancies. The upstream contribution guide requests exact model/checkpoint identity and official source links. Existing rows must remain unchanged when these entries are integrated.', '']
    (destination / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    index = {'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'audits_pass': True,
             'scope_sha256': sha(ROOT / 'round2_scope.json'),
             'upstream_commit': scope['upstream_commit'], 'merged_upstream': False,
             'models': models,
             'files_sha256': {path.relative_to(destination).as_posix(): sha(path) for path in destination.rglob('*')
                              if path.is_file() and path.name != 'manifest.json'}}
    (destination / 'manifest.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    # Stage the actual additive upstream patch only after both fresh audits.
    # This does not modify a checkout, create a PR, or publish any results.
    from prepare_upstream_leaderboard import main as prepare_upstream
    upstream = prepare_upstream()
    print(json.dumps({'review_bundle': str(destination), 'audits_pass': True, 'upstream_merged': False}), flush=True)
    print(json.dumps({'upstream_candidate': str(upstream), 'published': False}), flush=True)


if __name__ == '__main__':
    main()
