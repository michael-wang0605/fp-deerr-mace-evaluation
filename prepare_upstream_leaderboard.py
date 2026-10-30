"""Prepare an audit-gated, additive FP-DeErr leaderboard patch in a separate folder.

No production output, upstream checkout, Git remote, or email is modified.
The Small-only mode rehearses integration using the completed first-round audit.
The default includes Small and both additional models and refuses partial runs.
"""
import argparse
import ast
import copy
import csv
import datetime
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'FP-DeErr'
OUT = ROOT / 'outputs/round2'
FIELDS = ['dft_force_magnitude', 'fp_force_magnitude', 'force_magnitude_error',
          'force_angle_error', 'force_vector_error']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def specs(small_only):
    scope = read(ROOT / 'round2_scope.json')
    small = {'display_name': 'MACE-OMAT-0 Small', 'fp_key': 'MACE_OMAT_0_SMALL',
             'checkpoint': 'mace-omat-0-small.model', 'head': 'omat_pbe',
             'sha256': '0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5',
             'url': 'https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model',
             'training_overlap': 'OMat24 training data; exact structure-level overlap is not audited.',
             'dtype': 'float64', 'cutoff_angstrom': 6.0, 'dispersion': False}
    result = {'mace_omat_0_small': small}
    if not small_only:
        result.update(scope['models'])
    for key, model in result.items():
        model['training'] = 'OMat24' if key != 'mace_mh_1_omat_pbe' else 'OMat pretraining + cross-domain fine-tuning'
        model['force_folders'] = {dataset: f'outputs/force_benchmarks/{dataset}/{key}_' +
                                  ('cpu' if key == 'mace_omat_0_small' and dataset == 'omat24' else 'cuda')
                                  for dataset in ['matpes_pbe', 'omat24']}
    return scope, result


def gate(scope, models):
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == scope['upstream_commit']
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', 'HEAD', '--'], text=True)
    original = read(ROOT / 'research_scope.json')
    audit = read(ROOT / 'outputs/prof_mo_review/scientific_audit.json')
    assert audit['pass'] and audit['deep_source_verification']
    assert audit['scope_sha256'] == sha(ROOT / 'research_scope.json')
    assert audit['auditor_sha256'] == sha(ROOT / 'audit_research_results.py')
    for name, spec in original['required_evidence'].items():
        assert audit['checks'][name]['complete']
        assert audit['checks'][name]['summary_sha256'] == sha(ROOT / spec['summary'])
    evidence = {'mace_omat_0_small': 'outputs/prof_mo_review/scientific_audit.json'}
    for key in models:
        assert sha(ROOT / 'models' / models[key]['checkpoint']) == models[key]['sha256']
        if key == 'mace_omat_0_small':
            continue
        path = OUT / key / 'scientific_audit.json'
        current = read(path)
        assert current['pass'] and current['deep_source_verification']
        assert current['scope_sha256'] == sha(ROOT / 'round2_scope.json')
        assert current['auditor_sha256'] == sha(ROOT / 'audit_round2_model.py')
        for stage, spec in scope['required_evidence'][key].items():
            assert current['checks'][stage]['complete']
            assert current['checks'][stage]['summary_sha256'] == sha(ROOT / spec['summary'])
        evidence[key] = path.relative_to(ROOT).as_posix()
    return evidence


def force_rows(key, model, dataset):
    """Re-use public force functions, on the same finite population as the notebook."""
    import numpy as np
    import force_benchmark_analysis as analysis
    sys.path.insert(0, str(UPSTREAM / 'Force_error/scripts'))
    import force_error_metrics as metrics
    folder = ROOT / model['force_folders'][dataset]
    summary = read(folder / 'summary.json')
    assert summary['full_dataset_complete']
    assert summary['structures'] == {'matpes_pbe': 434712, 'omat24': 117004}[dataset]
    collected = {field: [] for field in FIELDS}
    index, atoms = 0, 0
    for path in sorted((folder / 'shards').glob('*.npz')):
        meta = read(path.with_suffix('.json'))
        assert meta['sha256'] == sha(path) and meta['first'] == index
        with np.load(path, allow_pickle=False) as shard:
            assert np.array_equal(shard['structure_indices'], np.arange(meta['first'], meta['last'] + 1))
            for field in FIELDS:
                assert len(shard[field]) == meta['atoms']
                collected[field].append(shard[field].copy())
        index = meta['last'] + 1
        atoms += meta['atoms']
    assert index == summary['structures'] and atoms == summary['atoms']
    arrays = {field: np.concatenate(parts) for field, parts in collected.items()}
    del collected
    table, _, _, _, finite, standard = analysis.summary_from_arrays(dataset, arrays, model['display_name'])
    assert table.loc[model['display_name']].to_dict() == summary['summary_metrics']
    assert int(finite.sum()) == summary['finite_angle_atoms']
    compatible = {a: standard[b] for a, b in zip(['F_dft', 'F_fp', 'deltaF', 'deltaTheta', 'e_vec'], FIELDS)}
    data = {model['display_name']: compatible}
    cuts = [.01, .02, .05, .07, .1, .2, .5]
    small = metrics.build_highly_accurate_force_fraction_table(data, cuts, fdf_min=.01).iloc[0]
    joint = metrics.build_joint_dF_theta_accuracy_table(data, cuts, [1, 20], fdf_min=.01)
    large = metrics.build_large_force_error_fraction_table(data, [.5, 1, 2, 3, 4, 5, 7, 10], fdf_min=.01).iloc[0]
    mae, rmse = metrics.build_dF_mae_rmse_fdft_subset(data, [0, .01, .05, .1, .2, .5, .7, 1, 2])
    vector = metrics.build_evec_high_accuracy_fraction_table(data, [.01, .05, .1, .2, .3, .4, .5, 1, 2], fdf_min=.01).iloc[0]
    values = list(summary['summary_metrics'].values())
    # Website column order differs from the notebook summary's column order.
    summary_values = [values[i] for i in [0, 5, 6, 7, 4, 3, 1, 2]]
    detailed = [[f'{v:.2f}' for v in small],
                [f'{a:.2f} / {b:.2f}' for a, b in zip(joint[1].iloc[0], joint[20].iloc[0])],
                [f'{v:.3f}' for v in large],
                [f'{a:.3f} / {b:.3f}' for a, b in zip(mae.iloc[0], rmse.iloc[0])],
                [f'{v:.2f}' for v in vector]]
    return summary_values, detailed, {'structures': index, 'atoms': atoms,
                                    'finite_angle_atoms': int(finite.sum()),
                                    'summary_sha256': sha(folder / 'summary.json')}


def phase_values(key, model):
    folder = ROOT / 'outputs' / key / 'relaxation'
    def pick(filename, first, skip):
        with (folder / filename).open(encoding='utf-8', newline='') as source:
            return next(row[skip:] for row in csv.reader(source) if row and row[0] == first)
    hull = pick('hull_metrics.csv', model['display_name'], 1)
    full = pick('ordering_metrics.csv', 'Full FP relaxation', 2)
    static = pick('ordering_metrics.csv', 'Static FP evaluations on the DFT-relaxed structures', 2)
    rmsd = pick('rmsd_metrics.csv', model['display_name'], 1)
    assert len(hull) == 8 and len(full) == len(static) == 7 and len(rmsd) == 5
    return [hull[:4], hull[4:], full, static, rmsd]


def cell(value, pair=False, integer_pair=False):
    escaped = html.escape(str(value))
    parts = str(value).split(' / ')
    for part in parts:
        assert __import__('math').isfinite(float(part)), 'Missing leaderboard value: ' + str(value)
    if len(parts) == 2:
        if pair:
            fmt = ' data-fmt="int"' if integer_pair else ''
            return f'<td data-val="{parts[0]}" data-val2="{parts[1]}" data-pair="1"{fmt}>{escaped}</td>'
        return f'<td data-val="{parts[0]}">{escaped}</td>'
    assert len(parts) == 1
    return f'<td data-val="{escaped}">{escaped}</td>'


def row(model, values, provenance=False, paired_cells=(), integer_pairs=False):
    result = '<tr><td class="fp-name">' + html.escape(model['display_name']) + '</td>'
    result += ''.join(cell(value, pair=i in paired_cells, integer_pair=integer_pairs) for i, value in enumerate(values))
    if provenance:
        title = f"head={model['head']}; SHA-256={model['sha256']}; {model['training_overlap']}"
        result += '<td class="training-col" title="' + html.escape(title, quote=True) + '">'
        result += '<a class="ver" href="' + html.escape(model['url'], quote=True) + '">' + html.escape(model['checkpoint']) + '</a>'
        result += ' <span class="trn">(' + html.escape(model['training']) + ')</span></td>'
    return result + '</tr>'


def add_rows(document, additions):
    """Insert complete rows without reserializing any existing HTML."""
    from bs4 import BeautifulSoup
    index = -1
    inserted = []
    def replace(match):
        nonlocal index
        block = match.group(0)
        parsed = BeautifulSoup(block, 'html.parser').find('table')
        if 'dt' not in parsed.get('class', []):
            return block
        index += 1
        if index not in additions:
            return block
        assert block.count('</tbody>') == 1
        expected = len(parsed.find('thead').select('tr')[-1].select('th'))
        for entry in additions[index]:
            parsed_row = BeautifulSoup(entry, 'html.parser').find('tr')
            assert len(parsed_row.find_all('td', recursive=False)) == expected
            label = parsed_row.find('td').get_text()
            assert all(old.find('td').get_text() != label for old in parsed.find('tbody').find_all('tr') if old.find('td'))
        text = ''.join(entry + '\n' for entry in additions[index])
        inserted.append(text)
        closing_line = re.search(r'(?m)^[ \t]*</tbody>', block)
        if closing_line:
            return block[:closing_line.start()] + text + block[closing_line.start():]
        return block.replace('</tbody>', text + '</tbody>')
    updated = re.sub(r'<table\b[\s\S]*?</table>', replace, document)
    assert set(additions) <= set(range(index + 1))
    restored = updated
    for addition in inserted:
        restored = restored.replace(addition, '', 1)
    assert restored == document, 'An existing HTML byte was changed'
    return updated


def extend_export_registry(document, models):
    tree = ast.parse(document)
    replacements = []
    lines = document.splitlines(keepends=True)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
            continue
        name = statement.targets[0].id
        if name not in ['FP_ORDER', 'FP_DISPLAY_NAMES']:
            continue
        value = ast.literal_eval(statement.value)
        if name == 'FP_ORDER':
            value += [model['fp_key'] for model in models.values()]
        else:
            value.update({model['fp_key']: model['display_name'] for model in models.values()})
        replacements.append((statement.lineno - 1, statement.end_lineno, name + ' = ' + repr(value) + '\n'))
    assert len(replacements) == 2
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    updated = ''.join(lines)
    # Register identity only; every executable metric/export function stays identical.
    before = [ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    after = [ast.dump(n) for n in ast.parse(updated).body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    assert before == after
    return updated


def compressed_result(path):
    """Retain the exact exporter input in a deterministic, Git-sized gzip."""
    stream = io.BytesIO()
    with path.open('rb') as source, gzip.GzipFile(filename='', mode='wb', fileobj=stream,
                                                  mtime=0, compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1 << 20)
    value = stream.getvalue()
    digest = hashlib.sha256()
    with gzip.GzipFile(fileobj=io.BytesIO(value)) as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    assert digest.hexdigest() == sha(path)
    assert len(value) < 50 * 2**20, 'Use a separate data artifact for a large gzip.'
    return value


def main(small_only=False):
    scope, models = specs(small_only)
    evidence = gate(scope, models)
    destination = OUT / ('upstream_small_rehearsal' if small_only else 'upstream_contribution')
    files = {}
    force_additions, phase_additions, force_proofs = {}, {}, {}
    for key, model in models.items():
        print('Preparing ' + model['display_name'], flush=True)
        force_proofs[key] = {}
        for dataset, summary_index, detail_start in [('matpes_pbe', 0, 3), ('omat24', 2, 13)]:
            summary, detailed, proof = force_rows(key, model, dataset)
            force_proofs[key][dataset] = proof
            force_additions.setdefault(summary_index, []).append(row(model, summary, provenance=True))
            for offset, values in enumerate(detailed):
                force_additions.setdefault(detail_start + offset, []).append(
                    row(model, values, paired_cells=range(len(values)) if offset in [1, 3] else ()))
        for index, values in enumerate(phase_values(key, model)):
            phase_additions.setdefault(index, []).append(row(model, values, provenance=index in [0, 2, 4],
                paired_cells=(6,) if index in [2, 3] else (1,) if index == 4 else (), integer_pairs=index in [2, 3]))
    def source(path):
        return (UPSTREAM / path).read_bytes().decode('utf-8')
    files['docs/force-error.html'] = add_rows(source('docs/force-error.html'), force_additions)
    files['docs/phase-stability-ordering.html'] = add_rows(source('docs/phase-stability-ordering.html'), phase_additions)
    base_neb = read(UPSTREAM / 'docs/data/ion_migration_neb_leaderboard_summary.json')
    neb = copy.deepcopy(base_neb)
    for key, model in models.items():
        added = read(ROOT / 'outputs' / key / 'neb/leaderboard.json')
        for name, value in base_neb.items():
            if name not in ['fp_order', 'models', 'generation_metadata']:
                assert added[name] == value, 'Incompatible NEB global field: ' + name
        assert added['fp_order'] == [model['fp_key']] and len(added['models']) == 1
        assert model['fp_key'] not in neb['fp_order']
        neb['fp_order'].append(model['fp_key'])
        neb['models'].extend(added['models'])
        meta = added['generation_metadata']
        assert meta['source_reference_sha256'] == neb['generation_metadata']['source_reference_sha256']
        raw = ROOT / 'outputs' / key / 'neb/results.json'
        assert sha(raw) == meta['source_results_sha256']
        relative = f'contributions/mace-models/{key}/neb_results.json.gz'
        files[relative] = compressed_result(raw)
        neb['generation_metadata']['sources'].append({
            'source_results_file': relative,
            'source_results_sha256': hashlib.sha256(files[relative]).hexdigest(),
            'uncompressed_export_input_sha256': meta['source_results_sha256'],
            'fp_keys': added['fp_order']})
    assert neb['models'][:len(base_neb['models'])] == base_neb['models']
    assert neb['fp_order'][:len(base_neb['fp_order'])] == base_neb['fp_order']
    neb['generation_metadata']['merged_by'] += '; additional audited per-model exports appended without recomputation; existing model records unchanged'
    neb_json = json.dumps(neb, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    for path in ['docs/data/ion_migration_neb_leaderboard_summary.json', 'Ion_migration_NEB/data/ion_migration_neb_leaderboard_summary.json']:
        assert read(UPSTREAM / path) == base_neb
        files[path] = neb_json
    page = source('docs/ion-migration-neb.html')
    assert page.count('const EXPECTED_FP_COUNT = 15;') == 1
    page = page.replace('const EXPECTED_FP_COUNT = 15;', f"const EXPECTED_FP_COUNT = {len(neb['fp_order'])};")
    entries = ''.join('    ' + json.dumps(model['display_name']) + ': ' +
                      json.dumps([model['checkpoint'] + ' (head ' + model['head'] + ')', model['training']]) + ',\n'
                      for model in models.values())
    marker = '  const FP_PROVENANCE = {\n'
    assert page.count(marker) == 1 or page.count(marker.replace('\n', '\r\n')) == 1
    actual_marker = marker if marker in page else marker.replace('\n', '\r\n')
    page = page.replace(actual_marker, actual_marker + entries)
    page = page.replace('the 15 FPs evaluated here', 'the original 15 FPs evaluated here')
    marker = '<!-- ============== USE FP-DeErr ============== -->'
    assert page.count(marker) == 1
    note = '<p>Additional MACE checkpoint identities, heads, training exposure, and audited execution provenance: <a href="../contributions/mace-models/README.md">contribution record</a>.</p>\n\n'
    files['docs/ion-migration-neb.html'] = page.replace(marker, note + marker)
    exporter = 'Ion_migration_NEB/scripts/export_neb_leaderboard.py'
    files[exporter] = extend_export_registry(source(exporter), models)
    report = ['# Additional MACE models', '',
              'Audited leaderboard additions using the supplied references and unchanged public metric functions. Existing leaderboard rows are preserved.', '',
              'These checkpoints target PBE/PBE+U material labels. MatPES-PBE supplies PBE labels; functional differences can affect errors. No r2SCAN rows are added. Exact training-set overlap is not audited; these are not claims of unseen-data performance.', '',
              '## Exact checkpoints', '']
    for key, model in models.items():
        report += [f"- [{model['display_name']}]({model['url']}): `{model['checkpoint']}`, head `{model['head']}`, SHA-256 `{model['sha256']}`. {model['training_overlap']}"]
    report += ['', '## Reproduction', '',
               'Use the component registries described in `ADDING_A_POTENTIAL.md`, with these identities and the following ASE calculator setup (replace MODEL_PATH and MODEL_HEAD with the pinned values above):', '',
               '```python', 'from mace.calculators import MACECalculator',
               'calc = MACECalculator(model_paths=MODEL_PATH, head=MODEL_HEAD,',
               '                      device="cpu", default_dtype="float64")', '```', '',
               'Evaluation used mace-torch 0.3.16, float64, a 6 Å cutoff and no dispersion correction. Fixed cells, convergence thresholds and public metric definitions were retained. CPU/GPU execution and exact dependency versions remain recorded by stage; the NEB ASE version differs from the upstream reference implementation and is explicitly retained in provenance. Non-converged paths remain counted.', '',
               'Source, checksums, audit records, standardized results, and command adapters: [review repository](https://github.com/michael-wang0605/fp-deerr-mace-evaluation). The accompanying manifest identifies the source audit and every contributed file.', '',
               'Force detail rows are computed with the unmodified upstream force functions from checksum-verified retained shards, after the same finite-angle filtering as the notebooks. Main summary strings and phase rows are copied from the audited notebook outputs. NEB rows are copied verbatim from the public exporter outputs.', '']
    files['contributions/mace-models/README.md'] = '\n'.join(report)
    files['contributions/mace-models/checkpoints.json'] = json.dumps(models, indent=2) + '\n'
    for key, relative in evidence.items():
        files[f'contributions/mace-models/{key}/scientific_audit.json'] = (ROOT / relative).read_bytes()
        current_audit = read(ROOT / relative)
        relaxation = current_audit['checks'].get('full_relaxation', {})
        if relaxation.get('cpu_cuda_profile_validation'):
            files[f'contributions/mace-models/{key}/relaxation_summary.json'] = (ROOT / 'outputs' / key / 'relaxation/summary.json').read_bytes()
            acceleration = OUT / 'relaxation_acceleration'
            for name in ['validation.json', 'calibration_cpu.json', 'calibration_cuda.json']:
                files[f'contributions/mace-models/relaxation_acceleration/{name}'] = (acceleration / name).read_bytes()
            proof = read(acceleration / 'validation.json')
            for name, digest in proof['source_files_sha256'].items():
                assert sha(ROOT / name) == digest
                files[f'contributions/mace-models/relaxation_acceleration/{name}'] = (ROOT / name).read_bytes()
            for path in (acceleration / 'calibration').rglob('*.json'):
                relative = path.relative_to(acceleration).as_posix()
                files[f'contributions/mace-models/relaxation_acceleration/{relative}'] = path.read_bytes()
            files['contributions/mace-models/README.md'] += f"\n{models[key]['display_name']} full-relaxation execution counts: {relaxation['execution_counts']}. CPU and CUDA used float64 and identical ASE FIRE settings (fixed cells, fmax 0.01 eV/Angstrom, 10,000-step limit). Complete calibration trajectories matched step counts, and numerical differences passed the recorded tolerances. Per-device versions, calibration proofs and execution counts accompany the results.\n"
        if any(check.get('saved_tensor_disk_offload', {}).get('count', 0) for check in current_audit['checks'].values()):
            for name in ['mace_cuda_disk_offload_validation.json', 'mace_cuda_disk_offload_retry_validation.json']:
                files[f'contributions/mace-models/{key}/{name}'] = (OUT / key / 'validation' / name).read_bytes()
            files[f'contributions/mace-models/{key}/round2_mh1_force_offload.py'] = (ROOT / 'round2_mh1_force_offload.py').read_bytes()
            files[f'contributions/mace-models/{key}/cuda_saved_tensor_offload.py'] = (ROOT / 'cuda_saved_tensor_offload.py').read_bytes()
            for validator in ['validate_round2_mh1_offload.py', 'verify_round2_mh1_offload_controls.py']:
                files[f'contributions/mace-models/{key}/{validator}'] = (ROOT / validator).read_bytes()
            files['contributions/mace-models/README.md'] += '\nFor MH1 singleton CUDA memory failures, validated saved-tensor disk storage was used. Every tensor retained its dtype and returned to its original CUDA device for arithmetic; no CPU model evaluation or precision reduction was substituted. The model audit lists each affected structure index and storage statistics, and the numerical/control proofs are included.\n'
        for filename in ['hull_metrics.csv', 'ordering_metrics.csv', 'rmsd_metrics.csv']:
            files[f'contributions/mace-models/{key}/{filename}'] = (ROOT / 'outputs' / key / 'relaxation' / filename).read_bytes()
        for dataset, folder in models[key]['force_folders'].items():
            for filename in ['summary_metrics.csv', 'summary.json']:
                files[f'contributions/mace-models/{key}/{dataset}_{filename}'] = (ROOT / folder / filename).read_bytes()
    files['contributions/mace-models/validate.py'] = (ROOT / 'validate_mace_contribution.py').read_bytes()
    # Preserve exact evidence bytes across platforms, and keep modified upstream
    # text in its canonical Git LF form rather than the Windows checkout's CRLF.
    assert not (UPSTREAM / '.gitattributes').exists(), 'Merge existing attributes explicitly.'
    files['.gitattributes'] = '\n'.join(
        [relative + ' -text' for relative in files if (UPSTREAM / relative).exists()] +
        ['.gitattributes -text', 'contributions/mace-models/** -text whitespace=cr-at-eol', ''])
    # Prepare everything in memory before writing any candidate file.
    for relative, contents in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents if isinstance(contents, bytes) else contents.replace('\r\n', '\n').encode('utf-8'))
    manifest = {'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'upstream_commit': scope['upstream_commit'], 'mode': 'small_only_rehearsal' if small_only else 'full_contribution',
                'models': list(models), 'audit_gate_passed': True, 'published': False,
                'force_proofs': force_proofs, 'existing_html_content_preserved': True,
                'existing_neb_model_records_preserved': True, 'neb_record_count': len(neb['models']),
                'force_added_rows': sum(map(len, force_additions.values())),
                'phase_added_rows': sum(map(len, phase_additions.values())),
                'audit_sha256': {key: sha(ROOT / relative) for key, relative in evidence.items()},
                'files_sha256': {relative: sha(destination / relative) for relative in files},
                'source_files_sha256': {relative: sha(UPSTREAM / relative) for relative in files if (UPSTREAM / relative).is_file()},
                'builder_sha256': sha(Path(__file__))}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (destination / 'contributions/mace-models/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'destination': str(destination), 'models': list(models), 'existing_rows_preserved': True}), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-only', action='store_true', help='Rehearse only with the completed first-round model.')
    main(parser.parse_args().small_only)
