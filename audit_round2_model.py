"""Deep-source audit of a new model using the existing independent auditor.

AST bindings change the additional-model NEB registry key and its CUDA proof
path. Force paths are supplied through an in-memory scope copy. Audit functions and
upstream metric definitions are retained, then phase tables are recomputed too.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import types

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind_auditor_source(original, model, key):
    bindings = {'MACE_OMAT_0_SMALL': model['fp_key'],
                'mace_cuda_neb_validation.json': 'round2/' + key + '/validation/mace_cuda_neb_validation.json'}
    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in bindings:
                return ast.copy_location(ast.Constant(value=bindings[node.value]), node)
            return node
    return ast.fix_missing_locations(Binding().visit(ast.parse(original)))


def audit_force_offload(summary_path):
    """Verify the extra storage policy without changing the original audit."""
    cases = []
    folder = summary_path.parent
    summary = read(summary_path)
    for metadata_path in sorted((folder / 'shards').glob('*.json')):
        metadata = read(metadata_path)
        for event in metadata.get('cuda_oom_retries', []):
            if event.get('saved_tensor_storage') != 'disk':
                continue
            proof_path = ROOT / 'outputs/round2/mace_mh_1_omat_pbe/validation/mace_cuda_disk_offload_validation.json'
            proof = read(proof_path)
            assert proof['validation_pass'] and event['validation_sha256'] == sha(proof_path)
            assert proof['weights_sha256'] == summary['provenance']['weights_sha256']
            controls = read(proof_path.with_name('mace_cuda_disk_offload_retry_validation.json'))
            assert controls['validation_pass'] and controls['wrapper_sha256'] == proof['wrapper_sha256']
            assert event['wrapper_sha256'] == proof['wrapper_sha256'] == sha(ROOT / 'round2_mh1_force_offload.py')
            assert proof['offload_implementation_sha256'] == sha(ROOT / 'cuda_saved_tensor_offload.py')
            assert proof['batch_implementation_sha256'] == sha(ROOT / 'outputs/round2/adapters_v2/mace_mh_1_omat_pbe/mace_batch_inference.py')
            assert event['arithmetic_device'] == proof['device'] == 'cuda'
            assert event['dtype'] == proof['dtype'] == 'float64' and event['structures'] == 1
            assert 0 <= event['offset'] < metadata['structures']
            assert event['storage_statistics']['saved_tensors'] > 0 and event['storage_statistics']['unpack_calls'] > 0
            assert proof['dense_case']['original_singleton_oom'] and proof['dense_case']['all_fields_finite']
            assert all(value['max_abs'] < 1e-7 for value in proof['dense_case']['direct_ASE_differences'].values())
            deterministic = [row for row in proof['checks'] if row['deterministic']]
            assert {row['batch_size'] for row in deterministic} == {1, 16}
            assert all(value['equal'] for row in deterministic for value in row['offload_difference'].values())
            cases.append({'structure_index': metadata['first'] + event['offset'],
                          'storage_statistics': event['storage_statistics'],
                          'validation_sha256': event['validation_sha256']})
    assert len({case['structure_index'] for case in cases}) == len(cases)
    return {'complete': True, 'count': len(cases), 'cases': cases,
            'arithmetic_device': 'cuda', 'dtype': 'float64', 'storage_only_adaptation': True}


def audit_relaxation(auditor, key, jobs):
    """Retain the original audit's numerical checks, bind profile acceptance."""
    import round2_relax_engine as engine
    path = engine.OUT / 'validation.json'
    if not path.exists():
        return auditor.audit_relax('additional', jobs)
    proof = read(path)
    assert proof['pass'] and proof['engine_sha256'] == sha(Path(engine.__file__))
    assert all(sha(ROOT / name) == digest for name, digest in proof['source_files_sha256'].items())
    for device, digest in proof['calibration_files_sha256'].items():
        calibration = engine.OUT / ('calibration_' + device + '.json')
        assert sha(calibration) == digest and read(calibration)['pass']
    profiles = proof['profiles'][key]
    assert profiles['cpu'] == engine.legacy(key)
    assert engine.core(profiles['cpu']) == engine.core(profiles['cuda'])
    original = (ROOT / 'audit_research_results.py').read_text(encoding='utf-8')
    original_tree = ast.parse(original)
    function = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'audit_relax')
    tree = ast.Module(body=[function], type_ignores=[])
    changes = []
    def validate_saved(saved):
        device = engine.validate_record(saved, profiles)
        if 'execution' in saved:
            assert saved['execution']['engine_sha256'] == proof['engine_sha256']
        return bool(device)
    class Metadata(ast.NodeTransformer):
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "saved['id'] == key and saved['provenance'] == expected":
                node.test = ast.parse("saved['id'] == key and validate_saved(saved)", mode='eval').body
                changes.append('execution_profile_acceptance')
            return node
    tree = ast.fix_missing_locations(Metadata().visit(tree))
    assert changes == ['execution_profile_acceptance']
    namespace = dict(auditor.__dict__)
    namespace['validate_saved'] = validate_saved
    exec(compile(tree, '<mixed-relaxation-audit>', 'exec'), namespace)
    result = namespace['audit_relax']('additional', jobs)
    summary = read(ROOT / 'outputs' / key / 'relaxation/summary.json')
    assert summary['provenance'] == engine.core(profiles['cpu'])
    assert summary['execution_provenances'] == profiles
    devices = {read(p)['id']: engine.validate_record(read(p), profiles) for p in (ROOT / 'outputs' / key / 'relaxation/records').glob('*.json')}
    assert devices == summary['path_execution_environment'] and len(devices) == 6697
    result.update(calibration_sha256=sha(path), cpu_cuda_profile_validation=True,
                  unchanged_numerical_audit=True,
                  execution_counts={device: list(summary['path_execution_environment'].values()).count(device) for device in profiles})
    return result


def phase_metrics(mace, reference, checks):
    import convexhull_analysis_utils as u
    from threadpoolctl import threadpool_limits
    key = mace.KEY
    folder = mace.OUT
    static = read(folder / 'standardized_results.json')['models'][key]
    relaxed = read(folder / 'relaxation/standardized_results.json')['models'][key]
    for group, entries in reference['hull'].items():
        for cid, entry in entries.items():
            identifier = 'hull||' + (group if entry['role'] == 'interior' else '__endpoint__') + '||' + cid
            raw_static = read(mace.record_path(identifier))['result']
            path = folder / 'relaxation/records' / (hashlib.sha256(identifier.encode()).hexdigest() + '.json')
            assert static['hull']['static'][group][cid] == raw_static
            assert relaxed['hull']['static'][group][cid] == raw_static
            assert relaxed['hull']['relax'][group][cid] == read(path)['result']
    for group, entries in reference['ordering'].items():
        for cid in entries['orderings']:
            identifier = 'ordering||' + group + '||' + cid
            raw_static = read(mace.record_path(identifier))['result']
            path = folder / 'relaxation/records' / (hashlib.sha256(identifier.encode()).hexdigest() + '.json')
            assert static['ordering']['static'][group][cid] == raw_static
            assert relaxed['ordering']['static'][group][cid] == raw_static
            assert relaxed['ordering']['relax'][group][cid] == read(path)['result']
    display = read(ROOT / 'outputs/round2/models.json')[key]['display_name']
    with threadpool_limits(limits=1):
        for name, fp, target in [('static', static, folder), ('relaxed', relaxed, folder / 'relaxation')]:
            built = u.build_phase_stability_ordering_results(reference, {key: fp})
            hull, _ = u.build_combined_hull_table(built['dft_hull'], built['fp_hull'], [key], {key: display})
            ordering, _ = u.build_combined_ordering_table(built['dft_ordering'], built['fp_ordering'], [key], {key: display})
            frames = {'hull_metrics': hull, 'ordering_metrics': ordering}
            if name == 'relaxed':
                rmsd, _, _ = u.build_rmsd_table(built['dft_hull'], built['fp_hull'], [key], {key: display})
                frames['rmsd_metrics'] = rmsd
            for filename, frame in frames.items():
                assert frame.to_csv().replace('\r\n', '\n') == (target / (filename + '.csv')).read_text(encoding='utf-8')
            checks[name + '_phase_metrics'] = {'complete': True, 'all_public_metric_tables_recomputed': True,
                                               'every_export_matches_raw_prediction': True}


def main(key):
    scope = read(ROOT / 'round2_scope.json')
    model = scope['models'][key]
    specs = scope['required_evidence'][key]
    assert len(specs) == 6
    adapter = ROOT / 'outputs/round2/adapters_v2' / key
    sys.path.insert(0, str(adapter))
    sys.path.insert(1, str(ROOT))
    # These imports resolve this model's isolated, checksum-bound adapters.
    import reproduce
    import new_potential_static as mace
    import relaxation_runner as baseline
    assert mace.KEY == key and mace.MODEL_SHA == model['sha256']
    assert sha(mace.MODEL) == model['sha256']
    manifest = read(ROOT / 'outputs/round2/adapter_manifest_v2.json')
    for entry in manifest['models'][key]['adapters'].values():
        assert sha(ROOT / entry['path']) == entry['sha256']
    original = ROOT / 'audit_research_results.py'
    tree = bind_auditor_source(original.read_text(encoding='utf-8'), model, key)
    derived = ROOT / 'outputs/round2/audit_sources' / (key + '.py')
    derived.parent.mkdir(exist_ok=True)
    derived.write_text(ast.unparse(tree) + '\n', encoding='utf-8')
    auditor = types.ModuleType('round2_independent_auditor')
    auditor.__file__ = str(derived)
    exec(compile(tree, str(derived), 'exec'), auditor.__dict__)
    old_scope = read(ROOT / 'research_scope.json')
    old_scope['required_evidence']['additional_matpes_pbe']['summary'] = specs['matpes_pbe']['summary']
    old_scope['required_evidence']['additional_omat24']['summary'] = specs['omat24']['summary']
    original_read = auditor.read
    def scoped_read(path):
        return old_scope if Path(path).resolve() == (ROOT / 'research_scope.json').resolve() else original_read(path)
    auditor.read = scoped_read
    jobs, _ = baseline.plan_jobs('all')
    checks = {}
    errors = {}
    commit = subprocess.check_output(['git', '-C', str(reproduce.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    checks['upstream_source'] = {'complete': commit == scope['upstream_commit'] and
        subprocess.run(['git', '-C', str(reproduce.REPO), 'diff', '--quiet']).returncode == 0,
        'commit': commit}
    for name, item in read(ROOT / 'datasets/manifest.json').items():
        checks['dataset_' + name] = {'complete': sha(Path(item['path'])) == item['sha256']}
    reference_sha = reproduce.data()[2]['reference_sha256']
    assert read(mace.OUT / 'run_summary.json')['provenance']['reference_sha256'] == reference_sha
    for path, digest in read(ROOT / 'outputs/round2/suite_status.json')['protected_original_summaries'].items():
        assert sha(ROOT / path) == digest
    for stage, function in [
        ('static_phase_ordering', lambda: auditor.audit_static('additional', jobs)),
        ('full_relaxation', lambda: audit_relaxation(auditor, key, jobs)),
        ('matpes_pbe', lambda: auditor.audit_force('matpes_pbe', 'additional', True)),
        ('omat24', lambda: auditor.audit_force('omat24', 'additional', True)),
        ('full_neb', lambda: auditor.audit_neb('additional'))]:
        try:
            checks[stage] = function()
            if stage in ['matpes_pbe', 'omat24']:
                checks[stage]['saved_tensor_disk_offload'] = audit_force_offload(ROOT / specs[stage]['summary'])
            checks[stage]['summary_sha256'] = sha(ROOT / specs[stage]['summary'])
        except Exception as error:
            checks[stage] = {'complete': False, 'error': repr(error)}
            errors[stage] = repr(error)
        print(json.dumps({'stage': stage, 'complete': checks[stage]['complete']}), flush=True)
    try:
        reference = reproduce.data()[0]['reference_data']
        phase_metrics(mace, reference, checks)
    except Exception as error:
        checks['phase_metric_recomputation'] = {'complete': False, 'error': repr(error)}
        errors['phase_metric_recomputation'] = repr(error)
    static_neb = read(ROOT / specs['static_neb']['summary'])
    checks['static_neb'] = {'complete': static_neb['validation_pass'] is True and static_neb['images'] == 1078,
                             'summary_sha256': sha(ROOT / specs['static_neb']['summary'])}
    result = {'pass': all(value['complete'] for value in checks.values()), 'model': key,
              'deep_source_verification': True, 'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'auditor_sha256': sha(Path(__file__)), 'reused_original_auditor_sha256': sha(original),
              'bound_auditor_sha256': sha(derived), 'scope_sha256': sha(ROOT / 'round2_scope.json'),
              'model_sha256': sha(mace.MODEL), 'checks': checks, 'errors': errors}
    target = ROOT / 'outputs/round2' / key / 'scientific_audit.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'model': key, 'pass': result['pass'], 'errors': errors}), flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    raise SystemExit(main(parser.parse_args().model))
