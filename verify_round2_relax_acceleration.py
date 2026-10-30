"""Gate the accelerated scheduler on complete trajectory and ownership checks."""
import json
from pathlib import Path
import numpy as np
import round2_relax_engine as engine
from round2_relax_engine import suite
from round2_joint_relaxation import pick


def main():
    cpu, gpu = (suite.read(engine.OUT / ('calibration_' + device + '.json')) for device in ['cpu', 'cuda'])
    assert cpu['pass'] and gpu['pass']
    digest = suite.sha(Path(engine.__file__))
    assert cpu['engine_sha256'] == gpu['engine_sha256'] == digest
    profiles, speedups = {}, {}
    for key in engine.MODELS:
        profiles[key] = {'cpu': cpu['models'][key]['profile'], 'cuda': gpu['models'][key]['profile']}
        assert profiles[key]['cpu'] == engine.legacy(key)
        assert engine.core(profiles[key]['cpu']) == engine.core(profiles[key]['cuda'])
        expected = suite.read(suite.OUT / key / 'validation/relaxation_acceleration_fixture_ids.json')
        assert [row['id'] for row in gpu['models'][key]['cases']] == expected
        for a, b in zip(cpu['models'][key]['cases'], gpu['models'][key]['cases']):
            assert a['id'] == b['id'] and a['steps'] == b['steps']
            cpu_path, gpu_path = (engine.ROOT / row['path'] for row in [a, b])
            first, second = (suite.read(path.with_name(path.stem + '.trace.json')) for path in [cpu_path, gpu_path])
            assert len(first) == len(second)
            assert [(row['step'], row['dt'], row['a']) for row in first] == [(row['step'], row['dt'], row['a']) for row in second]
            for field, tolerance in [('positions', 1e-6), ('forces', 1e-7), ('energy', 1e-7)]:
                assert max(float(np.max(np.abs(np.asarray(x[field]) - np.asarray(y[field])))) for x, y in zip(first, second)) < tolerance
        assert gpu['models'][key]['forced_storage']['pass']
        speedups[key] = sum(row['seconds'] for row in cpu['models'][key]['cases']) / sum(row['seconds'] for row in gpu['models'][key]['cases'])
        assert speedups[key] > 1.2
    pending = {(key, str(i)): (str(i), {'initial_structure': {'sites': [0] * (i + 1)}}, {}) for key in engine.MODELS for i in range(40)}
    assigned = []
    while pending:
        item = pick(pending, cuda=len(assigned) % 3 == 0)
        assigned.append(item); pending.pop(item)
    assert len(assigned) == len(set(assigned)) == 80
    suite.original_snapshot()
    sources = ['round2_relax_engine.py', 'round2_joint_relaxation.py', 'round2_relax_gpu_worker.py',
               'round2_relax_analysis.py', 'calibrate_round2_relaxation.py', 'cuda_saved_tensor_offload.py',
               'relaxation_runner.py']
    report = {'pass': True, 'gpu_enabled': True, 'prepared_utc': suite.now(),
              'engine_sha256': digest, 'profiles': profiles, 'calibration_speedups': speedups,
              'cpu_workers_max': 6, 'minimum_available_RAM_GiB_for_more_CPU_tasks': 5,
              'unchanged_optimizer_source_sha256': profiles[next(iter(profiles))]['cpu']['energy_func_sha256'],
              'source_files_sha256': {name: suite.sha(engine.ROOT / name) for name in sources},
              'calibration_files_sha256': {device: suite.sha(engine.OUT / ('calibration_' + device + '.json')) for device in ['cpu', 'cuda']},
              'ownership_test': {'jobs': 80, 'duplicates': 0, 'omitted': 0}}
    suite.atomic(engine.OUT / 'validation.json', report)
    print(json.dumps({'pass': True, 'speedups': speedups, 'cpu_workers': 6, 'gpu_workers': 1}), flush=True)


if __name__ == '__main__':
    main()
