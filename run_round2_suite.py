"""Finite, resource-bounded continuation of the requested model evaluations.

One CUDA owner, a configured CPU worker budget, and no
changes to benchmark precision or thresholds. All jobs use isolated adapters.
Failures remain visible; completed records are preserved. No email is sent.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import psutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/round2'
STATE = OUT / 'suite_status.json'
FIXTURES = ['245989|27', '20258|32']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    for attempt in range(80):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 79:
                raise
            time.sleep(min(.05 * (attempt + 1), .4))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus may temporarily deny replacement of an open
    # destination. Preserve the old complete JSON and retry only this I/O step.
    for attempt in range(80):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 79:
                raise
            time.sleep(min(.05 * (attempt + 1), .4))


def same_process(handle):
    try:
        return abs(psutil.Process(handle['pid']).create_time() - handle['created']) < .001
    except psutil.NoSuchProcess:
        return False


def original_snapshot():
    scope = read(ROOT / 'research_scope.json')
    audit = read(ROOT / 'outputs/prof_mo_review/scientific_audit.json')
    assert audit['pass'] and audit['deep_source_verification']
    assert audit['auditor_sha256'] == sha(ROOT / 'audit_research_results.py')
    summaries = {spec['summary']: sha(ROOT / spec['summary']) for spec in scope['required_evidence'].values()}
    for name, spec in scope['required_evidence'].items():
        assert audit['checks'][name]['summary_sha256'] == summaries[spec['summary']]
    return summaries


def build_jobs(models, relax_workers=6):
    jobs = {}
    def add(key, suffix, env, script, args=(), deps=(), cpu=0, gpu=False, priority=5, expected=None, optional=False):
        identifier = key + ':' + suffix
        jobs[identifier] = {'model': key, 'stage': suffix, 'env': env, 'script': script,
            'args': list(args), 'deps': [key + ':' + item for item in deps], 'cpu_slots': cpu,
            'gpu': gpu, 'priority': priority, 'expected': expected, 'optional': optional,
            'state': 'pending'}
    for key in models:
        adapter = 'outputs/round2/adapters_v2/' + key + '/'
        proof = 'outputs/round2/' + key + '/validation/'
        add(key, 'batch_validation', '.venv-mace', adapter + 'validate_mace_batches.py', cpu=1, priority=0,
            expected=proof + 'mace_batch_validation.json')
        add(key, 'cpu_profile', '.venv-mace', adapter + 'profile_mace_device.py', ['--device', 'cpu'],
            ['batch_validation'], cpu=1, priority=1, expected=proof + 'mace_device_profile_cpu.json')
        add(key, 'cuda_profile', '.venv-neb', adapter + 'profile_mace_device.py', ['--device', 'cuda'],
            ['cpu_profile'], gpu=True, priority=0, expected=proof + 'mace_cpu_cuda_validation.json')
        add(key, 'cuda_partitions', '.venv-neb', adapter + 'validate_mace_cuda_partitions.py',
            deps=['cuda_profile'], gpu=True, priority=1, expected=proof + 'mace_cuda_partition_validation.json')
        add(key, 'cuda_retry_controls', '.venv-neb', adapter + 'verify_mace_retry_controls.py',
            deps=['cuda_partitions'], gpu=True, priority=1, expected=proof + 'mace_cuda_retry_control_validation.json')
        add(key, 'static', '.venv-mace', adapter + 'new_potential_static.py', ['--workers', '2'],
            ['batch_validation'], cpu=2, priority=2, expected='outputs/' + key + '/summary.json')
        add(key, 'static_neb', '.venv-mace-neb', adapter + 'new_potential_neb.py',
            ['--mode', 'static', '--workers', '2'], ['batch_validation'], cpu=2, priority=3,
            expected='outputs/' + key + '/neb/static/summary.json')
        fixture_args = ['--model', key, '--workers', '2']
        for path in FIXTURES:
            fixture_args += ['--pathway', path]
        add(key, 'neb_fixtures', '.venv-mace-neb', 'round2_neb_fixtures.py', fixture_args,
            ['batch_validation'], cpu=2, priority=3, expected='outputs/' + key + '/neb/full/fixtures_validation.json')
        add(key, 'neb_cuda_validation', '.venv-neb', adapter + 'validate_mace_cuda_neb.py',
            sum((['--pathway', path] for path in FIXTURES), []), ['neb_fixtures'],
            gpu=True, priority=2, expected=proof + 'mace_cuda_neb_validation.json', optional=True)
        add(key, 'relaxation', '.venv-mace', adapter + 'new_potential_relaxation.py', ['--workers', '2'],
            ['static'], cpu=2, priority=4, expected='outputs/' + key + '/relaxation/summary.json')
        for dataset, priority in [('omat24', 3), ('matpes_pbe', 4)]:
            add(key, dataset, '.venv-neb', adapter + 'new_potential_force.py',
                ['--dataset', dataset, '--device', 'cuda', '--workers', '1'],
                ['cuda_retry_controls'], gpu=True, priority=priority,
                expected='outputs/force_benchmarks/' + dataset + '/' + key + '_cuda/summary.json')
        add(key, 'full_neb', '.venv-mace-neb', adapter + 'new_potential_neb_mixed.py',
            ['--cpu-workers', '2'], ['static_neb', 'neb_cuda_validation'], cpu=2, gpu=True,
            priority=5, expected='outputs/' + key + '/neb/summary.json')
    # A separate, checksum-bound memory policy may resume the failed MH1 force
    # run. Its calculator, graph conversion, float64 arithmetic and shard writer
    # are the original adapters; only singleton OOM storage changes.
    proof_path = OUT / 'mace_mh_1_omat_pbe/validation/mace_cuda_disk_offload_validation.json'
    if proof_path.exists():
        proof = read(proof_path)
        controls = read(proof_path.with_name('mace_cuda_disk_offload_retry_validation.json'))
        source = ROOT / 'round2_mh1_force_offload.py'
        assert proof['validation_pass'] and controls['validation_pass']
        assert proof['wrapper_sha256'] == controls['wrapper_sha256'] == sha(source)
        assert proof['offload_implementation_sha256'] == sha(ROOT / 'cuda_saved_tensor_offload.py')
        deterministic = [row for row in proof['checks'] if row['deterministic']]
        assert {row['batch_size'] for row in deterministic} == {1, 16}
        assert all(value['equal'] for row in deterministic for value in row['offload_difference'].values())
        assert proof['dense_case']['original_singleton_oom'] and proof['dense_case']['all_fields_finite']
        target = jobs['mace_mh_1_omat_pbe:matpes_pbe']
        target['script'] = 'round2_mh1_force_offload.py'
        target['memory_policy'] = {'saved_tensor_storage': 'disk on singleton OOM only',
                                   'arithmetic_device': 'cuda', 'dtype': 'float64',
                                   'numeric_validation_sha256': sha(proof_path),
                                   'control_validation_sha256': sha(proof_path.with_name('mace_cuda_disk_offload_retry_validation.json'))}
    acceleration = OUT / 'relaxation_acceleration/validation.json'
    if acceleration.exists():
        proof = read(acceleration)
        assert proof['pass']
        assert all(sha(ROOT / name) == digest for name, digest in proof['source_files_sha256'].items())
        medium, mh = 'mace_omat_0_medium', 'mace_mh_1_omat_pbe'
        jobs[medium + ':relaxation'].update(
            script='round2_joint_relaxation.py', env='.venv-mace', args=['--workers', str(relax_workers)], cpu_slots=relax_workers,
            deps=[medium + ':static', mh + ':static'],
            execution_route='Six RAM-bounded CPU workers and a separately scheduled validated CUDA assistant')
        jobs[mh + ':relaxation'].update(script='round2_stage_ready.py', env='.venv', args=['--model', mh],
                                        cpu_slots=0, deps=[medium + ':relaxation'])
        add(mh, 'cuda_relaxation_assist', '.venv-relax-gpu', 'round2_relax_gpu_worker.py',
            deps=['matpes_pbe'], gpu=True, priority=4,
            expected='outputs/round2/relaxation_acceleration/gpu_assist_summary.json')
        jobs[mh + ':cuda_relaxation_assist']['deps'].append(medium + ':matpes_pbe')
    return jobs


def validate_completed(job, models):
    if job['expected'] is None:
        return True
    row = read(ROOT / job['expected'])
    key, stage = job['model'], job['stage']
    model = models[key]
    if stage == 'batch_validation':
        assert row['validation_pass'] and row['weights_sha256'] == model['sha256']
        assert row['implementation_sha256'] == sha(ROOT / 'outputs/round2/adapters_v2' / key / 'mace_batch_inference.py')
        assert row['structures'] == 33 and 16 in [trial['batch_size'] for trial in row['trials']]
    elif stage == 'cpu_profile':
        assert row['weights_sha256'] == model['sha256'] and row['device'] == 'cpu'
        assert row['implementation_sha256'] == sha(ROOT / 'outputs/round2/adapters_v2' / key / 'mace_batch_inference.py')
    elif stage in ['cuda_profile', 'cuda_partitions', 'cuda_retry_controls']:
        assert row['validation_pass'] is True
        if 'weights_sha256' in row:
            assert row['weights_sha256'] == model['sha256']
        if 'runner_sha256' in row:
            assert row['runner_sha256'] == sha(ROOT / 'outputs/round2/adapters_v2' / key / 'new_potential_force.py')
    elif stage in ['neb_fixtures', 'neb_cuda_validation']:
        assert row['pass'] is True
        if stage == 'neb_cuda_validation':
            assert row['provenance']['weights_sha256'] == model['sha256']
            assert all(sha(ROOT / name) == digest for name, digest in row['scientific_source_sha256'].items())
    elif stage in ['omat24', 'matpes_pbe']:
        assert row['full_dataset_complete'] is True
        assert row['structures'] == (117004 if stage == 'omat24' else 434712)
        assert row['provenance']['weights_sha256'] == model['sha256']
    elif stage == 'static':
        assert row['completed'] == 6697 and row['failed'] == 0
        assert row['provenance']['weights_sha256'] == model['sha256']
    elif stage == 'relaxation':
        assert row['completed'] == 6697 and row['failed'] == 0
        assert row['provenance']['weights_sha256'] == model['sha256']
    elif stage == 'static_neb':
        assert row['pathways'] == 154 and row['images'] == 1078 and row['validation_pass']
    elif stage == 'full_neb':
        assert row['pathways'] == 154 and row['failed'] == 0 and row['validation_pass']
    elif stage == 'cuda_relaxation_assist':
        assert row['complete'] and row['engine_sha256'] == sha(ROOT / 'round2_relax_engine.py')
        assert row['validation_sha256'] == sha(OUT / 'relaxation_acceleration/validation.json')
    return True


def configure_full_neb(job, jobs):
    key = job['model']
    validator = jobs[key + ':neb_cuda_validation']
    if validator['state'] == 'optional_failed':
        job.update(script='outputs/round2/adapters_v2/' + key + '/new_potential_neb.py',
                   args=['--mode', 'full', '--workers', '4', '--active-paths', '4'],
                   cpu_slots=4, gpu=False, execution_route='CPU; CUDA trajectory validation did not pass')
        return
    proof_path = ROOT / validator['expected']
    proof = read(proof_path)
    record = read(ROOT / 'outputs' / key / 'neb/full/records' / (FIXTURES[0].replace('|', '_') + '.json'))
    reference = __import__('reproduce').load(ROOT / 'FP-DeErr/Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    completed = {read(path)['id'] for path in (ROOT / 'outputs' / key / 'neb/full/records').glob('*.json')}
    config = {'execution_provenances': {'cpu': record['provenance'], 'cuda': proof['provenance']},
              'gpu_validation_sha256': sha(proof_path), 'external_active_pathways': [],
              'gpu_eligible_pathways': [name for name in reference['pathways'] if name not in completed]}
    path = OUT / key / 'mixed_neb_config.json'
    atomic(path, config)
    workers = 4 if (OUT / 'relaxation_acceleration/validation.json').exists() else 2
    job['cpu_slots'] = workers
    job['args'] = ['--config', str(path), '--cpu-workers', str(workers)]
    job['execution_route'] = 'Disjoint CPU/CUDA queues after per-model full-trajectory validation'


class AdoptedJob:
    """Observe an existing calculation without restarting or cancelling it.

    Its former parent's exit code may be unavailable. Completion then requires
    the same expected-output validation used for cached completed stages.
    """
    def __init__(self, job, models):
        self.job, self.models = job, models

    def poll(self):
        if same_process(self.job['process']):
            return None
        try:
            validate_completed(self.job, self.models)
            self.job['completion_basis'] = 'Adopted process ended; expected complete artifact validated, exit code unavailable'
            return 0
        except Exception as error:
            self.job['adopted_completion_error'] = repr(error)
            return 1


def main(args):
    previous = None
    if STATE.exists():
        previous = read(STATE)
        assert not same_process(previous['controller']), 'A controller already owns this queue'
        live = [row for row in previous['jobs'].values() if row.get('state') == 'running' and same_process(row['process'])]
        assert not live or args.adopt_running, 'Recorded jobs are active; inspect and explicitly adopt them rather than duplicate them'
        if live:
            assert all((row['stage'] == 'relaxation' and not row['gpu']) or row['stage'] == 'full_neb' for row in live), 'Only inspected relaxation or full NEB jobs may be adopted'
        assert args.resume, 'Use --resume only after inspecting retained results and confirming the old owner/jobs ended.'
    else:
        assert not args.resume, 'No existing queue is available to resume.'
    models = read(OUT / 'models.json')
    adapters = read(OUT / 'adapter_manifest_v2.json')
    for entry in adapters['models'].values():
        for row in entry['adapters'].values():
            assert sha(ROOT / row['path']) == row['sha256']
    protected = original_snapshot()
    jobs = build_jobs(models, args.relax_workers)
    # Reuse only a current checksum-bound validation; preserve outdated pilot
    # proofs before the current isolated adapter writes a replacement.
    for key in models:
        job = jobs[key + ':batch_validation']
        path = ROOT / job['expected']
        if path.exists():
            try:
                validate_completed(job, models)
                job.update(state='complete', reused=True, proof_sha256=sha(path))
            except AssertionError:
                history = OUT / key / 'validation_history' / (sha(path) + '.json')
                atomic(history, read(path))
    recovery = None
    if previous is not None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        history = OUT / 'recovery_history' / stamp
        history.mkdir(parents=True)
        # Keep the exact stale status/error records and inspect actual complete
        # outputs; an orphaned child can finish after its parent controller dies.
        for name in ['suite_status.json', 'suite_status.json.tmp', 'finalization_status.json',
                     'controller_process.json', 'finalization_process.json']:
            path = OUT / name
            if path.exists():
                (history / name).write_bytes(path.read_bytes())
        retained = []
        for identifier, job in jobs.items():
            path = ROOT / job['expected']
            if not path.exists():
                continue
            validate_completed(job, models)
            job.update(state='complete', reused=True, proof_sha256=sha(path))
            retained.append(identifier)
        recovery = {'previous_controller': previous['controller'], 'history': history.relative_to(ROOT).as_posix(),
                    'reused_verified_jobs': retained, 'scientific_jobs_cancelled': False,
                    'reason': 'Explicit queue recovery; completed outputs verified and active CPU jobs preserved.'}
        atomic(history / 'resume_validation.json', recovery)
    handle = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
    state = {'state': 'running', 'controller': handle, 'started_utc': now(), 'updated_utc': now(),
             'max_cpu_slots': args.cpu_slots, 'max_gpu_owners': 1, 'models': list(models),
             'protected_original_summaries': protected, 'adapter_manifest_sha256': sha(OUT / 'adapter_manifest_v2.json'),
             'jobs': jobs, 'no_new_email_sent_by_controller': True, 'complete': False}
    if recovery:
        state['recovery'] = recovery
    live = {}
    if previous is not None and args.adopt_running:
        adopted = []
        for identifier, prior in previous['jobs'].items():
            if prior['state'] != 'running' or not same_process(prior['process']):
                continue
            job = jobs[identifier]
            if job['stage'] == 'full_neb':
                assert prior['script'] == 'outputs/round2/adapters_v2/' + job['model'] + '/new_potential_neb_mixed.py'
                config_path = OUT / job['model'] / 'mixed_neb_config.json'
                config = read(config_path)
                validator = jobs[job['model'] + ':neb_cuda_validation']
                assert config['gpu_validation_sha256'] == sha(ROOT / validator['expected'])
                workers = 4 if (OUT / 'relaxation_acceleration/validation.json').exists() else 2
                job.update(args=['--config', str(config_path), '--cpu-workers', str(workers)], cpu_slots=workers)
            assert all(job[field] == prior[field] for field in ['model', 'stage', 'script', 'env', 'args', 'cpu_slots', 'gpu', 'expected'])
            assert job['state'] == 'pending' and job['stage'] in ['relaxation', 'full_neb']
            job.update(prior)
            job['adopted_by_controller'] = handle
            live[identifier] = AdoptedJob(job, models), (ROOT / job['log']).open('ab', buffering=0)
            adopted.append(identifier)
        recovery['adopted_running_jobs'] = adopted
        atomic(history / 'resume_validation.json', recovery)
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    while True:
        for identifier, (process, stream) in list(live.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            job = jobs[identifier]
            job.update(exit_code=code, finished_utc=now())
            try:
                assert code == 0, 'Process exited ' + str(code)
                validate_completed(job, models)
                job.update(state='complete', proof_sha256=sha(ROOT / job['expected']))
            except Exception as error:
                job.update(state='optional_failed' if job['optional'] else 'failed', error=repr(error))
            del live[identifier]
            print(json.dumps({'job': identifier, 'state': job['state'], 'exit_code': code}), flush=True)
        # Original scientific evidence is checked at each process boundary.
        assert all(sha(ROOT / path) == digest for path, digest in protected.items())
        active_cpu = sum(jobs[name]['cpu_slots'] for name in live)
        active_gpu = any(jobs[name]['gpu'] for name in live)
        for identifier, job in sorted(jobs.items(), key=lambda item: item[1]['priority']):
            if job['state'] != 'pending':
                continue
            deps = [jobs[name]['state'] for name in job['deps']]
            if any(value in ['failed', 'blocked'] for value in deps):
                job.update(state='blocked', error='Required dependency failed; inspect its log and retained results.')
                continue
            if not all(value in ['complete', 'optional_failed'] for value in deps):
                continue
            if job['stage'] == 'full_neb' and 'execution_route' not in job:
                configure_full_neb(job, jobs)
            if active_cpu + job['cpu_slots'] > args.cpu_slots or (job['gpu'] and active_gpu):
                continue
            available = psutil.virtual_memory().available / 2 ** 30
            if available < (7 if job['gpu'] else 5):
                state['waiting_for_memory_GiB'] = round(available, 2)
                continue
            command = [str(ROOT / job['env'] / 'Scripts/python.exe'), '-u', str(ROOT / job['script']), *job['args']]
            log_path = logs / (identifier.replace(':', '_') + '.log')
            stream = log_path.open('ab', buffering=0)
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
            job.update(state='running', started_utc=now(), command=command,
                       log=log_path.relative_to(ROOT).as_posix(),
                       process={'pid': process.pid, 'created': psutil.Process(process.pid).create_time()})
            live[identifier] = process, stream
            active_cpu += job['cpu_slots']
            active_gpu = active_gpu or job['gpu']
            print(json.dumps({'job': identifier, 'state': 'running', 'pid': process.pid}), flush=True)
        state['updated_utc'] = now()
        state['active_cpu_slots'] = active_cpu
        state['active_gpu_owners'] = int(active_gpu)
        state['stage_counts'] = {name: sum(row['state'] == name for row in jobs.values())
                                 for name in ['pending', 'running', 'complete', 'failed', 'optional_failed', 'blocked']}
        atomic(STATE, state)
        if not live and not any(row['state'] == 'pending' for row in jobs.values()):
            failed = [name for name, row in jobs.items() if row['state'] in ['failed', 'blocked']]
            state.update(state='needs_attention' if failed else 'calculations_complete_audit_required',
                         finished_utc=now(), incomplete=failed,
                         final_audit_and_leaderboard_submission_required=True)
            atomic(STATE, state)
            return 1 if failed else 0
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-slots', type=int, default=8 if (OUT / 'relaxation_acceleration/validation.json').exists() else 4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--adopt-running', action='store_true', help='Adopt inspected live relaxation or full NEB jobs during a scheduler handoff.')
    parser.add_argument('--relax-workers', type=int, choices=[4, 6], default=6)
    args = parser.parse_args()
    assert 1 <= args.cpu_slots <= 8
    assert not args.adopt_running or args.resume
    try:
        raise SystemExit(main(args))
    except Exception as error:
        if STATE.exists():
            state = read(STATE)
            if state.get('controller', {}).get('pid') == os.getpid():
                state.update(state='needs_attention', error=repr(error), updated_utc=now())
                atomic(STATE, state)
        raise
