"""Disjoint CPU/CUDA queues with unchanged full.compute and NEB template."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import datetime
import json
from pathlib import Path
import subprocess
import time
import uuid
import psutil
import reproduce
import new_potential_neb as base
from mace_neb_provenance import load_records,validate_profiles,register_display_name
from relaxation_runner import atomic_json


class FinishCudaWorker:
    def __init__(self,process,queue):
        self.process,self.queue=process,queue
    def __enter__(self):return self
    def __exit__(self,kind,value,traceback):
        if self.process.poll() is None:
            atomic_json(self.queue/'task.json',{'task_id':uuid.uuid4().hex,'stop':True})
        # The worker reads stop only after its current trajectory finishes.
        self.process.wait()
        if kind is None:
            assert self.process.returncode==0


def analyze_existing(config_path):
    """Analyze complete mixed results while retaining truthful device metadata."""
    config=json.loads(config_path.read_text())
    profiles=config['execution_provenances']
    core=validate_profiles(profiles)
    proof_path=reproduce.OUT/'mace_cuda_neb_validation.json'
    proof=json.loads(proof_path.read_text())
    assert proof['pass'] and proof['provenance']==profiles['cuda']
    assert reproduce.sha(proof_path)==config['gpu_validation_sha256']
    assert all(reproduce.sha(reproduce.ROOT/name)==digest for name,digest in proof['scientific_source_sha256'].items())
    assert profiles['cpu']==base.provenance('full')
    folder=base.folder('full')
    reference=reproduce.load(base.REFERENCE)
    completed,environments=load_records(folder,reference,profiles)
    assert set(completed)==set(reference['pathways']) and len(completed)==154
    assert not any(r['neb_status']['calculation_status']=='failed' for r in completed.values())
    run=json.loads((folder/'run_summary.json').read_text())
    assert run['provenance']==core and run['execution_provenances']==profiles
    assert run['path_execution_environment']==environments
    assert run['gpu_validation_sha256']==config['gpu_validation_sha256']
    results=json.loads((folder/'results.json').read_text())
    assert results['provenance']==core and results['execution_provenances']==profiles
    standardized=results['models'][base.KEY]['full_fp_neb']['pathways']
    assert set(standardized)==set(completed)
    for key,record in completed.items():
        assert standardized[key]=={field:record[field] for field in ['identifiers','final_fp_neb_images','neb_status','calculation_metadata']}
    original_provenance=base.provenance
    try:
        register_display_name()
        base.provenance=lambda mode: {**core,'execution_provenances':profiles} if mode=='full' else original_provenance(mode)
        base.analyze()
        report_path=base.ROOT/'report.md'
        report=report_path.read_text(encoding='utf-8')
        old='float64 CPU MACE, omat_pbe head, cutoff 6 Å, no added dispersion.'
        assert old in report
        report_path.write_text(report.replace(old,
            'float64 MACE on CPU and validated CUDA, omat_pbe head, cutoff 6 Å, no added dispersion. '
            'Each raw pathway retains its actual execution device and package versions; two complete CPU/CUDA trajectory comparisons passed before GPU production.'),encoding='utf-8')
    finally:
        base.provenance=original_provenance


def main(args):
    config=json.loads(args.config.read_text())
    profiles=config['execution_provenances']
    core=validate_profiles(profiles)
    assert core=={k:v for k,v in base.provenance('full').items() if k not in ['device','versions']}
    assert profiles['cpu']==base.provenance('full')
    gpu_proof=reproduce.OUT/'mace_cuda_neb_validation.json'
    proof=json.loads(gpu_proof.read_text())
    assert proof['pass'] and proof['provenance']==profiles['cuda']
    assert reproduce.sha(gpu_proof)==config['gpu_validation_sha256']
    assert all(reproduce.sha(reproduce.ROOT/name)==digest for name,digest in proof['scientific_source_sha256'].items())
    folder=base.folder('full')
    reference=reproduce.load(base.REFERENCE)
    external=set(config['external_active_pathways'])
    completed,environments=load_records(folder,reference,profiles)
    pending=[(key,value) for key,value in reference['pathways'].items() if key not in completed and key not in external]
    assert set(config['gpu_eligible_pathways'])=={key for key,_ in pending}
    plan={'provenance':core,'pathways':list(reference['pathways']),
          'execution_provenances':profiles,'cpu_workers':args.cpu_workers,'gpu_workers':1,
          'external_active_pathways':sorted(external)}
    atomic_json(folder/'plan.json',plan)
    queue=folder/'cuda_queue';queue.mkdir(parents=True,exist_ok=True)
    assert not (queue/'ready.json').exists() and not (queue/'task.json').exists()
    log_path=reproduce.OUT/'job_logs/mace_neb_cuda_worker.log'
    with log_path.open('ab') as log:
        gpu=subprocess.Popen([str(reproduce.ROOT/'.venv-neb/Scripts/python.exe'),'-u',str(reproduce.ROOT/'mace_neb_cuda_worker.py'),
            '--config',str(args.config)],cwd=reproduce.ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    atomic_json(folder/'cuda_worker_process.json',{'pid':gpu.pid,'created':psutil.Process(gpu.pid).create_time(),'log':str(log_path)})
    def gpu_alive():
        assert gpu.poll() is None, 'CUDA worker ended; inspect its log and retained trajectory before resuming'
    while not (queue/'ready.json').exists():
        gpu_alive();time.sleep(.5)
    assert json.loads((queue/'ready.json').read_text())['provenance']==profiles['cuda']
    begun=time.perf_counter();futures={};gpu_job=None;submitted=set()
    def publish():
        nonlocal completed,environments
        completed,environments=load_records(folder,reference,profiles)
        atomic_json(folder/'progress.json',{'completed':len(completed),'planned':154,
            'converged':sum(r['neb_status']['neb_converged'] is True for r in completed.values()),
            'non_converged':sum(r['neb_status']['neb_converged'] is False for r in completed.values()),
            'failed':sum(r['neb_status']['calculation_status']=='failed' for r in completed.values()),
            'elapsed_seconds':time.perf_counter()-begun,'execution_counts':{name:list(environments.values()).count(name) for name in profiles}})
    with FinishCudaWorker(gpu,queue), ProcessPoolExecutor(max_workers=args.cpu_workers,initializer=base.initialize,initargs=('full',)) as pool:
        while pending or futures or gpu_job:
            gpu_alive()
            if gpu_job is not None and (queue/'done.json').exists():
                done=json.loads((queue/'done.json').read_text())
                if done['task_id']==gpu_job['task_id']:
                    publish()
                    result=completed[gpu_job['pathway']]
                    assert result['neb_status']['calculation_status']!='failed', ('CUDA path failed; preserved for investigation',gpu_job,result['neb_status'])
                    gpu_job=None
            if gpu_job is None and pending:
                # Favor medium/small paths for the GPU; large dense BFGS
                # eigensolves remain CPU work even with GPU model inference.
                idx=min(range(len(pending)),key=lambda i:len(pending[i][1]['dft_neb_reference']['images'][0]['structure']['sites']))
                job=pending.pop(idx)
                assert job[0] not in submitted and job[0] not in external
                submitted.add(job[0]);gpu_job={'task_id':uuid.uuid4().hex,'pathway':job[0]}
                atomic_json(queue/'task.json',gpu_job)
            cap_file=folder/'mixed_cpu_active_paths.json'
            cap=json.loads(cap_file.read_text())['active_paths'] if cap_file.exists() else args.cpu_workers
            assert 1<=cap<=args.cpu_workers
            while len(futures)<cap and pending:
                job=pending.pop(-1)
                assert job[0] not in submitted and job[0] not in external
                submitted.add(job[0]);futures[pool.submit(base.full.compute,job,profiles['cpu'])]=job[0]
            if futures:
                finished,_=wait(futures,return_when=FIRST_COMPLETED,timeout=1)
                for future in finished:
                    future.result();key=futures.pop(future);publish()
                    assert completed[key]['neb_status']['calculation_status']!='failed', ('CPU path failed; preserved for investigation',key)
            else:
                time.sleep(.5)
    while True:
        publish()
        missing=set(reference['pathways'])-set(completed)
        assert missing.issubset(external)
        if not missing:break
        print(json.dumps({'waiting_for_original_trajectories':sorted(missing)}),flush=True)
        time.sleep(15)
    assert len(completed)==154 and not any(r['neb_status']['calculation_status']=='failed' for r in completed.values())
    atomic_json(folder/'run_summary.json',{'provenance':core,'execution_provenances':profiles,
        'path_execution_environment':environments,'gpu_validation_sha256':config['gpu_validation_sha256'],
        'completed':154,'planned':154,'cpu_workers':args.cpu_workers,'gpu_workers':1})
    result={key:reference[key] for key in ['schema_version','component','units']}
    result.update(dataset_name='ion_migration_neb_results',provenance=core,execution_provenances=profiles,
        models={base.KEY:{'full_fp_neb':{'pathways':{key:{field:r[field] for field in ['identifiers','final_fp_neb_images','neb_status','calculation_metadata']} for key,r in completed.items()},'unsuccessful_pathways':{}}}})
    atomic_json(folder/'results.json',result)
    # Reuse unchanged public analysis and summary logic; the override describes
    # execution metadata only. No calculation or metric function is replaced.
    analyze_existing(args.config)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['full'],default='full')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cpu-workers',type=int,default=4)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    analyze_existing(args.config) if args.analyze else main(args)
