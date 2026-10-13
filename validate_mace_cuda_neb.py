"""Compare separate CUDA full-NEB reruns with retained CPU trajectories.

Diagnostic only. Production paths, metadata and scientific source files are not
modified, and GPU results are not imported into the benchmark.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
import reproduce
import new_potential_static as mace
import neb_full_runner as full
from relaxation_runner import atomic_json


def main(args):
    production = reproduce.OUT/'mace_omat_0_small/neb/full'
    output = reproduce.OUT/'mace_cuda_neb_validation'
    output.mkdir(parents=True,exist_ok=True)
    protected = {p.name:reproduce.sha(p) for p in (production/'records').glob('*.json')}
    scientific_files = ['new_potential_neb.py','neb_full_runner.py','new_potential_static.py',
                        'relaxation_runner.py','models/mace-omat-0-small.model']
    source_hashes = {name:reproduce.sha(reproduce.ROOT/name) for name in scientific_files}
    reference = reproduce.load(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.9)
    mace.initialize('cuda')
    assert all(parameter.dtype==torch.float64 for model in mace.CALC.models for parameter in model.parameters())
    full.CALC,full.OUT = mace.CALC,output
    full.EXECUTION = {'device':'cuda','dtype':'float64','inference_optimized':False,'purpose':'Isolated validation only'}
    cases=[]
    for key in args.pathway:
        cpu_path = production/'records'/(key.replace('|','_')+'.json')
        cpu = json.loads(cpu_path.read_text())
        assert cpu['result']['neb_status']['neb_converged'] is True
        provenance = dict(cpu['provenance'])
        assert provenance['weights_sha256']==mace.MODEL_SHA
        assert provenance['template_sha256']==hashlib.sha256(full.template().encode()).hexdigest()
        assert provenance['reference_sha256']==reproduce.sha(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
        provenance['device']='cuda'
        provenance['versions']={name:importlib.metadata.version(name) for name in provenance['versions']}
        print(json.dumps({'state':'running','pathway':key,'cpu_seconds':cpu['result']['seconds']}),flush=True)
        begun=time.perf_counter()
        full.compute((key,reference['pathways'][key]),provenance)
        gpu=json.loads(full.record_path(key).read_text())
        source,target=cpu['result'],gpu['result']
        statuses=target['neb_status']
        assert statuses['calculation_status']!='failed',statuses
        assert set(source['final_fp_neb_images'])==set(target['final_fp_neb_images'])=={str(i) for i in range(7)}
        differences={'energy_total_eV':0.,'forces_eV_per_A':0.,'positions_A':0.,'fixed_cell_A':0.}
        for index,old in source['final_fp_neb_images'].items():
            new=target['final_fp_neb_images'][index]
            differences['energy_total_eV']=max(differences['energy_total_eV'],abs(old['fp_energy_total_eV']-new['fp_energy_total_eV']))
            for field,label in [('fp_forces_eV_per_angstrom','forces_eV_per_A')]:
                a,b=np.asarray(old[field]),np.asarray(new[field])
                assert a.shape==b.shape and np.isfinite(b).all()
                differences[label]=max(differences[label],float(np.max(np.abs(a-b))))
            a=np.asarray([site['xyz'] for site in old['structure']['sites']])
            b=np.asarray([site['xyz'] for site in new['structure']['sites']])
            differences['positions_A']=max(differences['positions_A'],float(np.max(np.abs(a-b))))
            a=np.asarray(old['structure']['lattice']['matrix']);b=np.asarray(new['structure']['lattice']['matrix'])
            differences['fixed_cell_A']=max(differences['fixed_cell_A'],float(np.max(np.abs(a-b))))
        status_keys=['neb_converged','endpoint_converged','optimizer_steps','endpoint_start_steps','endpoint_end_steps','autosort_tol_used']
        same_status={name:source['neb_status'][name]==statuses[name] for name in status_keys}
        passed=(all(same_status.values()) and differences['energy_total_eV']<1e-7
                and differences['forces_eV_per_A']<1e-7 and differences['positions_A']<1e-6 and differences['fixed_cell_A']==0)
        case={'pathway':key,'pass':passed,'cpu_record_sha256':reproduce.sha(cpu_path),
              'gpu_record_sha256':reproduce.sha(full.record_path(key)),
              'cpu_seconds':source['seconds'],'gpu_seconds':time.perf_counter()-begun,
              'max_abs_differences':differences,'same_status_and_iteration_counts':same_status,
              'gpu_status':statuses,'cpu_status':source['neb_status']}
        cases.append(case)
        atomic_json(output/'progress.json',{'cases':cases,'complete':len(cases)==len(args.pathway)})
        print(json.dumps(case),flush=True)
    assert all(reproduce.sha(production/'records'/name)==digest for name,digest in protected.items())
    assert all(reproduce.sha(reproduce.ROOT/name)==digest for name,digest in source_hashes.items())
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':all(c['pass'] for c in cases),
            'scope':'Full-trajectory CPU/CUDA comparisons on the listed paths only; not a full-population guarantee',
            'production_unchanged':True,'production_GPU_paths_added':False,'cases':cases,
            'validator_sha256':reproduce.sha(Path(__file__)),'scientific_source_sha256':source_hashes,
            'cuda_device':torch.cuda.get_device_name(),'dtype':'float64','TF32':False,
            'protocol_changed':False,'provenance':provenance}
    atomic_json(reproduce.OUT/'mace_cuda_neb_validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pathway',action='append',required=True)
    main(parser.parse_args())
