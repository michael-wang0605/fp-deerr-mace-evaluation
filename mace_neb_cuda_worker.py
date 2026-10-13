"""One persistent, float64 CUDA calculator; tasks assigned by a single owner."""
import argparse
import json
from pathlib import Path
import time
import torch
import reproduce
import new_potential_static as mace
import neb_full_runner as full
import new_potential_neb as base
from relaxation_runner import atomic_json


def main(args):
    config=json.loads(args.config.read_text())
    profiles=config['execution_provenances']
    folder=reproduce.OUT/'mace_omat_0_small/neb/full'
    reference=reproduce.load(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    work=folder/'cuda_queue'
    work.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.9)
    mace.initialize('cuda')
    actual=base.provenance('full')
    actual['device']='cuda'
    assert actual==profiles['cuda']
    assert all(p.dtype==torch.float64 for model in mace.CALC.models for p in model.parameters())
    full.CALC,full.OUT=mace.CALC,folder
    full.EXECUTION={'device':'cuda','dtype':'float64','inference_optimized':False,
                    'validation_sha256':config['gpu_validation_sha256'],'TF32':False}
    atomic_json(work/'ready.json',{'provenance':profiles['cuda']})
    last=None
    while True:
        task_file=work/'task.json'
        if not task_file.exists():
            time.sleep(.25)
            continue
        task=json.loads(task_file.read_text())
        if task['task_id']==last:
            time.sleep(.25)
            continue
        if task.get('stop'):
            return
        key=task['pathway']
        assert key in config['gpu_eligible_pathways'] and key in reference['pathways']
        assert not (folder/'records'/(key.replace('|','_')+'.json')).exists()
        full.compute((key,reference['pathways'][key]),profiles['cuda'])
        last=task['task_id']
        atomic_json(work/'done.json',{'task_id':last,'pathway':key})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args())
