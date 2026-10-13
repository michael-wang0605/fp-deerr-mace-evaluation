"""Strict CPU/CUDA execution profiles for the same MACE full-NEB protocol."""
import json
from pathlib import Path


def scientific_core(profile):
    return {key:value for key,value in profile.items() if key not in ['device','versions']}


def register_display_name():
    """Extend display metadata in memory; keep upstream files and math intact."""
    import sys
    import reproduce
    sys.path.insert(0,str(reproduce.REPO/'Ion_migration_NEB/scripts'))
    import export_neb_leaderboard as exporter
    key,name='MACE_OMAT_0_SMALL','MACE-OMAT-0 Small'
    assert exporter.FP_DISPLAY_NAMES.get(key,name)==name
    exporter.FP_DISPLAY_NAMES[key]=name


def validate_profiles(profiles):
    assert set(profiles)=={'cpu','cuda'}
    cpu,gpu=profiles['cpu'],profiles['cuda']
    assert cpu['device']=='cpu' and gpu['device']=='cuda'
    assert scientific_core(cpu)==scientific_core(gpu)
    assert cpu['dtype']=='float64' and cpu['head']=='omat_pbe'
    assert cpu['cutoff_angstrom']==6.0 and cpu['dispersion_correction'] is False
    assert cpu['threads']==1 and cpu['protocol']=='full_fp_neb'
    assert cpu['versions']['torch']=='2.7.1'
    assert gpu['versions']['torch']=='2.7.1+cu126'
    assert {k:v for k,v in cpu['versions'].items() if k!='torch'}=={k:v for k,v in gpu['versions'].items() if k!='torch'}
    return scientific_core(cpu)


def record_environment(record,profiles):
    validate_profiles(profiles)
    matches=[name for name,profile in profiles.items() if record['provenance']==profile]
    assert len(matches)==1, 'A raw record must match one exact, declared execution profile'
    name=matches[0]
    execution=record.get('execution',{})
    assert execution.get('device')==name and execution.get('dtype')=='float64'
    assert execution.get('inference_optimized') is False
    return name


def load_records(folder,reference,profiles):
    complete={}
    environments={}
    for path in (Path(folder)/'records').glob('*.json'):
        record=json.loads(path.read_text())
        key=record['id']
        assert key in reference['pathways'] and key not in complete
        assert path.name==key.replace('|','_')+'.json'
        environments[key]=record_environment(record,profiles)
        complete[key]=record['result']
    return complete,environments
