"""Independently audit every completed MACE MatPES source row and force metric."""
import datetime
import json
import reproduce
import audit_research_results as auditor
from relaxation_runner import atomic_json


def main():
    summary = reproduce.OUT/'force_benchmarks/matpes_pbe/mace_omat_0_small_cuda/summary.json'
    assert summary.exists(), 'The complete-population summary is still pending'
    data = json.loads(summary.read_text())
    assert data['full_dataset_complete'] and data['structures']==434712
    result = auditor.audit_force('matpes_pbe','additional',True)
    assert result['complete'] and result['structures']==434712
    proof = {'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'pass':True,'deep_source':True,'dataset':'matpes_pbe','model':'MACE-OMAT-0 Small',
             'scope':'Full MatPES population, every original ID and DFT force, all canonical arrays and displayed metrics',
             'auditor_sha256':reproduce.sha(reproduce.ROOT/'audit_research_results.py'),
             'validator_sha256':reproduce.sha(reproduce.ROOT/'validate_mace_matpes_full_results.py'),
             'summary_sha256':reproduce.sha(summary),'result':result}
    atomic_json(reproduce.OUT/'mace_matpes_full_validation.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()
