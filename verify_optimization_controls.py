"""Check retry coverage and resumed input identity without inference."""
import itertools
import json
import torch
import force_benchmark_runner as runner
from relaxation_runner import atomic_json


class RetryModel:
    def __init__(self): self.calls=[]
    def predict_graph(self,graphs,task,return_crystal_feas,batch_size):
        self.calls.append([graphs[0],len(graphs),batch_size])
        assert task=='efsm' and return_crystal_feas
        if graphs[0]==16 and batch_size>8:
            raise torch.cuda.OutOfMemoryError('Simulated allocation failure')
        result=[{'id':g} for g in graphs]
        return result[0] if len(graphs)==1 else result


if __name__=='__main__':
    model=RetryModel()
    result=runner.predict(model,list(range(40)),16)
    assert [r['id'] for r in result]==list(range(40))
    assert model.calls==[[0,16,16],[16,16,16],[16,8,8],[24,8,8],[32,8,8]]
    path=json.loads((runner.reproduce.ROOT/'datasets/manifest.json').read_text())['matpes_pbe']['path']
    old=list(itertools.islice(runner.structures('matpes_pbe',path),57344,57346))
    new=list(itertools.islice(runner.structures('matpes_pbe',path,start=57344),2))
    assert old==new
    record={'retry_preserves_every_prediction_once':True,'already_successful_batches_not_repeated':True,
        'simulated_retry_calls':model.calls,'resumed_input_structure_ids_and_forces_exactly_equal':True}
    atomic_json(runner.reproduce.OUT/'optimization_control_validation.json',record)
    print(json.dumps(record))
