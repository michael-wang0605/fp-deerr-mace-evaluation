"""Exercise splitting, saved-result retention, and singleton failure controls."""
from contextlib import contextmanager
import json
import torch
import cuda_saved_tensor_offload as helper
import force_benchmark_runner as runner
from relaxation_runner import atomic_json


def main():
    state = {'offload': False}
    @contextmanager
    def fake_context(*args, **kwargs):
        state['offload'] = True
        try:
            yield {'saved_tensors': 1, 'saved_tensor_bytes': 1024, 'unpack_calls': 1}
        finally:
            state['offload'] = False
    original = helper.saved_tensors_on_disk
    helper.saved_tensors_on_disk = fake_context
    class Model:
        def __init__(self, fail_offload=False):
            self.successful = []
            self.fail_offload = fail_offload
        def predict_graph(self, part, **kwargs):
            assert kwargs['task'] == 'efsm' and kwargs['return_crystal_feas']
            assert kwargs['batch_size'] == len(part)
            if len(part) > 2 or (5 in part and (len(part) > 1 or not state['offload'] or self.fail_offload)):
                raise torch.cuda.OutOfMemoryError('controlled test failure')
            self.successful += part
            results = [{'index': i} for i in part]
            return results[0] if len(part) == 1 else results
    try:
        model = Model()
        events = []
        result = runner.predict(model, list(range(40)), 16, singleton_disk_offload=True, retry_events=events)
        assert [r['index'] for r in result] == list(range(40))
        assert model.successful == list(range(40))
        assert len(events) == 1 and events[0]['graph_offset'] == 5
        raised = False
        try:
            runner.predict(Model(fail_offload=True), [5], 1, singleton_disk_offload=True)
        except torch.cuda.OutOfMemoryError:
            raised = True
        assert raised, 'An offloaded singleton failure must remain a visible failure'
        report = {'validation_pass': True, 'runner_sha256': runner.reproduce.sha(runner.reproduce.ROOT/'force_benchmark_runner.py'),
                  'structures': 40, 'order_preserved': True, 'successful_predictions_exactly_once': True,
                  'only_failed_singleton_offloaded': True, 'offloaded_singleton_failure_raises': True}
        atomic_json(runner.reproduce.OUT/'chgnet_disk_offload_retry_validation.json', report)
        print(json.dumps(report), flush=True)
    finally:
        helper.saved_tensors_on_disk = original


if __name__ == '__main__':
    main()
