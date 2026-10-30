# Generated isolated adapter; see adapter_manifest.json.
"""Verify CUDA retry control flow preserves every structure and raises failures."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import torch
import mace_batch_inference as helper
import new_potential_force as runner
from relaxation_runner import atomic_json

def main():
    original = helper.predict
    calls = []
    wrapped_message = 'The following operation failed in the TorchScript interpreter.\nRuntimeError: CUDA out of memory. Tried to allocate 1.72 GiB.'
    failure_kind = 'direct'

    def simulated(calculator, atoms, batch_size):
        calls.append([atoms[0], len(atoms), batch_size])
        if atoms[0] == 16 and len(atoms) > 8:
            if failure_kind == 'wrapped':
                raise RuntimeError(wrapped_message)
            raise torch.cuda.OutOfMemoryError('Simulated batch allocation failure')
        if atoms == [-1]:
            if failure_kind == 'wrapped':
                raise RuntimeError(wrapped_message)
            raise torch.cuda.OutOfMemoryError('Simulated singleton allocation failure')
        if atoms == [-2]:
            raise RuntimeError('Simulated unrelated model failure')
        return [{'id': atom} for atom in atoms]
    helper.predict = simulated
    try:
        for failure_kind in ['direct', 'wrapped']:
            calls.clear()
            (results, events) = runner.predict_cuda(list(range(40)), 16)
            assert [row['id'] for row in results] == list(range(40))
            assert calls == [[0, 16, 16], [16, 16, 16], [16, 8, 8], [24, 8, 8], [32, 8, 8]]
            assert events == [{'offset': 16, 'structures': 16, 'retry_batch_size': 8}]
            raised = False
            try:
                runner.predict_cuda([-1], 16)
            except RuntimeError as error:
                assert isinstance(error, torch.cuda.OutOfMemoryError) if failure_kind == 'direct' else str(error) == wrapped_message
                raised = True
            assert raised
        calls.clear()
        raised = False
        try:
            runner.predict_cuda([-2], 16)
        except RuntimeError as error:
            assert str(error) == 'Simulated unrelated model failure'
            raised = True
        assert raised and calls == [[-2, 1, 1]]
    finally:
        helper.predict = original
    report = {'validation_pass': True, 'every_structure_predicted_once_and_in_order': True, 'successful_batches_not_repeated': True, 'singleton_failure_propagates_without_skipping': True, 'torchscript_wrapped_cuda_allocation_failures_partitioned': True, 'unrelated_runtime_errors_propagate_without_retry': True, 'runner_sha256': runner.reproduce.sha(runner.reproduce.ROOT / 'outputs/round2/adapters_v2/mace_omat_0_medium/new_potential_force.py')}
    atomic_json(runner.reproduce.OUT / 'round2/mace_omat_0_medium/validation/mace_cuda_retry_control_validation.json', report)
    print(report, flush=True)
if __name__ == '__main__':
    main()
