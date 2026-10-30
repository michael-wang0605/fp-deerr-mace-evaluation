"""Validate an already-produced scientific stage without recalculating it."""
import argparse
import run_round2_suite as suite


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    job = {'model': args.model, 'stage': 'relaxation', 'expected': 'outputs/' + args.model + '/relaxation/summary.json'}
    suite.validate_completed(job, suite.read(suite.OUT / 'models.json'))
