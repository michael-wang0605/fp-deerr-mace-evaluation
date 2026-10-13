"""Analyze complete CPU or mixed MACE outputs with a registered display label."""
import argparse
import json
from pathlib import Path
import new_potential_neb as base
from mace_neb_provenance import register_display_name


def main(config):
    run=json.loads((base.folder('full')/'run_summary.json').read_text())
    assert run['completed']==154 and run['planned']==154
    if run.get('execution_provenances'):
        from new_potential_neb_mixed import analyze_existing
        analyze_existing(config)
    else:
        assert run['provenance']==base.provenance('full')
        register_display_name()
        base.analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=base.reproduce.OUT/'mace_neb_mixed_config.json')
    main(parser.parse_args().config)
