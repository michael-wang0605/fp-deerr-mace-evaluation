"""Generate isolated model adapters while retaining original scientific functions.

Only model identity, private output/proof/source paths, explicit calculator head,
and in-memory leaderboard display metadata are adapted. Numeric constants,
optimizers, error handling and calls into upstream metric functions are retained.
The original scripts and completed outputs are never edited.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ['new_potential_static.py', 'new_potential_relaxation.py',
           'new_potential_force.py', 'new_potential_neb.py', 'validate_mace_batches.py',
           'profile_mace_device.py', 'validate_mace_cuda_partitions.py',
           'verify_mace_retry_controls.py', 'mace_neb_provenance.py',
           'new_potential_neb_mixed.py', 'mace_neb_cuda_worker.py',
           'validate_mace_cuda_neb.py', 'mace_batch_inference.py']
PROOFS = ['mace_batch_validation', 'mace_device_profile', 'mace_cpu_cuda_validation',
          'mace_cuda_partition_validation', 'mace_cuda_retry_control_validation',
          'mace_cuda_neb_validation']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Adapter(ast.NodeTransformer):
    def __init__(self, key, row):
        self.key, self.row = key, row
        self.prefix = 'outputs/round2/adapters_v2/' + key + '/'
        self.replacements = {
            'MACE-OMAT-0 Small': row['display_name'], 'MACE_OMAT_0_SMALL': row['fp_key'],
            'mace_omat_0_small': key, 'mace-omat-0-small.model': row['checkpoint'],
            '0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5': row['sha256'],
            'https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model': row['url'],
            'official small MACE-OMAT-0 checkpoint': 'official ' + row['display_name'] + ' checkpoint'}

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        value = node.value
        if value == 'omat_pbe':
            value = self.row['head']
        if value in SOURCES:
            value = self.prefix + value
        else:
            # Replace full URLs before checkpoint filename fragments.
            for old in sorted(self.replacements, key=len, reverse=True):
                value = value.replace(old, self.replacements[old])
            for proof in PROOFS:
                value = value.replace(proof, 'round2/' + self.key + '/validation/' + proof)
        return ast.copy_location(ast.Constant(value=value), node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'MACECalculator':
            assert not any(keyword.arg == 'head' for keyword in node.keywords)
            node.keywords.append(ast.keyword(arg='head', value=ast.Constant(self.row['head'])))
        return node

    def visit_ImportFrom(self, node):
        if node.module == 'export_neb_leaderboard':
            extra = ast.parse(
                'import export_neb_leaderboard as display_registry\n'
                f'display_registry.FP_DISPLAY_NAMES[KEY] = {self.row["display_name"]!r}\n').body
            return [node, *extra]
        return node


def main():
    catalog = json.loads((ROOT / 'outputs/round2/models.json').read_text())
    manifest = {'source_files_sha256': {name: sha(ROOT / name) for name in SOURCES},
                'generator_sha256': sha(Path(__file__)), 'models': {}}
    for key, row in catalog.items():
        folder = ROOT / 'outputs/round2/adapters_v2' / key
        folder.mkdir(parents=True, exist_ok=True)
        adapter = Adapter(key, row)
        paths = {}
        for name in SOURCES:
            original = ast.parse((ROOT / name).read_text(encoding='utf-8'))
            tree = ast.fix_missing_locations(adapter.visit(original))
            # Prepend only path setup, after the module docstring. Using append
            # keeps this model's own sibling adapters first on the import path.
            setup = ast.parse(f'import sys\nsys.path.append({str(ROOT)!r})\n').body
            tree.body[1:1] = setup
            text = '# Generated isolated adapter; see adapter_manifest.json.\n' + ast.unparse(tree) + '\n'
            if name in ['new_potential_neb.py', 'new_potential_neb_mixed.py']:
                text = text.replace('omat_pbe head', row['head'] + ' head')
            target = folder / name
            if target.exists():
                assert target.read_text(encoding='utf-8') == text, 'Refuse to mutate an existing adapter: ' + str(target)
            else:
                target.write_text(text, encoding='utf-8')
            compile(text, str(target), 'exec')
            paths[name] = {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)}
        manifest['models'][key] = {'identity': row, 'adapters': paths}
    target = ROOT / 'outputs/round2/adapter_manifest_v2.json'
    if target.exists():
        assert json.loads(target.read_text()) == manifest
    else:
        target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'models': list(catalog), 'adapters_per_model': len(SOURCES),
                      'upstream_metrics_modified': False, 'original_scripts_modified': False}), flush=True)


if __name__ == '__main__':
    main()
