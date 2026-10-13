"""Verify and prepare the exact official force benchmark inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
from ase_db_backends.aselmdb import LMDBDatabase
import relaxation_runner as persistence

ROOT = Path(__file__).resolve().parent
D = ROOT / 'datasets'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


if __name__ == '__main__':
    mat = D / 'MatPES-PBE-2025.1.jsonl'
    h = digest(mat)
    assert h == '0d3fc076cfe9782e4fbba77f41315b24be034ee837fa8708e41dba652fa062a2'
    with mat.open('rb') as f:
        count = sum(1 for line in f if line.strip())
    archive = D / 'OMat24-val-rattled-1000.tar.gz'
    lmdb = D / 'rattled-1000/data.aselmdb'
    lmdb.parent.mkdir(exist_ok=True)
    if not lmdb.exists() or lmdb.stat().st_size != 349057024:
        with tarfile.open(archive, 'r:gz') as tar:
            member = tar.getmember('rattled-1000/data.aselmdb')
            assert member.isfile() and member.size == 349057024
            with tar.extractfile(member) as src, lmdb.with_suffix('.tmp').open('wb') as dst:
                shutil.copyfileobj(src, dst)
        lmdb.with_suffix('.tmp').replace(lmdb)
    db = LMDBDatabase(str(lmdb), readonly=True)
    n = db.count()
    assert n == 117004
    manifest = {
        'matpes_pbe': {'path': str(mat), 'sha256': h, 'structures': count,
            'source_url': 'https://huggingface.co/datasets/materialyze/matpes/resolve/main/MatPES-PBE-2025.1.jsonl',
            'version': '2025.1', 'anchor_force_indices': [29632, 29634, 29635, 29639, 29640]},
        'omat24': {'path': str(lmdb), 'sha256': digest(lmdb), 'structures': n,
            'archive_sha256': digest(archive), 'version': '241220 validation rattled-1000',
            'source_url': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz'}}
    persistence.atomic_json(D / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
