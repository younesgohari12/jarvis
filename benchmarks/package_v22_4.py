"""Package the v22.4 release: SHA256SUMS.json regenerated for the FINAL tree,
ZIP with a single Jarvis_v0.11.0/ root, CRC + every hash verified (spec §72/§73).

Usage: python benchmarks/package_v22_4.py <output_dir>
"""
from pathlib import Path
import hashlib, json, sys, zipfile

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = 'Jarvis_v0.11.0'   # spec §73: exactly one root directory
OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
ZIP = OUT / 'Jarvis_v0.11.0_Intelligence_v22_4_TRUE_ROOT_CAUSE_HARDENED.zip'
exclude = {'__pycache__', '.pytest_cache', '.git', '.venv', 'node_modules',
           'blind_chunks'}
def included(p):
    return (p.is_file()
            and not any(x in exclude for x in p.relative_to(ROOT).parts)
            and not p.name.endswith(('.pyc', '.partial.json'))
            and 'blind_run.log' not in p.name
            and not p.relative_to(ROOT).as_posix().endswith('.zip'))
def sha(data): return hashlib.sha256(data).hexdigest()

files = sorted(p for p in ROOT.rglob('*') if included(p) and p.name != 'SHA256SUMS.json')
manifest = {
    'algorithm': 'sha256',
    'scope': 'all packaged files except this manifest',
    'release': 'v0.11.0-intelligence-v22.4 — TRUE ROOT-CAUSE HARDENED',
    'file_count': len(files),
    'files': {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files},
}
mp = ROOT / 'SHA256SUMS.json'
mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, PKG_ROOT + '/' + p.relative_to(ROOT).as_posix())
    z.write(mp, PKG_ROOT + '/SHA256SUMS.json')

with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None, 'CRC failure'
    for rel, digest in manifest['files'].items():
        assert sha(z.read(PKG_ROOT + '/' + rel)) == digest, rel
    assert sha(z.read(PKG_ROOT + '/SHA256SUMS.json')) == sha(mp.read_bytes())
    names = z.namelist()
    roots = {n.split('/')[0] for n in names}
    assert roots == {PKG_ROOT}, roots   # exactly one root directory (spec §73)
    missing = mismatch = extra = 0
    listed = set(manifest['files'])
    stripped = {n.split('/', 1)[1] for n in names if '/' in n}
    for rel in listed:
        if rel not in stripped: missing += 1
    for rel in stripped:
        if rel not in listed and rel != 'SHA256SUMS.json': extra += 1
result = {
    'archive': ZIP.name,
    'size_bytes': ZIP.stat().st_size,
    'sha256': sha(ZIP.read_bytes()),
    'file_count': len(names),
    'crc_verified': True,
    'all_manifest_hashes_verified': True,
    'package_root': PKG_ROOT,
    'integrity': {'missing': missing, 'mismatch': 0, 'extra': extra},
    'release_gate': json.loads((ROOT / 'RELEASE_V22_4.json').read_text())['acceptance_gate'],
}
(OUT / 'Jarvis_v22_4_Archive_Check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
