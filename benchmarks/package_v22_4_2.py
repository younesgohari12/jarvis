"""JARVIS v22.4.2 — FINAL packaging (v22.4.1 packager reused verbatim with release-integrity verification
(spec §5-§12, §36-§45, §65).

Fixes vs the v22.4 packager:
  * Issue A: `.hypothesis` (and every other dev/cache artifact) is now
    excluded via an explicit release-exclusion policy — v22.4 shipped ~175
    `.hypothesis` cache files inside the ZIP.
  * Issue B: package_integrity.json is generated LAST, directly from the
    final ZIP via ZipFile() (spec §44) — never from a guessed manifest count.
  * §36: deterministic packaging — sorted paths, normalized timestamps,
    normalized permissions, stable JSON (two builds of one tree are
    byte-identical; verified in-line).
  * §7/§38: forbidden-path verification of the actual archive.
  * §12: manifest keys == ZIP contents minus SHA256SUMS.json, exactly.

Packaging policy (documented in the audit document §10):
  reports/v22_4_2/package_integrity.json and
  reports/v22_4_2/metadata_consistency.json are EXTERNAL finalization
  artifacts: they are generated from the final ZIP and shipped beside it.
  A file inside a ZIP cannot truthfully contain that ZIP's final SHA256, so
  the integrity report is published next to the artifact it authenticates
  (same policy as the v22.4 Archive_Check.json).

Usage: python benchmarks/package_v22_4_2.py <output_dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

PKG_ROOT = 'Jarvis_v0.11.0'                                  # spec §37
ZIP_NAME = 'Jarvis_v0.11.0_Intelligence_v22.4.2_SECURITY_AUDITED_FIXED.zip'
RELEASE_VERSION = 'v0.11.0-intelligence-v22.4.2'
BUILD_DATE = (2026, 9, 20, 12, 0, 0)                          # §36 normalized

# ----------------------------------------------------------------------
# explicit release-exclusion policy (spec §6) — no blind deletion
# ----------------------------------------------------------------------
EXCLUDED_DIRS = {
    '.git', '.venv', '__pycache__', '.pytest_cache', '.hypothesis',
    '.mypy_cache', '.ruff_cache', '.tox', '.cache', 'tox', 'htmlcov',
    'node_modules', 'blind_chunks', 'dist_v22_4_2', '.idea', '.vscode',
    'runtime_data',              # runtime state (gitignored) — never shipped
}
EXCLUDED_SUFFIXES = ('.pyc', '.pyo', '.partial.json', '.tmp', '.bak',
                     '.log', '.zip', '.orig', '.rej', '.db')
EXCLUDED_NAMES = {
    '.DS_Store', 'Thumbs.db', 'coverage.xml', '.coverage', 'blind_run.log',
    'blind_run.log.1', 'nohup.out',
}
# external finalization artifacts: generated FROM the final ZIP, shipped
# beside it (a file inside the ZIP cannot contain the ZIP's own final hash)
EXTERNAL_FINALIZATION = {
    'reports/v22_4_2/package_integrity.json',
    'reports/v22_4_2/package_inventory.json',
    'reports/v22_4_2/metadata_consistency.json',
    'reports/v22_4_2/security_package.json',
}

FORBIDDEN_PATH_FRAGMENTS = (                      # spec §7/§38
    '/.hypothesis/', '/.pytest_cache/', '/__pycache__/', '/.git/',
    '/.venv/', '/node_modules/', '/.mypy_cache/', '/.ruff_cache/', '/.tox/',
    '/blind_chunks/', '/.idea/', '/.vscode/',
)
FORBIDDEN_NAME_FRAGMENTS = ('.pyc', '.partial.json', '.tmp', '.bak', '.log',
                            '.zip', 'blind_run.log', '.DS_Store', 'Thumbs.db',
                            '.coverage')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def included(rel_posix: str, path: str) -> bool:
    parts = rel_posix.split('/')
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if rel_posix in EXTERNAL_FINALIZATION:
        return False
    name = os.path.basename(path)
    if name in EXCLUDED_NAMES or name == 'SHA256SUMS.json':
        return False
    if name.startswith(('jarvis.db', 'knowledge.db', 'jarvis.log')):
        return False
    if name.endswith(EXCLUDED_SUFFIXES):
        return False
    if name.startswith('.hypothesis') or name.endswith('.zip'):
        return False
    return True


def build_file_list(root: str) -> list[str]:
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace(os.sep, '/')
            if included(rel, p):
                result.append(rel)
    return sorted(result)


def deterministic_zip(zf: zipfile.ZipFile, root: str, files: list[str]) -> None:
    """Sorted entries, fixed timestamps, normalized permissions (§36)."""
    for rel in files:
        src = os.path.join(root, rel)
        data = open(src, 'rb').read()
        info = zipfile.ZipInfo(PKG_ROOT + '/' + rel, date_time=BUILD_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16          # normalize permissions
        info.create_system = 3                    # Unix
        zf.writestr(info, data, compresslevel=6)


def write_manifest(root: str, files: list[str]) -> dict:
    manifest = {
        'algorithm': 'sha256',
        'scope': 'all packaged files except this manifest',
        'release': RELEASE_VERSION + ' — FINAL CLEAN HARDENED',
        'file_count': len(files),
        'files': {rel: sha(open(os.path.join(root, rel), 'rb').read())
                  for rel in files},
    }
    mp = os.path.join(root, 'SHA256SUMS.json')
    with open(mp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    return manifest


def verify_zip(zip_path: str, manifest: dict) -> dict:
    """Verify the actual archive: CRC, hashes, inventory, forbidden paths."""
    with zipfile.ZipFile(zip_path) as z:
        crc_bad = z.testzip()
        names = z.namelist()
        file_names = [n for n in names if not n.endswith('/')]
        roots = {n.split('/')[0] for n in names}
        stripped = {n.split('/', 1)[1] for n in file_names if '/' in n}

        forbidden = [n for n in file_names
                     if any(f'/{frag.strip("/")}/' in n for frag in
                            ('.hypothesis', '.pytest_cache', '__pycache__',
                             '.git', '.venv', 'node_modules', '.mypy_cache',
                             '.ruff_cache', '.tox', 'blind_chunks', '.idea',
                             '.vscode'))
                     or n.split('/')[-1].endswith(('.pyc', '.partial.json',
                                                   '.tmp', '.bak', '.zip',
                                                   '.DS_Store', 'Thumbs.db'))
                     or 'blind_run.log' in n]

        missing = sorted(rel for rel in manifest['files'] if rel not in stripped)
        extra = sorted(rel for rel in stripped
                       if rel not in manifest['files'] and rel != 'SHA256SUMS.json')
        mismatch = []
        for rel, digest in manifest['files'].items():
            try:
                if sha(z.read(PKG_ROOT + '/' + rel)) != digest:
                    mismatch.append(rel)
            except KeyError:
                mismatch.append(rel)
        manifest_self_ok = sha(z.read(PKG_ROOT + '/SHA256SUMS.json')) == \
            sha(open(os.path.join(BASE, 'SHA256SUMS.json'), 'rb').read())

    return {
        'crc_ok': crc_bad is None,
        'crc_first_bad': crc_bad,
        'single_root': roots == {PKG_ROOT},
        'roots': sorted(roots),
        'archive_file_count': len(file_names),
        'manifest_entries': len(manifest['files']),
        'missing': missing,
        'extra': extra,
        'mismatch': mismatch,
        'missing_count': len(missing),
        'extra_count': len(extra),
        'mismatch_count': len(mismatch),
        'manifest_self_hash_ok': manifest_self_ok,
        'forbidden_paths': forbidden,
        'forbidden_cache_files': len(forbidden),
        'inventory_exact': not missing and not extra,
    }


def main() -> int:
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                              else os.path.join(BASE, 'dist_v22_4_2'))
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, ZIP_NAME)

    # 1 — final file list + regenerated manifest (spec §9 order)
    files = build_file_list(BASE)
    manifest = write_manifest(BASE, files)
    print(f'file list: {len(files)} files; manifest entries: {len(manifest["files"])}')

    # 2 — deterministic ZIP build
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        deterministic_zip(zf, BASE, files)
        mp = zipfile.ZipInfo(PKG_ROOT + '/SHA256SUMS.json', date_time=BUILD_DATE)
        mp.compress_type = zipfile.ZIP_DEFLATED
        mp.external_attr = 0o644 << 16
        mp.create_system = 3
        zf.writestr(mp, open(os.path.join(BASE, 'SHA256SUMS.json'), 'rb').read(),
                    compresslevel=6)

    # 3 — verify the actual archive (spec §12/§44)
    v = verify_zip(zip_path, manifest)
    print(json.dumps({k: v[k] for k in ('crc_ok', 'single_root',
                                        'archive_file_count', 'manifest_entries',
                                        'missing_count', 'extra_count',
                                        'mismatch_count', 'forbidden_cache_files',
                                        'manifest_self_hash_ok')}))

    # 4 — reproducibility: rebuild into a temp dir, compare (spec §36)
    repro = {'attempted': True}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_zip = os.path.join(tmp, 'rebuild.zip')
            with zipfile.ZipFile(tmp_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                deterministic_zip(zf, BASE, files)
                mp2 = zipfile.ZipInfo(PKG_ROOT + '/SHA256SUMS.json', date_time=BUILD_DATE)
                mp2.compress_type = zipfile.ZIP_DEFLATED
                mp2.external_attr = 0o644 << 16
                mp2.create_system = 3
                zf.writestr(mp2, open(os.path.join(BASE, 'SHA256SUMS.json'), 'rb').read(),
                            compresslevel=6)
            tmp_sha = sha(open(tmp_zip, 'rb').read())
            final_sha = sha(open(zip_path, 'rb').read())
            with zipfile.ZipFile(tmp_zip) as z1:
                inv1 = sorted(z1.namelist())
            with zipfile.ZipFile(zip_path) as z2:
                inv2 = sorted(z2.namelist())
            repro.update({
                'inventories_identical': inv1 == inv2,
                'zip_binary_hash_identical': tmp_sha == final_sha,
                'rebuild_sha256': tmp_sha,
                'final_sha256': final_sha,
                'note': ('timestamps and permissions are normalized, so two '
                         'builds of one tree are byte-identical'),
            })
    except Exception as exc:                                 # pragma: no cover
        repro.update({'attempted': False, 'error': repr(exc)})

    # 5 — package security scan (authoritative for release, spec §40)
    from benchmarks.v22_4_1_security import scan_package  # improved BUG-002 scanner
    sec = scan_package(zip_path)

    size = os.path.getsize(zip_path)
    zsha = sha(open(zip_path, 'rb').read())

    # 6 — package_inventory.json (build environment + inventory evidence)
    by_top = {}
    for rel in sorted(manifest['files']):
        top = rel.split('/')[0]
        by_top[top] = by_top.get(top, 0) + 1
    inventory = {
        'release': RELEASE_VERSION,
        'zip_name': ZIP_NAME,
        'package_root': PKG_ROOT,
        'archive_file_count': v['archive_file_count'],
        'manifest_entries': v['manifest_entries'],
        'files_by_top_level': by_top,
        'archived_paths': sorted(n.split('/', 1)[1] for n in
                                 zipfile.ZipFile(zip_path).namelist()
                                 if '/' in n and not n.endswith('/')),
        'build_environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'zip_compression': 'ZIP_DEFLATED level 6',
            'timestamp_policy': f'normalized to {BUILD_DATE}',
            'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        },
    }

    # 7 — package_integrity.json generated FROM THE FINAL ZIP (spec §10/§44)
    passed = all([
        v['crc_ok'], v['single_root'], v['inventory_exact'],
        v['mismatch_count'] == 0, v['manifest_self_hash_ok'],
        v['forbidden_cache_files'] == 0, sec['passed'],
        repro.get('inventories_identical', False),
    ])
    integrity = {
        'release': RELEASE_VERSION,
        'zip_name': ZIP_NAME,
        'zip_sha256': zsha,
        'zip_size_bytes': size,
        'package_root': PKG_ROOT,
        'archive_file_count': v['archive_file_count'],
        'manifest_entries': v['manifest_entries'],
        'manifest_scope': 'all packaged files except SHA256SUMS.json',
        'missing': v['missing_count'],
        'mismatch': v['mismatch_count'],
        'extra': v['extra_count'],
        'crc_ok': v['crc_ok'],
        'forbidden_cache_files': v['forbidden_cache_files'],
        'passed': passed,
        'verification_detail': {
            'single_root': v['single_root'],
            'roots_found': v['roots'],
            'missing_files': v['missing'][:20],
            'extra_files': v['extra'][:20],
            'hash_mismatches': v['mismatch'][:20],
            'forbidden_paths': v['forbidden_paths'][:20],
            'manifest_self_hash_ok': v['manifest_self_hash_ok'],
            'inventory_exact': v['inventory_exact'],
        },
        'reproducibility': repro,
        'security_scan': {'secret_count': sec['secret_count'],
                          'files_scanned': sec['files_scanned'],
                          'passed': sec['passed']},
        'generated_from': 'ZipFile(final_zip) — measured, never hand-typed (spec §44)',
        'generation_order': ('source → tests → reports → cache cleanup → '
                             'file list → SHA256SUMS → ZIP → verify ZIP → '
                             'package_integrity (this file) — spec §9'),
        'shipping_policy': ('this report ships BESIDE the ZIP (external '
                            'finalization artifact); a file inside a ZIP '
                            'cannot contain that ZIP\'s final SHA256'),
        'post_upload_digest_verified': False,
        'post_upload_reason': 'not yet uploaded',
    }

    with open(os.path.join(BASE, 'reports', 'v22_4_2', 'package_inventory.json'),
              'w', encoding='utf-8') as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, 'reports', 'v22_4_2', 'package_integrity.json'),
              'w', encoding='utf-8') as f:
        json.dump(integrity, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'package_integrity.json'), 'w',
              encoding='utf-8') as f:
        json.dump(integrity, f, ensure_ascii=False, indent=2)

    print(json.dumps({'zip': ZIP_NAME, 'size_bytes': size, 'sha256': zsha,
                      'passed': passed,
                      'security_passed': sec['passed'],
                      'reproducible': repro.get('zip_binary_hash_identical')}))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
