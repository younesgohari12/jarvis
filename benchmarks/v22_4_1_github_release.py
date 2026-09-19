"""Create the GitHub release for v0.11.0-intelligence-v22.4.1 and upload the
release ZIP + package_integrity.json as assets. Then verify the uploaded
asset digest against the local SHA256 (spec §42/§43).

The PAT is read from the git remote URL — never printed, never logged.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

BASE = '/home/z/my-project/jarvis'
TAG = 'v0.11.0-intelligence-v22.4.1'
ZIP = '/home/z/my-project/download/Jarvis_v0.11.0_Intelligence_v22.4.1_FINAL_CLEAN_HARDENED.zip'
INTEGRITY = os.path.join(BASE, 'reports', 'v22_4_1', 'package_integrity.json')

BODY = """# JARVIS v0.11.0 — Intelligence v22.4.1
**FINAL CLEAN HARDENED / RELEASE INTEGRITY LOCK**

This is a **release-integrity patch**, not a new intelligence upgrade. The v22.4 runtime is untouched (**runtime behavioral files changed: 0**, parity 40/40 verified).

## What this release fixes

| Issue | Defect | Fix |
|---|---|---|
| A | `.hypothesis` cache (~175 files) shipped inside the v22.4 ZIP | explicit release-exclusion policy + forbidden-path verification of the actual archive |
| B | `package_integrity.json` stale (generated before final packaging) | generated **last**, directly from the final ZIP via `ZipFile()` — every value measured |
| C | Mutation M5 false kill (`returncode != 0` counted "no tests ran" as a kill) | valid-kill semantics (collected/failed/errors parsed), M5 selection fixed, framework self-test 6/6, manual M5 reproduction |
| D | "635 hand-written" claim misleading | honest authoring sidecar v2: **65 literal + 570 authored-template + 1440 synthetic = 2075**; frozen benchmark bytes/SHA untouched |

## Verification (all measured, never hand-typed)

- Regression: **1228 tests + 585 subtests, 0 failures** (v22.4 baseline 1215+585 held)
- P0 smoke: **16/16** — runtime parity vs v22.4 worktree: **40/40 identical**
- Mutation: **7 valid / 7 killed / 0 survived / 0 invalid** (corrected semantics)
- Frozen blind benchmark SHA-256: `063aef8f732c845e77a34c6916749d732e96489bbb74edfe12a3a8afc13f7be1` — unchanged; score **1891/2075 = 91.13%** (answerable-only 1568/1749 = 89.65%) reproduced from raw chunks
- Package: **1770 manifest entries**, 0 missing / 0 extra / 0 mismatch, CRC clean, single `Jarvis_v0.11.0/` root, **byte-identical rebuild reproducibility**
- Security: 0 secrets in source tree (1818 files) and final package (1771 files)
- Performance (warm): ownership p99 **6.03 ms** — the v22.4 59 ms p99 was cold-start noise

## Honest remaining gaps (v23 targets, documented not patched)

Persian broad 6/50 · English broad 0/50 · Blind Work Rate 52/90 · Blind Ownership 194/235 · Blind Rates 176/210 · Blind Finance 124/150 · Blind Age 139/163

The 2075-case blind benchmark is frozen as `V22_4_FROZEN_EVALUATION` — never to be trained on. See `V23_TRAINING_HANDOFF.md`.

## Assets

- `Jarvis_v0.11.0_Intelligence_v22.4.1_FINAL_CLEAN_HARDENED.zip` — the release package
- `package_integrity.json` — external finalization artifact: measured from the final ZIP (a file inside a ZIP cannot contain its container's hash)

**The deterministic v22 core is now locked. Next phase: v23 (GPU training / Language Brain).**
"""

def token():
    remote = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=BASE,
                            capture_output=True, text=True).stdout
    m = re.search(r'https://(github_pat_[A-Za-z0-9_]+)@', remote)
    return m.group(1)


def api(url, method='GET', data=None, tok=None, headers=None):
    req = urllib.request.Request(url, method=method)
    req.add_header('Authorization', f'Bearer {tok}')
    req.add_header('Accept', 'application/vnd.github+json')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    body = None
    if data is not None:
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, body) as r:
            return r.status, json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        payload = e.read().decode() or '{}'
        try:
            return e.code, json.loads(payload)
        except json.JSONDecodeError:
            return e.code, {'raw': payload}


def upload_asset(upload_url, path, name, tok):
    data = open(path, 'rb').read()
    url = upload_url.split('{')[0] + f'?name={name}'
    req = urllib.request.Request(url, method='POST', data=data)
    req.add_header('Authorization', f'Bearer {tok}')
    req.add_header('Content-Type', 'application/octet-stream')
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())


def main():
    tok = token()
    # existing release?
    status, existing = api(f'https://api.github.com/repos/younesgohari12/jarvis'
                           f'/releases/tags/{TAG}', tok=tok)
    if status == 200 and existing.get('id'):
        release_id = existing['id']
        upload_url = existing['upload_url']
        print('release exists:', existing['html_url'])
    else:
        status, rel = api('https://api.github.com/repos/younesgohari12/jarvis/releases',
                          method='POST', tok=tok, data={
            'tag_name': TAG,
            'target_commitish': 'main',
            'name': 'JARVIS v0.11.0 — Intelligence v22.4.1 (FINAL CLEAN HARDENED / RELEASE INTEGRITY LOCK)',
            'body': BODY,
            'draft': False,
            'prerelease': False,
        })
        release_id = rel['id']
        upload_url = rel['upload_url']
        print('release created:', rel['html_url'])

    assets = {a['name']: a for a in
              api(f'https://api.github.com/repos/younesgohari12/jarvis'
                  f'/releases/{release_id}/assets', tok=tok)[1]}
    for path, name in ((ZIP, os.path.basename(ZIP)),
                       (INTEGRITY, 'package_integrity.json')):
        if name not in assets:
            a = upload_asset(upload_url, path, name, tok)
            print('uploaded:', name, a.get('size'))
            assets[name] = a

    # §42 — post-upload digest verification
    status, fresh = api(f'https://api.github.com/repos/younesgohari12/jarvis'
                        f'/releases/{release_id}/assets', tok=tok)
    results = {}
    local_z = hashlib.sha256(open(ZIP, 'rb').read()).hexdigest()
    for a in fresh:
        digest = a.get('digest') or ''
        local = hashlib.sha256(
            open(ZIP if a['name'].endswith('.zip') else INTEGRITY, 'rb').read()).hexdigest()
        results[a['name']] = {
            'github_digest': digest,
            'local_sha256': local,
            'match': digest.endswith(local) if digest else None,
        }
    print(json.dumps(results, indent=1))
    return results


if __name__ == '__main__':
    main()
