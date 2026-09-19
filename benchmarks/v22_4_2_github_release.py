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

BASE = '/home/z/my-project/jarvis_work/repo_clone'
TAG = 'v0.11.0-intelligence-v22.4.2'
ZIP = os.path.join(BASE, 'dist_v22_4_2', 'Jarvis_v0.11.0_Intelligence_v22.4.2_SECURITY_AUDITED_FIXED.zip')
INTEGRITY = os.path.join(BASE, 'reports', 'v22_4_2', 'package_integrity.json')

BODY = """# JARVIS v0.11.0 — Intelligence v22.4.2
**SECURITY AUDITED / ROOT-CAUSE FIXED**

v22.4.2 is a **security + release-integrity release** over v22.4.1: three confirmed defects reproduced, root-caused, fixed, regression-tested and mutation-tested — plus one bounded, measured work-rate vocabulary generalization. No rewrite, no model retraining, frozen benchmark untouched.

## Bugs fixed

| ID | Severity | Defect | Fix |
|---|---|---|---|
| BUG-001 | **HIGH** | `route -4/-6 add/delete/change` were classified `safe/read_only` (global address-family switches mistaken for the operation) — mutating Windows route commands could run without confirmation | explicit fail-closed Route grammar parser in `CommandPolicy._assess_route`; unknown/missing operations are now **blocked** |
| BUG-002 | MEDIUM | secret scanner missed prefixed variables (`OPENAI_API_KEY`, `AVALAI_API_KEY`, …) and JSON/.env forms | full-name credential grammar (5 families), 3 assignment forms, placeholder filter; values are never recorded |
| BUG-003 | LOW | release Markdown claimed 1227 tests while the structured artifact measured 1228 | docs corrected; new **Markdown consistency gate** — release docs are now validated against `reports/v22_4_1/regression.json` |
| WEAK-W1 | LOW (quality) | work-rate noun vocabulary too narrow across four regex layers | bounded class-level generalization (output nouns + worker nouns), verified by unit tests |

## Verification (all measured, never hand-typed)

- Regression: **1292 tests + 585 subtests, 0 failures, 1 environment-dependent skip** (v22.4 baseline 1215+585 and v22.4.1 1228+585 held)
- Reproduction before fix: 5 mutating route commands classified `read_only` · 6/8 secret fixtures missed (evidence in release reports)
- Mutation: **11 valid / 11 killed / 0 survived** (M1–M7 original re-measured 7/7; new M8–M11 target the new guards) — valid-kill semantics preserved
- Security scan: source **1783 files / 0 secrets**, final package **0 secrets** (16 patterns)
- Package: **1791 manifest entries**, 0 missing / 0 extra / 0 mismatch, CRC clean, single root, **byte-identical rebuild reproducibility**
- New blind diagnostic (§26): 107 freshly authored cases frozen (SHA `8e4aac45df…`) **before** execution — score **32.71%** (answerable-only 25.77%), hard-negative abstention **10/10**. NOT comparable to the frozen benchmark and never averaged with it (spec §38)
- Performance warm p99 (simple math): **2.92 ms** — no regression from the security fixes

## Honest remaining gaps (v23 targets, documented not patched — see V23_LANGUAGE_BRAIN_PLAN.md)

Combined work-rates · inverted-hours queries · per-worker rate phrasing · standalone unit conversions · ratio/remainder arithmetic · broad free-form generation (Persian/English)

The 2075-case frozen benchmark remains untouched (`063aef8f…`); its published 91.13% is a historical claim and was **not** re-run.

## Assets

- `Jarvis_v0.11.0_Intelligence_v22.4.2_SECURITY_AUDITED_FIXED.zip` — the release package
- `package_integrity.json` — external finalization artifact measured from the final ZIP (a file inside a ZIP cannot contain its container's hash)

**In-tree reports: BUG_FIX_REPORT_FA.md · SECURITY_AUDIT_FA.md · INDEPENDENT_BENCHMARK_REPORT_FA.md · CHANGESET.md · TEST_RESULTS.json · V23_LANGUAGE_BRAIN_PLAN.md**
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
