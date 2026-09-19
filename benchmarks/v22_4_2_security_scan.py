"""v22.4.2 security scan artifacts (reuses the improved scanner).

Writes reports/v22_4_2/security_source.json (source tree) and, when a ZIP
path is given as argv[1], reports/v22_4_2/security_package.json.
"""
from __future__ import annotations

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from benchmarks.v22_4_1_security import scan_package, scan_source_tree  # noqa: E402

OUT_DIR = os.path.join(BASE, 'reports', 'v22_4_2')
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> int:
    source = scan_source_tree()
    with open(os.path.join(OUT_DIR, 'security_source.json'), 'w',
              encoding='utf-8') as f:
        json.dump(source, f, ensure_ascii=False, indent=2)
    result = {'source_tree': {'files_scanned': source['files_scanned'],
                              'secret_count': source['secret_count'],
                              'passed': source['passed'],
                              'patterns_checked': source['patterns_checked']}}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        package = scan_package(sys.argv[1])
        with open(os.path.join(OUT_DIR, 'security_package.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(package, f, ensure_ascii=False, indent=2)
        result['final_package'] = {'files_scanned': package['files_scanned'],
                                   'secret_count': package['secret_count'],
                                   'passed': package['passed']}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if source['passed'] and result.get('final_package', {}).get('passed', True) else 1


if __name__ == '__main__':
    sys.exit(main())
