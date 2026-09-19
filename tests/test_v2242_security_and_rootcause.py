"""JARVIS v22.4.2 — independent bug regression suite (spec §25).

Never relies on the frozen benchmark. Covers the confirmed v22.4.1 defects
plus the work-rate generalization fix:

  BUG-001  Windows route command permission classification bypass
  BUG-002  secret scanner prefixed-variable detection gap
  BUG-003  release metadata test-count inconsistency (Markdown gate)
  WEAK-W1  work-rate output-noun vocabulary generalization
  §16.4    source <-> semantic slot consistency invariants

Fake credential values below are assembled from string fragments so this
test source never matches the scanner's own credential grammar (and no real
secret exists anywhere in the project).
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from jarvis.tools.terminal import CommandPolicy, CommandRunner
from jarvis.tools.registry import ToolRegistry, ToolSpec, ToolError
from jarvis.agent.permissions import PermissionLayer
from benchmarks.v22_4_1_security import PATTERNS, scan_text
from benchmarks.v22_4_1_metadata_consistency import check_markdown_consistency
from jarvis.agent.numeric_roles_v22 import (
    classify_source_numbers_v2,
    role_slot_consistency,
)


# ======================================================================
# BUG-001 — route command permission classification (spec §5)
# ======================================================================
class TestRouteCommandPolicy:
    @pytest.mark.parametrize('command', [
        'route print', 'route -4 print', 'route -6 print', 'ROUTE PRINT',
    ])
    def test_route_print_stays_read_only(self, command):
        assessment = CommandPolicy.assess(command)
        assert assessment.level == 'safe'
        assert assessment.code == 'read_only'

    @pytest.mark.parametrize('command', [
        'route add 10.0.0.0 mask 255.0.0.0 1.2.3.4',
        'route -4 add 10.0.0.0 mask 255.0.0.0 1.2.3.4',
        'route -6 add 10.0.0.0 mask 255.0.0.0 1.2.3.4',
        'route delete 10.0.0.0',
        'route -4 delete 10.0.0.0',
        'route -6 delete 10.0.0.0',
        'route change 10.0.0.0 mask 255.0.0.0 1.2.3.4',
        'route -4 change 10.0.0.0 mask 255.0.0.0 1.2.3.4',
        'route -6 change 10.0.0.0 mask 255.0.0.0 1.2.3.4',
    ])
    def test_route_mutating_operations_require_confirmation(self, command):
        assessment = CommandPolicy.assess(command)
        assert assessment.level == 'dangerous', command
        assert assessment.code == 'network_change_confirmation_required'

    @pytest.mark.parametrize('command', [
        'route', 'route -4', 'route -6', 'route nonsense', 'route -4 nonsense',
        'route -f',                       # clearing flag is never read-only
        'route -4 add 10.0.0.0',          # mutating op is caught above, but the
    ])
    def test_route_missing_or_unknown_operation_fails_closed(self, command):
        """Fail closed: unknown/missing operations must never be safe."""
        assessment = CommandPolicy.assess(command)
        assert assessment.level in {'blocked', 'dangerous'}, command
        assert assessment.level != 'safe'

    def test_route_risk_category_maps_mutating_to_shell(self):
        assert CommandRunner.risk_category(
            {'command': 'route -4 delete 10.0.0.0'}) == 'shell'
        assert CommandRunner.risk_category({'command': 'route print'}) == 'read_only'

    def test_mutating_route_never_reaches_subprocess_without_confirmation(self):
        """Integration: registry + permission layer + dynamic risk resolver."""
        registry = ToolRegistry(PermissionLayer())
        handler = MagicMock(return_value='should-not-run')
        spec = ToolSpec(
            'run_command', 'Run one policy-classified command without a shell',
            'caution', handler, required=('command',),
            risk_resolver=CommandRunner.risk_category,
        )
        registry.register(spec)
        mutating = {'command': 'route -6 add 10.0.0.0 mask 255.0.0.0 1.2.3.4'}
        with patch('subprocess.Popen') as popen_mock:
            with pytest.raises(ToolError):
                registry.invoke('run_command', mutating, confirmed=False)
            popen_mock.assert_not_called()
        handler.assert_not_called()

    def test_mutating_route_runs_only_after_explicit_confirmation(self):
        registry = ToolRegistry(PermissionLayer())
        handler = MagicMock(return_value='ok')
        registry.register(ToolSpec(
            'run_command', 'Run one policy-classified command without a shell',
            'caution', handler, required=('command',),
            risk_resolver=CommandRunner.risk_category,
        ))
        mutating = {'command': 'route add 10.0.0.0 mask 255.0.0.0 1.2.3.4'}
        assert registry.invoke('run_command', mutating, confirmed=True) == 'ok'
        handler.assert_called_once()


# ======================================================================
# permission floor (spec §25 — dynamic risk can never be lowered)
# ======================================================================
class TestPermissionFloor:
    def test_dynamic_shell_result_passes_through(self):
        spec = ToolSpec('t', 'd', 'caution', lambda a: None,
                        risk='caution', risk_resolver=lambda a: 'shell')
        assert spec.effective_permission_category({'command': 'x'}) == 'shell'

    def test_dangerous_declared_floor_raises_read_only_dynamic(self):
        spec = ToolSpec('t', 'd', 'read_only', lambda a: None,
                        risk='dangerous', risk_resolver=lambda a: 'read_only')
        assert spec.effective_permission_category({}) == 'destructive'

    def test_shell_category_requires_confirmation(self):
        decision = PermissionLayer().check('shell', confirmed=False)
        assert decision.allowed is False and decision.requires_confirmation is True

    def test_unknown_category_denied(self):
        decision = PermissionLayer().check('mystery_category', confirmed=False)
        assert decision.allowed is False


# ======================================================================
# BUG-002 — secret scanner prefixed-variable coverage (spec §6)
# ======================================================================
_FAKE_SECRET_A = 'TESTSECRET' + '123456789012345678'
_FAKE_SECRET_B = 'TESTTOKEN' + '123456789012345678'
_FAKE_BOT_TOKEN = '123456789:' + 'TESTTOKEN' + 'abcdefghij' * 3

_POSITIVE_CREDENTIAL_CASES = [
    ('plain_api_key', f'API_KEY = "{_FAKE_SECRET_A}"'),
    ('prefixed_openai', f'OPENAI_API_KEY = "{_FAKE_SECRET_A}"'),
    ('prefixed_avalai', f'AVALAI_API_KEY = "{_FAKE_SECRET_A}"'),
    ('multi_prefix', f'CUSTOM_PROVIDER_API_KEY = "{_FAKE_SECRET_A}"'),
    ('no_spaces', f'SERVICE_API_KEY="{_FAKE_SECRET_A}"'),
    ('access_token', f'MY_SERVICE_ACCESS_TOKEN = "{_FAKE_SECRET_B}"'),
    ('secret_key', f'CUSTOM_SECRET_KEY = "{_FAKE_SECRET_A}"'),
    ('secret_token', f'SERVICE_SECRET_TOKEN = "{_FAKE_SECRET_B}"'),
    ('bot_token_name', f'BOT_TOKEN = "{_FAKE_BOT_TOKEN}"'),
    ('json_form', '{{"service_api_key": "{v}"}}'.format(v=_FAKE_SECRET_A)),
    ('env_form', f'SERVICE_API_KEY={_FAKE_SECRET_A}'),
]

_NEGATIVE_CREDENTIAL_CASES = [
    ('empty_value', 'API_KEY = ""'),
    ('none_value', 'API_KEY = None'),
    ('placeholder_your', 'API_KEY = "YOUR_API_KEY"'),
    ('placeholder_change_me', 'API_KEY = "CHANGE_ME"'),
    ('placeholder_angle', 'API_KEY = "<API_KEY>"'),
    ('placeholder_dollar', 'API_KEY = "${API_KEY}"'),
    ('env_lookup_getenv', 'API_KEY = os.getenv("API_KEY")'),
    ('env_lookup_environ', 'API_KEY = os.environ.get("API_KEY")'),
    ('env_placeholder', 'API_KEY=YOUR_API_KEY_HERE'),
    ('short_value', 'API_KEY = "short"'),
]


class TestSecretScanner:
    @pytest.mark.parametrize('label,content', _POSITIVE_CREDENTIAL_CASES,
                             ids=[c[0] for c in _POSITIVE_CREDENTIAL_CASES])
    def test_prefixed_credential_names_are_detected(self, label, content):
        findings = scan_text('config.py', content)
        assert findings, f'credential assignment not detected: {label}'
        assert any(f['secret_type'] != 'hardcoded_password' for f in findings)

    @pytest.mark.parametrize('label,content', _NEGATIVE_CREDENTIAL_CASES,
                             ids=[c[0] for c in _NEGATIVE_CREDENTIAL_CASES])
    def test_placeholders_and_lookups_are_not_reported(self, label, content):
        assert scan_text('config.py', content) == [], label

    def test_findings_contain_only_safe_metadata(self):
        findings = scan_text('config.py', f'OPENAI_API_KEY = "{_FAKE_SECRET_A}"')
        assert findings == [{'file': 'config.py', 'secret_type': 'api_key_generic'}]
        dumped = json.dumps(findings)
        assert _FAKE_SECRET_A not in dumped
        assert 'TESTSECRET' not in dumped

    def test_bot_token_shape_still_detected(self):
        findings = scan_text('bot.py', f'TOKEN = "{_FAKE_BOT_TOKEN}"')
        assert any(f['secret_type'] == 'bot_token' for f in findings)

    def test_github_and_aws_patterns_remain(self):
        types = {t for t, _ in PATTERNS}
        assert {'github_pat', 'github_token_classic', 'github_oauth',
                'github_token_finegrained', 'aws_access_key',
                'private_key_block', 'slack_token', 'google_api_key',
                'hardcoded_password'} <= types

    def test_scanner_never_records_values(self):
        for stype, pattern in PATTERNS:
            assert stype  # every pattern is (type, compiled) with no value sink
        findings = scan_text('x.env', f'SERVICE_API_KEY={_FAKE_SECRET_A}')
        for finding in findings:
            assert set(finding) == {'file', 'secret_type'}


# ======================================================================
# BUG-003 — release metadata consistency gate (spec §7)
# ======================================================================
class TestMarkdownMetadataGate:
    def _write_doc(self, base, name, text):
        path = os.path.join(base, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    def test_real_release_docs_agree_with_structured_artifact(self):
        with open(os.path.join(BASE, 'reports', 'v22_4_1', 'regression.json'),
                  encoding='utf-8') as f:
            regression = json.load(f)
        report = check_markdown_consistency(regression, base=BASE)
        assert report['passed'] is True, report['problems']

    def test_stale_markdown_claim_is_rejected(self, tmp_path):
        self._write_doc(tmp_path, 'JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md',
                        'Regression: **1227 tests + 585 subtests, 0 failures**.')
        self._write_doc(tmp_path, 'V23_TRAINING_HANDOFF.md',
                        'Regression baseline to protect: 1228 tests + 585 subtests.')
        report = check_markdown_consistency(
            {'tests': 1228, 'subtests': 585}, base=str(tmp_path))
        assert report['passed'] is False
        assert any('1227' in p for p in report['problems'])

    def test_historical_counts_remain_allowed(self, tmp_path):
        self._write_doc(tmp_path, 'JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md',
                        'v22.4 baseline was 1215 + 585; current is '
                        '1228 tests + 585 subtests.')
        self._write_doc(tmp_path, 'V23_TRAINING_HANDOFF.md', 'clean')
        report = check_markdown_consistency(
            {'tests': 1228, 'subtests': 585}, base=str(tmp_path))
        assert report['passed'] is True, report['problems']

    def test_table_current_column_is_checked(self, tmp_path):
        self._write_doc(tmp_path, 'JARVIS_V22_4_1_RELEASE_INTEGRITY_AUDIT.md',
                        '| tests | 1215 | **1227** |\n'
                        'current: 1228 tests + 585 subtests')
        self._write_doc(tmp_path, 'V23_TRAINING_HANDOFF.md', 'clean')
        report = check_markdown_consistency(
            {'tests': 1228, 'subtests': 585}, base=str(tmp_path))
        assert report['passed'] is False
        assert any('table' in p for p in report['problems'])


# ======================================================================
# §16.4 / §10.3 — source <-> semantic slot consistency
# ======================================================================
CANONICAL_EN = ('3 workers produce 84 units in 4 hours. '
                'How many units do 8 workers produce in 6 hours?')
CANONICAL_FA = ('ابتدا ۳ کارگر در ۴ ساعت ۸۴ واحد تولید می‌کنند؛ '
                'سپس ۸ کارگر در ۶ ساعت چند واحد تولید می‌کنند؟')


class TestSourceSlotConsistency:
    def test_canonical_roles_bind_initial_before_target(self):
        rows = classify_source_numbers_v2(CANONICAL_EN)
        roles = {f['role']: float(f['value']) for f in rows}
        assert roles['workers_initial'] == 3.0
        assert roles['workers_target'] == 8.0
        assert roles['output_initial'] == 84.0
        assert roles['hours_initial'] == 4.0
        assert roles['hours_target'] == 6.0

    def test_source_never_binds_initial_quantity_to_target_slot(self):
        """A quantity from '3 workers' must not land in workers_target."""
        from jarvis.agent.source_semantics_v20_1 import source_work_facts
        slots = source_work_facts(CANONICAL_EN)
        assert slots['workers_initial'] == 3.0
        assert slots['workers_target'] == 8.0
        rows = classify_source_numbers_v2(CANONICAL_EN)
        start = {f['role']: f['start'] for f in rows}
        assert start['workers_initial'] < start['workers_target']

    def test_role_slot_consistency_detects_swapped_slots(self):
        swapped_source = [
            {'value': 8, 'role': 'workers_initial', 'start': 0, 'end': 1},
            {'value': 3, 'role': 'workers_target', 'start': 10, 'end': 11},
        ]
        ir = {'task': 'work_rate',
              'slots': {'workers_initial': 3, 'workers_target': 8},
              'source_text': CANONICAL_EN}
        result = role_slot_consistency(ir, swapped_source)
        assert 'numeric_role_consistency' in result['failed_checks']
        assert result['details']['workers_order']['slots'] == [3.0, 8.0]
        assert result['details']['workers_order']['source'] == [8.0, 3.0]

    def test_consistent_slots_pass_role_check(self):
        source = [
            {'value': 3, 'role': 'workers_initial', 'start': 0, 'end': 1},
            {'value': 8, 'role': 'workers_target', 'start': 10, 'end': 11},
        ]
        ir = {'task': 'work_rate',
              'slots': {'workers_initial': 3, 'workers_target': 8},
              'source_text': CANONICAL_EN}
        result = role_slot_consistency(ir, source)
        assert 'numeric_role_consistency' not in result['failed_checks']


class TestWorkRateAnswers:
    """End-to-end work-rate answers with verified slot binding (WEAK-W1)."""

    @pytest.fixture(scope='class')
    def engine(self):
        from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
        return LocalIntelligenceV22()

    def test_canonical_english_answer(self, engine):
        answer = engine.solve(CANONICAL_EN, language='en')
        assert answer is not None and '336' in answer.text
        assert 'source_slot:workers_initial' in answer.checks
        assert 'source_slot:workers_target' in answer.checks

    def test_canonical_persian_answer(self, engine):
        answer = engine.solve(CANONICAL_FA, language='fa')
        assert answer is not None and '336' in answer.text

    def test_output_noun_generalization_widgets(self, engine):
        answer = engine.solve(
            'If 5 workers build 90 widgets in 3 hours, how many widgets '
            'do 10 workers build in 2 hours?', language='en')
        assert answer is not None and '120' in answer.text

    def test_slot_binding_is_positional_not_first_two_numbers(self, engine):
        """84/(4x4) x 8 x 6 = 252 — proves 84 stays bound to the FIRST crew."""
        answer = engine.solve(
            'If 4 workers can produce 84 units in 4 hours, how many units '
            'can 8 workers produce in 6 hours?', language='en')
        assert answer is not None and '252' in answer.text

    def test_worker_noun_generalization_machines_and_robots(self, engine):
        """WEAK-W1b/W1c/W1d: machines/robots are worker nouns across all
        regex layers (extractor, role audit, source-facts gate)."""
        answer = engine.solve(
            'If 6 robots weld 96 frames in 4 hours, how many frames do '
            '9 robots weld in 5 hours?', language='en')
        assert answer is not None and '180' in answer.text
        answer = engine.solve(
            'If 4 machines produce 220 parts in 5 hours, how many parts do '
            '7 machines produce in 3 hours?', language='en')
        assert answer is not None and '231' in answer.text
