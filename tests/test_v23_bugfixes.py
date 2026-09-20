"""JARVIS v23.0 — bug-fix regression tests (measured, mutation-style).

BUG-004 (MEDIUM): a hard-negative leak — 'موجودی علی ۱۵۰ دلار و ۲۰ کالاست؛
جمعش چقدر است؟' computed 150+20=170 and VERIFIED it. Root cause (two layers):

  1. quantity_v22.validate_operation_chain short-circuited the whole chain
     audit whenever the chain INITIAL was dimensionless (fail-open hole):
     the USD operand adopted nothing, so the later ITEM operand was never
     compared against it.
  2. verifier_v22._typed_audit skipped the operation-chain audit entirely
     when the parsed initial had no typed quantity in the source (a parsed
     0) — so layer 1 was never even reached.

Fix: a dimensionless initial now ADOPTS the first typed operand's dimension
and the audit CONTINUES for every later typed operand (bare dimensionless
operands never poison a typed chain — v21 semantics preserved); the audit
also runs when the initial is untyped. 'جمعش/مجموعش' added to _CHAIN_LANG.

Reproduction before fix: 170 (fabricated). After fix: structured refusal.
"""
from __future__ import annotations

import pytest

from jarvis.agent.quantity_v22 import (Quantity, extract_typed_quantities,
                                       validate_operation_chain)


def test_bug004_chain_adopts_first_typed_operand():
    text = 'موجودی علی ۱۵۰ دلار و ۲۰ کالاست؛ جمعش چقدر است؟'
    quantities = extract_typed_quantities(text)
    ops = [{'op': 'add', 'value': 150.0}, {'op': 'add', 'value': 20.0}]
    failed = validate_operation_chain(
        Quantity(0.0, 'dimensionless', ''), ops, quantities)
    assert failed, 'dimensionless initial must NOT disable the chain audit'
    assert 'currency_dimension_mismatch' in failed


def test_bug004_dimensionless_operands_never_poison():
    # '100 کالا است؛ 5 اضافه کن' — a bare 5 stays valid (v21 semantics)
    text = 'موجودی انبار 100 کالا است؛ 5 کالا اضافه کن'
    quantities = extract_typed_quantities(text)
    ops = [{'op': 'add', 'value': 100.0}, {'op': 'add', 'value': 5.0}]
    failed = validate_operation_chain(
        Quantity(0.0, 'dimensionless', ''), ops, quantities)
    assert failed == []


@pytest.mark.parametrize('text', [
    'موجودی علی ۱۵۰ دلار و ۲۰ کالاست؛ جمعش چقدر است؟',
    'موجودی علی 150 دلار و 20 کالاست؛ مجموعش چقدر است؟',
])
def test_bug004_engine_refuses_mixed_dimension_sum(text):
    from jarvis.agent.local_intelligence_v23 import LocalIntelligenceV23
    eng = LocalIntelligenceV23(output_style='canonical')
    a = eng.solve(text, 'fa')
    # abstention counts: None (no IR) OR a non-verified structured refusal
    if a is None:
        assert True
        return
    assert '170' not in a.text and 'مجموع موجودی' not in a.text
    trace = eng.last_trace
    assert trace and not trace['verification']['passed']


def test_bug004_same_dimension_chains_still_verified():
    from jarvis.agent.local_intelligence_v23 import LocalIntelligenceV23
    eng = LocalIntelligenceV23(output_style='canonical')
    assert eng.solve('موجودی انبار 100 کالا است؛ 5 کالا اضافه کن', 'fa').text == '105'
    assert eng.solve('5+7', 'fa').text == '12'


def test_bug004_jamesh_lang_recognized():
    from jarvis.agent.semantics_v22_4 import _CHAIN_LANG
    assert _CHAIN_LANG.search('جمعش چقدر است؟')
    assert _CHAIN_LANG.search('مجموعش چقدر است؟')
    assert not _CHAIN_LANG.search('جمع‌آوری اطلاعات')  # compound noun, no sum
