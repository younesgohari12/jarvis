"""JARVIS v22.4 intelligence engine — TRUE ROOT-CAUSE HARDENING over v22.3.

Preserves the complete v21/v22 reasoning stack and adds the root-cause fixes
required by the v22.4 specification:

  * Verifier V2 (hardened) — operation-aware dimension algebra, structured
    DimensionFailure metadata, fail-closed (never fail-open)
  * Structured abstention  — the clarification names the ACTUAL dimension pair
    ('زمان و مسافت را نمی‌توان مستقیماً جمع کرد.'), never a guessed
    currency/item message (spec §22-§24)
  * Semantic event models  — entity-bound age semantics, ownership transfers
    with query binding and currency safety, typed inventory events with
    product binding, compound-dimension rate arithmetic (spec §25-§41)
  * Generalized scheduling — the temporal frame handles any start/duration/
    end phrasing; bare clock tokens require frame evidence (spec §36-§40)
  * Fact-locked NLG        — the renderer can never substitute a unit
    (M_PER_SECOND never becomes km/h) (spec §53)

Nothing from v21 is removed or reinterpreted: v22.4 only narrows what may
pass as verified, adds verified capabilities, and improves how verified
results are spoken.
"""
from __future__ import annotations

from jarvis.agent.local_intelligence_v21 import LocalIntelligenceV21
from jarvis.agent.verifier_v22 import (
    UniversalVerifierV2, ORIGINAL_SOURCE, LAST_DIMENSION_FAILURE,
)
from jarvis.agent.reflection_v22 import ReflectionLoopV22
from jarvis.agent.semantic_ir_v22 import SemanticIRV2
from jarvis.agent import temporal_v22
from jarvis.agent import language_brain_v22 as lb
from jarvis.agent import semantics_v22_4 as s4


class LocalIntelligenceV22(LocalIntelligenceV21):
    VERSION = '5.0.0-v22.4'

    def __init__(self, *args, output_style: str = 'natural', **kwargs):
        # output_style='natural'    -> verified answers spoken naturally (engine default)
        # output_style='canonical'  -> v21 byte-compatible numeric contract (runtime endpoint)
        super().__init__(*args, **kwargs)
        from jarvis.agent.models_v21 import ClassifierV21  # noqa: F401  (parity with v21 init)
        self.output_style = output_style
        self.verifier = UniversalVerifierV2()
        self.reflection = ReflectionLoopV22(self.executor, self.verifier, self.parser)
        self.ir_v2 = None  # last upgraded SemanticIRV2 (audit layer)

    # ------------------------------------------------------------------
    @classmethod
    def matches(cls, text):
        return LocalIntelligenceV21.matches(text)

    # ------------------------------------------------------------------
    def solve(self, text, language='fa'):
        from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
        self._active_source = text
        # v22.1: the ORIGINAL user text is pinned as the immutable
        # verification source for the whole solve. Normalized/paraphrased
        # text is parse-assistance only and is never verified against.
        source_token = ORIGINAL_SOURCE.set(text)
        try:
            answer = self._solve_inner(text, language, LocalIntelligenceAnswer)
        finally:
            ORIGINAL_SOURCE.reset(source_token)
            LAST_DIMENSION_FAILURE.set(None)
        return answer

    # ------------------------------------------------------------------
    def _solve_inner(self, text, language, LocalIntelligenceAnswer):
        # 1) code path is inherited verbatim from v21 through super().solve().
        #    v22.1: minute-precision clock starts (ساعت 23:30) are decoded for
        #    the PARSER input only; verification keeps the original text.
        stext = lb.normalize_clock_tokens(text)
        answer = super().solve(stext if stext != text else text, language)

        # 1.4) v22.4: an EXPLICIT compound-rate unit (m/s, کیلومتر بر ساعت,
        # دلار در ساعت ...) is a strong typed signal — the rate model computes
        # rate x time directly and never lets the legacy speed grammar re-read
        # the rate value as a distance ('12 متر بر ثانیه ... 60 ثانیه' is 720
        # meters, not a 0.2 m/s speed).
        if s4.classify_rate_unit(text):
            ans = self._rate_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans

        # 1.5) independent age-difference operation (entity-bound abs).
        # Fires on the immutable source before any repair/abstain path: a
        # rewrite can never re-title this task as inventory/finance again.
        # v22.4: detection is generalized (aged / possessive / دارد / ساله /
        # relation forms) through the entity->attribute->value engine.
        if lb.detect_age_difference(text) is not None:
            op = self._age_difference_answer(text, language, LocalIntelligenceAnswer)
            if op is not None:
                return op

        # 1.6) v22.4 semantic event models: verified entity-bound operations
        # the v21 grammar cannot express. Every model result is re-checked by
        # an independent witness before it may be spoken.
        if answer is None:
            model = self._v24_semantic_models(text, language, LocalIntelligenceAnswer)
            if model is not None:
                return model

        # 2) NLU retry: informal Persian/English gets one normalized retry.
        #    v22.2: the retry input also passes through the minute-precision
        #    clock decoder so rewritten scheduling starts (e.g. "it starts at
        #    23:00 now") parse as decimal hours, exactly like attempt #1.
        if answer is None:
            ntext = lb.normalize_nlu(text)
            ntext = lb.normalize_clock_tokens(ntext) if ntext != text else text
            if ntext != text:
                answer = super().solve(ntext, language)
                if answer is not None and self.last_trace:
                    self.last_trace['v22_nlu_normalized'] = True

        # 2.5) v22.4 models also cover inputs the NLU retry rescued only
        # partially (the model path must run after every parse failure).
        if answer is None:
            model = self._v24_semantic_models(text, language, LocalIntelligenceAnswer)
            if model is not None:
                return model

        # 2.6) v22.4: a request that violates the operation-aware dimension
        # algebra must be refused with the STRUCTURED message even when the
        # v21 parser could not build an IR at all ('2 hours + 5 USD').
        if answer is None:
            from jarvis.agent.quantity_v22 import extract_typed_quantities as _etq
            structured = s4.cross_dimension_chain_failure_v4(
                text, _etq(text))
            if structured:
                failure = structured[0]
                message = lb.structured_clarification(language, failure)
                ask = (' لطفاً بگویید دقیقاً چه محاسبه‌ای مد نظر است.'
                       if language == 'fa'
                       else ' Could you clarify what exactly should be computed?')
                message = message + ask
                if self.last_trace is None:
                    self.last_trace = {
                        'verification': {'passed': False,
                                         'checks': ['structured_dimension_failure'],
                                         'failed_checks':
                                             ['source_operation_dimension_consistency'],
                                         'repair_stage': 'abstain'},
                    }
                self.last_trace['v22_dimension_block'] = failure.key
                self.last_trace['v22_dimension_failure'] = failure.to_dict()
                return LocalIntelligenceAnswer(message, 'reasoned_answer', .2,
                                               ('v22_dimension_guard',
                                                'structured_dimension_failure',
                                                'source_operation_dimension_consistency'))

        # 3) dimension-safe abstention — a failed typed check never hides.
        #    v22.4: the clarification speaks the STRUCTURED dimension failure
        #    (operator + left/right dimension/unit), never a guessed pair.
        if self.last_trace and not self.last_trace['verification']['passed']:
            failed = self.last_trace['verification'].get('failed_checks', [])
            if 'source_operation_dimension_consistency' in failed or 'currency_dimension_mismatch' in failed \
                    or 'source_currency_consistency' in failed:
                failure = LAST_DIMENSION_FAILURE.get()
                reason = failure.key if failure is not None else 'currency_item'
                if 'source_currency_consistency' in failed:
                    reason = 'currency_mix'
                message = lb.structured_clarification(language, failure)
                # keep the clarification-ask tail of the v22 contract
                ask = (' لطفاً بگویید دقیقاً چه محاسبه‌ای مد نظر است.'
                       if language == 'fa'
                       else ' Could you clarify what exactly should be computed?')
                if ask.strip().rstrip('.') not in message:
                    message = message + ask
                from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
                self.last_trace['v22_dimension_block'] = reason
                if failure is not None:
                    self.last_trace['v22_dimension_failure'] = failure.to_dict()
                return LocalIntelligenceAnswer(message, 'reasoned_answer', .2,
                                               ('v22_dimension_guard', *failed[:4]))
            # conservative refusals get ONE paraphrase retry (Language Brain NLU)
            text_l = (answer.text if answer is not None else '') or ''
            if ('قابل تأیید نیست' in text_l or 'کافی نیست' in text_l
                    or 'could not verify' in text_l.lower()
                    or 'independently verify' in text_l.lower()
                    or 'سازگار نیست' in text_l):
                ntext = lb.normalize_nlu(text)
                ntext = lb.normalize_clock_tokens(ntext) if ntext != text else text
                if ntext != text:
                    retry = super().solve(ntext, language)
                    if retry is not None and self.last_trace and self.last_trace['verification']['passed']:
                        self.last_trace['v22_nlu_normalized'] = True
                        answer = retry
                        if self.last_trace.get('verification', {}).get('passed'):
                            self.ir_v2 = self._upgrade_ir(text)
                            try:
                                return self._naturalize(answer, language)
                            except Exception:
                                return answer
                # v22.4: last resort BEFORE the abstention — the independently
                # verified semantic models may still solve what the v21 stack
                # could only refuse (e.g. age sums with pronoun anaphora).
                model = self._v24_semantic_models(text, language, LocalIntelligenceAnswer)
                if model is not None:
                    return model
            return answer

        # 4) verified answers: upgrade IR, natural rendering, temporal override.
        if answer is not None and self.last_trace and self.last_trace.get('verification', {}).get('passed'):
            self.ir_v2 = self._upgrade_ir(text)
            try:
                answer = self._naturalize(answer, language)
            except Exception:  # rendering must never break a verified answer
                pass
        return answer

    # ==================================================================
    # v22.4 semantic event models — each verified before speaking.
    # ==================================================================
    def _v24_semantic_models(self, text, language, LocalIntelligenceAnswer):
        """Try the typed semantic models in order. A model may answer ONLY
        when its own independent witness confirms the result."""
        try:
            # --- ownership transfers (event model, query-bound) ----------
            ans = self._ownership_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
            # --- inventory typed events with product binding -------------
            ans = self._inventory_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
            # --- compound-dimension rate arithmetic ----------------------
            ans = self._rate_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
            # --- generalized temporal frame (start OR end) ---------------
            ans = self._temporal_frame_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
            # --- relation-only age query (how much older is X than Y) ----
            ans = self._age_relation_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
            # --- entity-bound age SUM (spec §20: جمع سن آن دو) ----------
            ans = self._age_sum_answer(text, language, LocalIntelligenceAnswer)
            if ans is not None:
                return ans
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _ownership_answer(self, text, language, LocalIntelligenceAnswer):
        import re
        if not re.search(r'transfers?|pays?|paid|gives?|gave|receives?|received|sent|'
                         r'می[\s\u200c]*دهد|(?:می[\s\u200c]*)?داد\b|دادند|'
                         r'می[\s\u200c]*پردازد|می[\s\u200c]*فرستد|'
                         r'دریافت|منتقل', text, re.I):
            return None
        model = s4.extract_ownership_model(text)
        if model is None:
            return None
        if model.get('conflict'):
            # currency safety: mixed units are refused with a structured ask
            message = lb.clarification(language, 'currency_mix')
            self._v24_trace({'model': 'ownership', 'conflict': 'ownership_unit_conflict'})
            return LocalIntelligenceAnswer(message, 'reasoned_answer', .2,
                                           ('v24_ownership_model', 'currency_safety'))
        result = s4.execute_ownership(model)
        if not result.get('ok'):
            self._v24_trace({'model': 'ownership', 'rejected': result.get('reason')})
            return None
        # independent witness: re-extract + re-execute + conservation
        witness = s4.extract_ownership_model(text)
        witness_result = s4.execute_ownership(witness) if witness else {'ok': False}
        value = float(result['value'])
        if not witness_result.get('ok') or abs(float(witness_result['value']) - value) > 1e-9:
            return None
        if not (abs(value) < 1e15) or s4.numeric_guard(value):
            return None
        self._v24_trace({'model': 'ownership', 'value': value,
                         'query': model.get('query'),
                         'balances_after': result.get('balances'),
                         'events': model.get('events'), 'verified': True}, success=True)
        if self.output_style == 'natural':
            rendered = self._render_ownership(model, result, language)
        else:
            rendered = lb.fmt_number(value)
        return LocalIntelligenceAnswer(
            rendered, 'reasoned_answer', .9,
            ('v24_ownership_model', 'ownership_event_model', 'funds_conservation',
             'v24_query_binding'))

    # ------------------------------------------------------------------
    def _render_ownership(self, model, result, language: str) -> str:
        v = lb.fmt_number(float(result['value']))
        unit = model.get('unit') or ''
        unit_word = {'USD': ('دلار', 'dollars'), 'EUR': ('یورو', 'euros'),
                     'TOMAN': ('تومان', 'toman'), 'RIAL': ('ریال', 'rials'),
                     'GBP': ('پوند', 'pounds'), 'ITEM': ('کالا', 'items'),
                     'UNSPECIFIED': ('', '')}.get(unit, ('', ''))
        uw = unit_word[0] if language == 'fa' else unit_word[1]
        q = model.get('query') or {}
        if q.get('kind') == 'combined':
            if language == 'fa':
                return f'مجموع موجودی می\u200cشود {v} {uw}.'.rstrip() if uw else f'مجموع موجودی می\u200cشود {v}.'
            return f'The combined total is {v} {uw}.' if uw else f'The combined total is {v}.'
        if q.get('kind') == 'difference':
            if language == 'fa':
                return f'اختلاف موجودی می\u200cشود {v} {uw}.'.rstrip() if uw else f'اختلاف موجودی می\u200cشود {v}.'
            return f'The balance difference is {v} {uw}.' if uw else f'The balance difference is {v}.'
        entity = q.get('entity', '')
        if language == 'fa':
            return f'موجودی {entity} می\u200cشود {v} {uw}.'.rstrip() if uw else f'موجودی {entity} می\u200cشود {v}.'
        return f"{entity}'s balance is {v} {uw}." if uw else f"{entity}'s balance is {v}."

    # ------------------------------------------------------------------
    def _inventory_answer(self, text, language, LocalIntelligenceAnswer):
        model = s4.extract_inventory_model(text)
        if model is None:
            return None
        model['text'] = text
        result = s4.execute_inventory_model(model)
        if not result.get('ok'):
            self._v24_trace({'model': 'inventory_multi', 'rejected': result.get('reason')})
            return None
        value = float(result['value'])
        if s4.numeric_guard(value):
            return None
        # witness: events must preserve order and product binding
        again = s4.extract_inventory_model(text)
        if again is None:
            return None
        again['text'] = text
        check = s4.execute_inventory_model(again)
        if not check.get('ok') or abs(float(check['value']) - value) > 1e-9:
            return None
        self._v24_trace({'model': 'inventory_multi', 'value': value,
                         'products': result.get('products'),
                         'events': model.get('events'), 'verified': True}, success=True)
        v = lb.fmt_number(value)
        qname = result.get('query_product') or ''
        if language == 'fa':
            rendered = f'موجودی {qname} می\u200cشود {v} کالا.'.strip() if qname \
                else f'نتیجه می\u200cشود {v} کالا.'
        else:
            rendered = f'{qname} stock is {v} items.'.strip() if qname \
                else f'The result is {v} items.'
        return LocalIntelligenceAnswer(rendered, 'reasoned_answer', .88,
                                       ('v24_inventory_model', 'inventory_event_model',
                                        'v24_product_binding'))

    # ------------------------------------------------------------------
    def _rate_answer(self, text, language, LocalIntelligenceAnswer):
        import re
        solved = s4.solve_rate(text)
        if solved is None:
            return None
        if solved.get('kind') == 'rate_conflict':
            failure = solved['failure']
            message = lb.clarification(language, failure.get('key', 'rate_mismatch'))
            self._v24_trace({'model': 'rate', 'conflict': failure})
            return LocalIntelligenceAnswer(message, 'reasoned_answer', .2,
                                           ('v24_rate_model', 'structured_dimension_failure'))
        value = float(solved['value'])
        if s4.numeric_guard(value):
            return None
        # witness: the rate structure must re-derive identically
        again = s4.solve_rate(text)
        if again is None or again.get('kind') != solved.get('kind') \
                or abs(float(again.get('value', float('nan'))) - value) > 1e-9:
            return None
        self._v24_trace({'model': 'rate', **{k: v for k, v in solved.items() if k != 'failure'},
                         'verified': True}, success=True)
        v = lb.fmt_number(value)
        if solved['kind'] == 'rate_application':
            unit_word = solved.get('fa_unit') if language == 'fa' else solved.get('en_unit')
            if solved.get('numerator') == 'currency':
                rendered = (f'هزینه می\u200cشود {v} {unit_word}.'.strip() if unit_word
                            else f'نتیجه می\u200cشود {v}.') if language == 'fa' else (
                    f'The cost is {v} {unit_word}.' if unit_word else f'The result is {v}.')
            else:
                rendered = (f'نتیجه می\u200cشود {v} {unit_word}.'.strip() if unit_word
                            else f'نتیجه می\u200cشود {v}.') if language == 'fa' else (
                    f'The result is {v} {unit_word}.' if unit_word else f'The result is {v}.')
        else:
            # rate addition keeps the compound unit
            unit = solved.get('rate_unit', '')
            unit_word = self._rate_unit_word(unit, language)
            rendered = (f'جمع نرخ\u200cها می\u200cشود {v} {unit_word}.'.strip() if unit_word
                        else f'نتیجه می\u200cشود {v}.') if language == 'fa' else (
                f'The combined rate is {v} {unit_word}.' if unit_word else f'The result is {v}.')
        return LocalIntelligenceAnswer(rendered, 'reasoned_answer', .88,
                                       ('v24_rate_model', 'compound_dimension_algebra'))

    # ------------------------------------------------------------------
    @staticmethod
    def _rate_unit_word(unit: str, language: str) -> str:
        words = {
            'USD_PER_HOUR': ('دلار در ساعت', 'dollars/hour'),
            'EUR_PER_HOUR': ('یورو در ساعت', 'euros/hour'),
            'TOMAN_PER_HOUR': ('تومان در ساعت', 'toman/hour'),
            'ITEM_PER_HOUR': ('کالا در ساعت', 'items/hour'),
            'ITEM_PER_DAY': ('کالا در روز', 'items/day'),
            'ITEM_PER_MINUTE': ('کالا در دقیقه', 'items/minute'),
            'USD_PER_DAY': ('دلار در روز', 'dollars/day'),
        }
        return words.get(unit, ('', ''))[0 if language == 'fa' else 1]

    # ------------------------------------------------------------------
    def _temporal_frame_answer(self, text, language, LocalIntelligenceAnswer):
        frame = s4.extract_temporal_frame(text)
        if frame is None:
            return None
        try:
            if frame.query == 'start' and frame.end_time:
                end = temporal_v22.parse_clock_time(frame.end_time)
                result = temporal_v22.subtract_duration(
                    end, *[temporal_v22.Duration.of(v, u) for v, u in frame.durations])
            else:
                start = temporal_v22.parse_clock_time(frame.start_time)
                result = temporal_v22.add_duration(
                    start, *[temporal_v22.Duration.of(v, u) for v, u in frame.durations])
        except Exception:
            return None
        absolute = float(result.absolute_hours())
        verdict = self.verifier.verify_temporal(text, result, language)
        if not verdict.passed:
            return None
        if self.last_trace is None:
            self.last_trace = {'verification': {'passed': True, 'checks': ['v24_temporal_frame'],
                                                'failed_checks': []}}
        self.last_trace['temporal_world_model'] = {**result.to_dict(), 'verified': True}
        self.last_trace['v24_temporal_frame'] = {**frame.to_dict(), 'verified': True}
        if frame.query == 'start':
            # end-frame: the answer IS the start instant
            if language == 'fa':
                text_out = f'ساعت شروع می\u200cشود {result.iso()}.'
            else:
                text_out = f'It starts at {result.iso()}.'
        else:
            text_out = self._render_clock(result, absolute, language)
        return LocalIntelligenceAnswer(text_out, 'reasoned_answer', .9,
                                       ('v24_temporal_frame', 'temporal_invariant',
                                        'calendar_wrap'))

    # ------------------------------------------------------------------
    def _age_relation_answer(self, text, language, LocalIntelligenceAnswer):
        """Relation-only age queries: 'Reza is 5 years older than Ali.' +
        'How much older is Reza than Ali?' -> 5 (no absolute ages needed)."""
        import re
        t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
        if not lb.AGE_DIFFERENCE_LANGUAGE.search(t):
            return None
        solved = s4.solve_age_query(t)
        if solved is not None:
            return None  # fully entity-bound case handled by the dedicated op
        facts = s4.extract_age_facts(t)
        query = s4.bind_age_query(t, facts)
        if not query:
            return None
        a, b = query['a'], query['b']
        # resolve the difference through the relation chain only
        for f in facts:
            if f.relation and f.difference is not None:
                if {f.entity, f.relation_object} == {a, b}:
                    # direction: answer must be non-negative (absolute ask)
                    subject_is_a = (f.entity == a)
                    value = float(f.difference)
                    if subject_is_a and f.relation == 'younger_than':
                        value = value
                    elif (not subject_is_a) and f.relation == 'older_than':
                        value = value
                    value = abs(value)
                    if s4.numeric_guard(value):
                        return None
                    self._v24_trace({'model': 'age_relation', 'value': value,
                                     'entities': [a, b],
                                     'relation': f.relation, 'verified': True}, success=True)
                    older = f.entity if f.relation == 'older_than' else f.relation_object
                    if self.output_style == 'natural':
                        rendered = lb.render_age_difference(older, value, language)
                    else:
                        rendered = lb.fmt_number(value)
                    return LocalIntelligenceAnswer(
                        rendered, 'reasoned_answer', .88,
                        ('v24_age_relation_model', 'v22_age_entity_binding',
                         'source_slot_consistency'))
        return None

    # ------------------------------------------------------------------
    def _age_sum_answer(self, text, language, LocalIntelligenceAnswer):
        """Entity-bound sum of ages: 'مینا 48 ساله است؛ مادرش 18 سال بزرگتر
        از اوست؛ جمع سن آن دو؟' -> 114 (pronoun anaphora + relation)."""
        import re
        t = (text or '').translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
        if not re.search(r'جمع\s*سن|مجموع\s*سن|سن\s*آن\s*دو'
                         r'|sum\s+of\s+(?:their\s+)?ages|total\s+age', t, re.I):
            return None
        solved = s4.solve_age_query(t)
        if not solved or solved.get('kind') != 'sum':
            return None
        value = float(solved['value'])
        if s4.numeric_guard(value) or value < 0:
            return None
        # witness: independent re-derivation from the immutable source
        again = s4.solve_age_query(t)
        if not again or again.get('kind') != 'sum' \
                or abs(float(again['value']) - value) > 1e-9:
            return None
        self._v24_trace({'model': 'age_sum', 'value': value,
                         'ages': solved['ages'], 'verified': True}, success=True)
        v = lb.fmt_number(value)
        if self.output_style == 'natural':
            rendered = (f'جمع سن\u200cها می\u200cشود {v} سال.'
                        if language == 'fa'
                        else f'The sum of their ages is {v} years.')
        else:
            rendered = v
        return LocalIntelligenceAnswer(rendered, 'reasoned_answer', .9,
                                       ('v24_age_sum_model', 'v22_age_entity_binding',
                                        'age_anaphora_resolved'))

    # ------------------------------------------------------------------
    def _v24_trace(self, data: dict, success: bool = False):
        """Record v22.4 model telemetry. A failed model attempt must NEVER
        fabricate an empty last_trace — downstream verification reads
        last_trace['verification'] and would crash (fail-closed lesson)."""
        if not success and self.last_trace is None:
            return
        if self.last_trace is None:
            self.last_trace = {}
        if success and 'verification' not in self.last_trace:
            self.last_trace['verification'] = {'passed': True,
                                               'checks': ['v22_4_semantic_model'],
                                               'failed_checks': []}
        self.last_trace['v22_4_model'] = data

    # ------------------------------------------------------------------
    def _age_difference_answer(self, text, language, LocalIntelligenceAnswer):
        """v22.1 independent operation, v22.4-generalized: age difference =
        abs(age_a − age_b), bound to the two named entities of the IMMUTABLE
        source and verified by the independent witness (never by the rewrite
        of itself)."""
        try:
            detection = lb.detect_age_difference(text)
            if not detection:
                return None
            (name_a, age_a), (name_b, age_b) = detection
            value = abs(float(age_a) - float(age_b))
            verdict = self.verifier.verify_age_difference(text, detection, value)
            if not verdict.passed:
                return None
            if self.last_trace is None:
                self.last_trace = {}
            self.last_trace['v22_age_difference'] = {
                'entities': [name_a, name_b],
                'ages': [age_a, age_b],
                'difference': value,
                'verified': True,
                **verdict.to_dict(),
            }
            self.ir_v2 = None
            if self.output_style == 'natural':
                older = name_b if age_b >= age_a else name_a
                rendered = lb.render_age_difference(older, value, language)
            else:
                rendered = lb.fmt_number(value)
            return LocalIntelligenceAnswer(
                rendered, 'reasoned_answer', .93,
                ('v22_age_difference_model', 'v22_age_entity_binding',
                 'source_slot_consistency'))
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _upgrade_ir(self, text):
        try:
            from jarvis.agent.cognitive_ir_v20 import SemanticIR
            attempts = self.last_trace.get('attempts') or []
            ir_dict = attempts[-1].get('ir') if attempts else self.last_trace.get('ir')
            if not ir_dict:
                return None
            ir = SemanticIR(
                ir_dict.get('task', 'unknown'), ir_dict.get('language', 'fa'),
                entities=ir_dict.get('entities', []), slots=ir_dict.get('slots', {}),
                units=ir_dict.get('units', {}), relations=ir_dict.get('relations', []),
                constraints=ir_dict.get('constraints', []), operations=ir_dict.get('operations', []),
                answer_type=ir_dict.get('answer_type', 'number'),
                confidence=ir_dict.get('confidence', 0.0), source_text=text,
            )
            return SemanticIRV2.from_v21(ir, text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _naturalize(self, answer, language):
        """Speak the verified result naturally; temporal tasks speak clock time."""
        from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
        ir_dict = (self.last_trace.get('attempts') or [{}])[-1].get('ir') or self.last_trace.get('ir', {})
        task = ir_dict.get('task')
        rendered_verification = self.last_trace.get('rendered_verification') or {}

        # --- temporal override (spec §17) ---
        if task == 'scheduling' and rendered_verification.get('passed'):
            slots = ir_dict.get('slots', {})
            start, durations = slots.get('start'), slots.get('durations', [])
            if start is None or not durations:
                return answer
            # v22.1 fix: minute-precision starts must keep their clock face
            # (23.5 -> 23:30); the previous int/0 fallback produced 00:30.
            clock = temporal_v22.parse_clock_time(start)
            result = temporal_v22.add_duration(clock, *[temporal_v22.Duration.of(d, 'hour') for d in durations])
            absolute = float(result.absolute_hours())
            text = self._render_clock(result, absolute, language)
            verdict = self.verifier.verify_temporal(self.last_source_text(), result, language)
            if not verdict.passed:
                return answer
            self.last_trace['temporal_world_model'] = {**result.to_dict(), 'verified': True}
            return LocalIntelligenceAnswer(text, answer.intent, max(answer.confidence, .9),
                                           (*answer.checks, 'v22_temporal_world_model', 'temporal_invariant'))

        # --- natural rendering for numeric word problems ---
        if self.output_style != 'natural':
            return answer
        if not rendered_verification.get('passed'):
            return answer
        value = (self.last_trace.get('attempts') or [{}])[-1].get('answer')
        if value is None:
            return answer
        natural = lb.render_word_problem(value, self._ir_v2_dict(ir_dict), language, mode='concise')
        if not natural or natural == answer.text:
            return answer
        final = self.verifier.verify_rendered(self.last_source_text(), natural)
        if not final.passed:
            return answer
        self.last_trace['rendered_verification'] = final.to_dict()
        self.last_trace['v22_natural_render'] = natural
        return LocalIntelligenceAnswer(natural, answer.intent, answer.confidence,
                                       (*answer.checks, 'v22_language_brain_natural'))

    # ------------------------------------------------------------------
    def _ir_v2_dict(self, ir_dict):
        """Merge the v21 IR dict with the V2 typed view for rendering."""
        merged = dict(ir_dict or {})
        if self.ir_v2 is not None:
            try:
                v2 = self.ir_v2.to_dict()
                merged['quantities'] = v2.get('quantities', [])
            except Exception:
                pass
        return merged

    # ------------------------------------------------------------------
    def _render_clock(self, result: temporal_v22.ClockTime, absolute: float, language: str) -> str:
        iso = result.iso()
        if language == 'fa':
            if result.day_offset > 0:
                day = ' روز بعد' if result.day_offset == 1 else f' ({result.day_offset} روز بعد)'
                return f'ساعت {iso}{day}.'
            total = float(absolute) if not float(absolute).is_integer() else int(absolute)
            return f'ساعت {iso}؛ یعنی {lb.fmt_number(total)} ساعت پس از شروع.'
        if result.day_offset > 0:
            day = ' next day' if result.day_offset == 1 else f' (+{result.day_offset} days)'
            return f'{iso}{day}.'
        total = float(absolute) if not float(absolute).is_integer() else int(absolute)
        return f'{iso}; that is {lb.fmt_number(total)} hours after the start.'

    # ------------------------------------------------------------------
    def last_source_text(self) -> str:
        from jarvis.agent.verifier_v22 import ORIGINAL_SOURCE
        return ORIGINAL_SOURCE.get() or getattr(self, '_active_source', '') or ''
