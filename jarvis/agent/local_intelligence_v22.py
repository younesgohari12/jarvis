"""JARVIS v22 intelligence engine — incremental upgrade over v21.

Preserves the complete v21 reasoning stack (parser, execution graph, world
model, verifier, reflection, code intelligence) and adds:

  * Verifier V2   — typed-dimension + numeric-role invariants (independent)
  * Reflection V2 — targeted repair with traceability
  * Temporal world model — clock arithmetic with calendar wrap
  * Language Brain V2 — natural Persian/English rendering + NLU retry
  * Semantic IR V2 — typed quantities + provenance (audit layer)
  * Dimension-safe abstention — never silently combine USD + items

Nothing from v21 is removed or reinterpreted: v22 only narrows what may pass
as verified and improves how verified results are spoken.
"""
from __future__ import annotations
from fractions import Fraction

from jarvis.agent.local_intelligence_v21 import LocalIntelligenceV21
from jarvis.agent.verifier_v22 import UniversalVerifierV2
from jarvis.agent.reflection_v22 import ReflectionLoopV22
from jarvis.agent.semantic_ir_v22 import SemanticIRV2
from jarvis.agent import temporal_v22
from jarvis.agent import language_brain_v22 as lb


class LocalIntelligenceV22(LocalIntelligenceV21):
    VERSION = '4.0.0-v22'

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
        self._active_source = text
        # 1) code path is inherited verbatim from v21 through super().solve().
        answer = super().solve(text, language)

        # 2) NLU retry: informal Persian/English gets one normalized retry.
        if answer is None:
            ntext = lb.normalize_nlu(text)
            if ntext != text:
                answer = super().solve(ntext, language)
                if answer is not None and self.last_trace:
                    self.last_trace['v22_nlu_normalized'] = True

        # 3) dimension-safe abstention — a failed typed check never hides.
        if self.last_trace and not self.last_trace['verification']['passed']:
            failed = self.last_trace['verification'].get('failed_checks', [])
            if 'source_operation_dimension_consistency' in failed or 'currency_dimension_mismatch' in failed \
                    or 'source_currency_consistency' in failed:
                reason = 'currency_item'
                if 'source_currency_consistency' in failed:
                    reason = 'currency_mix'
                message = lb.clarification(language, reason)
                from jarvis.agent.local_intelligence_v14 import LocalIntelligenceAnswer
                self.last_trace['v22_dimension_block'] = reason
                return LocalIntelligenceAnswer(message, 'reasoned_answer', .2,
                                               ('v22_dimension_guard', *failed[:4]))
            # conservative refusals get ONE paraphrase retry (Language Brain NLU)
            text_l = (answer.text if answer is not None else '') or ''
            if ('قابل تأیید نیست' in text_l or 'کافی نیست' in text_l
                    or 'could not verify' in text_l.lower()
                    or 'independently verify' in text_l.lower()
                    or 'سازگار نیست' in text_l):
                ntext = lb.normalize_nlu(text)
                if ntext != text:
                    retry = super().solve(ntext, language)
                    if retry is not None and self.last_trace and self.last_trace['verification']['passed']:
                        self.last_trace['v22_nlu_normalized'] = True
                        self._active_source = ntext
                        answer = retry
                        if self.last_trace.get('verification', {}).get('passed'):
                            self.ir_v2 = self._upgrade_ir(ntext)
                            try:
                                return self._naturalize(answer, language)
                            except Exception:
                                return answer
            return answer

        # 4) verified answers: upgrade IR, natural rendering, temporal override.
        if answer is not None and self.last_trace and self.last_trace.get('verification', {}).get('passed'):
            self.ir_v2 = self._upgrade_ir(text)
            try:
                answer = self._naturalize(answer, language)
            except Exception:  # rendering must never break a verified answer
                pass
        return answer

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
            clock = temporal_v22.ClockTime(
                int(start) if float(start).is_integer() else 0,
                int(round((float(start) % 1) * 60)) if not float(start).is_integer() else 0)
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
        from jarvis.agent.verification_v21 import AUTHORITATIVE_SOURCE
        return AUTHORITATIVE_SOURCE.get() or getattr(self, '_active_source', '') or ''
