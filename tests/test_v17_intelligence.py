from __future__ import annotations
import json, unittest
from unittest import mock
from jarvis.agent.semantic_ir_v17 import SemanticIRParserV17
from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime

CASES=(
('احتمال شیر آن 70 درصد است و سکه را 4 بار می‌اندازیم؛ احتمال دقیقاً 3 شیر؟',('41.16',)),
('شانس شیر برای این سکه 65٪ است؛ 5 بار می‌اندازیم، احتمال دقیقاً 2 شیر؟',('18.1146875',)),
('برای سکه‌ای که احتمال شیرش 80٪ است، در 4 پرتاب احتمال دقیقاً 4 شیر؟',('40.96',)),
('A coin has heads probability 70%. In 4 tosses, exactly 3 heads?',('41.16',)),
('جایگشت‌های چهارتایی بدون جایگذاری از میان 11 شیء',('7920',)),
('آرایش سه‌تایی بدون تکرار از میان 10 گزینه چند حالت دارد؟',('720',)),
('از 8 نفر نفرات اول، دوم و سوم را انتخاب کنیم؛ چند حالت داریم؟',('336',)),
('از 9 نفر رتبه اول و دوم و سوم را انتخاب کنیم؛ چند حالت؟',('504',)),
('What is the chance that two fair dice add up to 11?',('1/18',)),
('Two fair dice add up to 5 with what probability?',('1/9',)),
('دو تاس سالم؛ احتمال اینکه جمعشان 10 شود؟',('1/12',)),
('دو تاس سالم؛ احتمال مجموع 4؟',('1/12',)),
('180 کیلومتر در 3 ساعت؛ سرعت متوسط چقدر است؟',('60',)),
('240 کیلومتر را در 4 ساعت می‌رویم؛ سرعت؟',('60',)),
('Distance 150 km in 2.5 hours. What is the speed?',('60',)),
('4 کارگر در 6 ساعت 120 جعبه؛ 2 کارگر در 3 ساعت چند جعبه؟',('30',)),
('6 کارگر در 5 ساعت 300 جعبه؛ 3 کارگر در 2 ساعت چند جعبه؟',('60',)),
('4 workers make 160 boxes in 8 hours. 2 workers in 4 hours make how many?',('40',)),
('علی > رضا > مهدی؛ آیا علی از مهدی بزرگ‌تر است؟',('علی','مهدی')),
('سارا > ندا > مینا؛ آیا سارا از مینا جلوتر است؟',('سارا','مینا')),
('آیا 97 فقط بر 1 و خودش بخش‌پذیر است؟',('بله','اول')),
('آیا 91 فقط بر 1 و خودش بخش‌پذیر است؟',('خیر','اول نیست')),
('یک تابع Python بنویس که فقط اعداد زوج را برگرداند',('def even_numbers','% 2 == 0')),
('Write a Python function that returns only even numbers',('def even_numbers','% 2 == 0')),
('210 را به نسبت 2 به 5 تقسیم کن',('60','150')),
('420 را به نسبت 3:4 تقسیم کن',('180','240')),
('Split 450 in ratio 2:3',('180','270')),
('احتمال شیر سکه 55٪ است و 2 بار می‌اندازیم؛ احتمال دقیقاً 1 شیر؟',('49.5',)),
('شانس شیر 30 درصد است؛ سکه را 3 بار می‌اندازیم. دقیقاً 2 شیر؟',('18.9',)),
('A coin has a 25% chance of heads. In 4 tosses exactly 1 head?',('42.1875',)),
('این پیام را رسمی‌تر کن: لطفا قراردادو تا دوشنبه برام بفرست',('قرارداد را','تا دوشنبه','برای من','ارسال کنید')),
('این متن را رسمی‌تر کن: مقاله رو برام بفرست',('مقاله را','برای من','ارسال کنید')),
('این جمله را رسمی‌تر کن: کد رو برام بفرست',('کد را','برای من','ارسال کنید')),
('این پیام را رسمی‌تر کن: پروژه رو زود بفرست',('پروژه را','ارسال کنید')),
('از 7 نفر نفرات اول، دوم و سوم را تعیین کنیم',('210',)),
('جایگشت سه‌تایی بدون جایگذاری از میان 6 شیء',('120',)),
('120 کیلومتر در 2 ساعت؛ سرعت چند است؟',('60',)),
('5 کارگر در 4 ساعت 200 جعبه؛ 2 کارگر در 2 ساعت چند جعبه؟',('40',)),
('دو تاس سالم؛ شانس اینکه مجموع 12 شود؟',('1/36',)),
('احتمال شیر برای این سکه 75٪ است؛ در 4 پرتاب دقیقاً 2 شیر؟',('21.09375',)),
)

class IntelligenceV17Tests(unittest.TestCase):
    def test_v17_semantic_ir_generalization_without_tools(self):
        with TemporaryRuntime() as rt:
            with mock.patch.object(rt.tools,'invoke',side_effect=AssertionError('unexpected tool')):
                for prompt,frags in CASES:
                    with self.subTest(prompt=prompt):
                        ans=rt.agent.respond(prompt)
                        low=ans.text.casefold()
                        for f in frags:self.assertIn(f.casefold(),low)
                        self.assertNotIn(ans.intent,{'tool_clarification','tool_search_result','tool_research_result','unknown_information','fact_not_found'})
    def test_v17_reported_frames_parse_to_ir(self):
        for prompt,_ in CASES[:30]:
            if 'رسمی' in prompt: continue
            with self.subTest(prompt=prompt):
                self.assertIsNotNone(SemanticIRParserV17.parse(prompt))
    def test_v17_holdout_not_exact_match_v008(self):
        seen=set()
        for split in ('train','validation','test'):
            p=ROOT/'datasets'/'splits'/f'dataset_v008_{split}.jsonl'
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():seen.add(normalize_text(json.loads(line)['input']).casefold())
        for prompt,_ in CASES:self.assertNotIn(normalize_text(prompt).casefold(),seen)

if __name__=='__main__':unittest.main()
