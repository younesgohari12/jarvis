from __future__ import annotations
import re
from jarvis.agent.rewrite_v16 import RewriteEngineV16, RewriteResult

class RewriteEngineV17(RewriteEngineV16):
    """v17 Persian register normalization layered on fidelity-preserving v16 rewrite."""
    @staticmethod
    def _formal_fa(source:str)->str:
        value=source.strip()
        extra=(
            (r'\bقراردادو\b','قرارداد را'),(r'\bبرام\b','برای من'),(r'\bبرات\b','برای شما'),
            (r'\bمقاله رو\b','مقاله را'),(r'\bکد رو\b','کد را'),(r'\bپروژه رو\b','پروژه را'),(r'\bنسخه رو\b','نسخه را'),(r'\bسند رو\b','سند را'),(r'\bبرنامه رو\b','برنامه را'),
            (r'\bمیخوام\b','می‌خواهم'),(r'\bمیخواد\b','می‌خواهد'),(r'\bمیشه\b','امکان دارد'),
            (r'\bبفرست\b','ارسال کنید'),(r'\bبذار\b','قرار دهید'),(r'\bبگیر\b','دریافت کنید'),
        )
        for p,r in extra: value=re.sub(p,r,value,flags=re.I)
        return RewriteEngineV16._formal_fa(value)

    @staticmethod
    def _meaning_guard(source:str, rewritten:str)->bool:
        # Preserve v16 guard while normalizing protected colloquial noun suffixes.
        normalized_source=re.sub(r'(?i)(قراردادو|قرارداد\s+رو)', 'قرارداد', source)
        normalized_source=re.sub(r'(?i)برام','برای من',normalized_source)
        return RewriteEngineV16._meaning_guard(normalized_source,rewritten)

__all__=['RewriteEngineV17','RewriteResult']
