from __future__ import annotations
import re
from jarvis.agent.rewrite_v17 import RewriteEngineV17, RewriteResult

class RewriteEngineV18(RewriteEngineV17):
    VERSION='1.8.0'
    @staticmethod
    def _formal_fa(source:str)->str:
        v=source.strip()
        # General detached object clitic: «فایل رو» -> «فایل را».
        v=re.sub(r'([\u0600-\u06ff‌]+)\s+رو\b',r'\1 را',v)
        # ZWNJ spelling such as «نتیجه‌رو» is a full clitic, not a final واو.
        v=re.sub(r'([\u0600-\u06ff]{2,})‌رو\b',r'\1 را',v)
        # Common attached colloquial object marker when followed by a deadline/recipient/send verb.
        v=re.sub(r'([\u0600-\u06ff‌]{3,})و(?=\s+(?:تا\s|برام|برات|واسم|واست|بفرست|ارسال|فردا|امروز|دوشنبه|سه.?شنبه))',r'\1 را',v)
        for p,r in ((r'\bبرام\b','برای من'),(r'\bواسم\b','برای من'),(r'\bبرات\b','برای شما'),(r'\bواست\b','برای شما'),(r'\bبهم\b','به من'),(r'\bبهت\b','به شما')):
            v=re.sub(p,r,v)
        return RewriteEngineV17._formal_fa(v)

    @staticmethod
    def _meaning_guard(source:str,rewritten:str)->bool:
        src=re.sub(r'([\u0600-\u06ff‌]+)\s+رو\b',r'\1',source)
        src=re.sub(r'([\u0600-\u06ff]{2,})‌رو\b',r'\1',src)
        src=re.sub(r'([\u0600-\u06ff‌]{3,})و(?=\s+(?:تا\s|برام|برات|واسم|واست|بفرست|فردا|امروز))',r'\1',src)
        for p,r in ((r'\bبرام\b','برای من'),(r'\bواسم\b','برای من'),(r'\bبرات\b','برای شما'),(r'\bواست\b','برای شما')):src=re.sub(p,r,src)
        return RewriteEngineV17._meaning_guard(src,rewritten)
__all__=['RewriteEngineV18','RewriteResult']
