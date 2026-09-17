"""Bounded bilingual query expansion and explicit self-contained task protection."""
from __future__ import annotations
import re
from jarvis.agent.normalization_v20 import normalize

# Small documented retrieval lexicon; not a translator or a claim of broad bilingual QA.
ALIASES=[('memory','حافظه'),('processor','پردازنده'),('network','شبکه'),('database','پایگاه داده'),
         ('photosynthesis','فتوسنتز'),('gravity','گرانش'),('temperature','دما'),('electricity','برق'),
         ('computer','رایانه'),('water','آب'),('energy','انرژی'),('algorithm','الگوریتم')]

def expand_query(query):
    text=normalize(query); extras=[]
    for en,fa in ALIASES:
        if re.search(r'\b'+re.escape(en)+r'\b',text): extras.append(fa)
        if re.search(r'\b'+re.escape(fa)+r'\b',text): extras.append(en)
    return query+' '+' '.join(extras) if extras else query

def self_contained(query):
    if re.search(r'```|\b(?:print|def)\s*\(',query): return True
    if re.search(r'(?:translate|rewrite|rephrase|ترجمه|بازنویسی).{0,80}[:：]',query,re.I): return True
    if re.fullmatch(r'\s*[\d۰-۹٠-٩\s+*/().!%^−-]+\s*',query): return True
    from jarvis.agent.parser_v20 import get_parser
    return get_parser().parse(query) is not None

def content_signature(text):
    return re.sub(r'\W+',' ',normalize(text)).strip()
