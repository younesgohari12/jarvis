"""Span-stable numeric normalization for the semantic pipeline (never for code)."""
from __future__ import annotations
import re
import unicodedata

DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يك', '01234567890123456789یک')
FA = dict(zip('صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده'.split(), range(20)))
FA.update(dict(zip('بیست سی چهل پنجاه شصت هفتاد هشتاد نود صد دویست سیصد چهارصد پانصد ششصد هفتصد هشتصد نهصد'.split(), [20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900])))
EN = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
EN.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10))))
WORDS = {**FA, **EN, 'هیچ':0,'یکی':1,'none':0,'نیم':.5,'نصف':.5,'half':.5,'quarter':.25}
ORDINALS = dict(zip('اول نخست دوم سوم چهارم پنجم ششم هفتم هشتم نهم دهم'.split(),[1,1,2,3,4,5,6,7,8,9,10]))
ORDINALS.update(dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth'.split(),range(1,11))))
NUMBER = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])')

def normalize(text: str) -> str:
    t = unicodedata.normalize('NFKC', text).translate(DIGITS).casefold()
    t = t.replace('٫','.').replace('٪','%').replace('\u200c',' ').replace('ـ','')
    t = re.sub('[\u064b-\u065f]', '', t)
    t = re.sub(r'(?<=\d)٬(?=\d)', '', t)
    t = re.sub(r'\bیک\s+چهارم\b', '0.25', t)
    t = re.sub(r'\b(twice|double)\b','multiply by 2',t)
    t = re.sub(r'\btriple\b','multiply by 3',t)
    t = re.sub(r'(?<=[a-z])-(?=[a-z])|(?<=[آ-ی])-(?=[آ-ی])', ' ', t)
    tokens = re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+|[^\s]',t,re.UNICODE)
    out=[]; i=0
    while i<len(tokens):
        word=tokens[i]
        unit_second=(word=='second' and i>0 and (tokens[i-1] in ('per','a','each','every') or tokens[i-1] in WORDS or tokens[i-1].replace('.','',1).isdigit()))
        if word in ORDINALS and not unit_second:
            out.append(str(ORDINALS[word])); i+=1; continue
        if word not in WORDS:
            out.append(word); i+=1; continue
        value=WORDS[word]; i+=1
        # Compound numbers: hundred/thousand multiplication, tens+ones and Persian و.
        while i<len(tokens):
            w=tokens[i]
            if w in ('hundred','thousand','هزار'):
                value*=100 if w=='hundred' else 1000; i+=1
            elif w in ('and','و') and i+1<len(tokens) and tokens[i+1] in WORDS and value>=20:
                value+=WORDS[tokens[i+1]]; i+=2
            elif w in EN and value>=20 and EN[w]<10:
                value+=EN[w]; i+=1
            else: break
        out.append(f'{value:g}')
    t=' '.join(out)
    # Preserve decimals, signs, ordinals and percentages after token reconstruction.
    t=re.sub(r'(?<!\w)-\s+(?=\d)', '-', t)
    t=re.sub(r'(\d)\s+(st|nd|rd|th)\b',r'\1',t)
    return re.sub(r'\s+',' ',t).strip()

def skeleton(text: str) -> str:
    return NUMBER.sub('<num>', normalize(text))
