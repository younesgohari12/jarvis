"""Explicit episodic topic recall on existing versioned SQLite memory tables.
No ungrounded inference of hardware facts or automatic side effects from recall.
"""
import re,json
class MemoryContextV21:
    def __init__(self,store):self.store=store
    def observe(self,text,session_id):
        clean=' '.join(text.replace('\u200c',' ').split())
        if len(clean)>1000:return
        patterns=[r'^من (?:با|روی) ([\w +.-]{2,60}) (?:کار می کنم|کار میکنم|کار می کنم\.|کار می کنم؟)$',r'^I (?:work with|am working on) ([\w +.-]{2,60})\.?$']
        topic=None
        for pattern in patterns:
            m=re.match(pattern,clean,re.I)
            if m:topic=m.group(1).rstrip('.');break
        if topic:
            previous=self.store.semantic_memories().get('v21_project_topics','[]')
            try:topics=json.loads(previous)
            except ValueError:topics=[]
            topics=[x for x in topics if x.casefold()!=topic.casefold()]+[topic]
            self.store.remember_semantic('v21_project_topics',json.dumps(topics[-4:],ensure_ascii=False),1.0,'explicit_user_statement')
            self.store.record_episode(session_id,clean,.85)
        # Store user-asserted semantic relations with provenance, not as verified external facts.
        m=re.fullmatch(r'([\w +.-]{2,50}) (?:یک|نوعی) ([\w +.-]{2,70}) است[.؟]?',clean)
        if m:self.store.remember_semantic('v21_user_isa:'+m.group(1).casefold(),m.group(2),.7,'user_assertion')
    def recall(self,text,language='fa'):
        match=re.fullmatch(r'([\w +.-]{2,50})\s+(?:چیست|چیه)[؟?]?',text.strip())
        if match:
            subject=match.group(1).strip();value=self.store.semantic_memories().get('v21_user_isa:'+subject.casefold())
            if value:return f'طبق توضیح قبلی خودت، {subject} یک {value} است.'
        if not re.search(r'پروژه جدیدم|پروژه بعدی|new project|next project',text,re.I):return None
        raw=self.store.semantic_memories().get('v21_project_topics')
        if not raw:return None
        try:topics=json.loads(raw)
        except ValueError:return None
        if len(topics)==1:
            return f'قبلاً گفتی با {topics[0]} کار می‌کنی. پروژهٔ جدیدت هم در همین زمینه است؟ هدفت را بگو تا ادامه بدهیم.' if language=='fa' else f'You previously said you work with {topics[0]}. Is the new project in that area too? What do you want to build?'
        if topics:return ('برای پروژهٔ جدید منظورت کدام زمینه است: ' if language=='fa' else 'Which previous area do you mean for the new project: ')+'، '.join(topics)+'؟'
        return None
