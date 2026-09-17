from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jarvis.neural.transformer import JarvisTransformer
from jarvis.neural.tokenizer import JarvisTokenizer
from jarvis.neural.language_adapter_v21 import attach_adapter
from jarvis.neural.conversation import NeuralConversationEngine
base_path=ROOT/'models/jarvis_nano_v18.npz';tok=JarvisTokenizer.load(ROOT/'models/jarvis_tokenizer_v003.json')
base=JarvisTransformer.load(base_path);adapted=attach_adapter(JarvisTransformer.load(base_path),ROOT/'models/language_adapter_v21.npz',base_path)
questions=['سلام، خودت را معرفی کن.','برای یادگیری پایتون از کجا شروع کنم؟','فرق متغیر و تابع را توضیح بده.','Write a function that checks whether a number is prime.','Please explain why percentages compound.','چرا جواب بدون شواهد ممکن است اشتباه باشد؟']
rows=[]
for q in questions:
 prompt=[tok.bos_id,tok.special_to_id['<user>'],*tok.encode(q),tok.special_to_id['<assistant>']]
 replies=[]
 for model in [base,adapted]:
  ids=model.sample(prompt,max_new_tokens=40,temperature=0,eos_id=tok.eos_id,seed=21)
  text=tok.decode(ids);quality=NeuralConversationEngine._quality(text,q)
  replies.append({'text':text,'heuristic_quality_gate':quality[0],'gate_reason':quality[1]})
 rows.append({'question':q,'base':replies[0],'adapter':replies[1],'changed':replies[0]['text']!=replies[1]['text']})
report={'mode':'direct deterministic neural generation; bypasses retrieval only for ablation','examples':rows,'adapter_calls':adapted.language_adapter_v21.calls,'changed_outputs':sum(r['changed'] for r in rows),'meaning':'Output changes are causal evidence, not proof of response correctness. Production retains existing quality and routing gates.'}
(ROOT/'reports/v21/language_generation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('changed',report['changed_outputs'],'calls',report['adapter_calls'])
