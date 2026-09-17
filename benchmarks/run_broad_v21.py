from pathlib import Path
import sys,json,time,re,math,hashlib,signal,resource,ast,importlib.util
ROOT=Path(sys.argv[1]).resolve();CASES=Path(sys.argv[2]).resolve();OUT=Path(sys.argv[3]).resolve();sys.path.insert(0,str(ROOT))
from tests.helpers import TemporaryRuntime
# Same isolated AST evaluator is used for both versions, regardless of selected root.
spec=importlib.util.spec_from_file_location('frozen_code_evaluator',Path(__file__).resolve().parents[1]/'jarvis/agent/code_v21.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
DIGITS=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')
def normalized(s):return re.sub(r'[^\w]+','',s.translate(DIGITS).replace('ي','ی').replace('ك','ک').lower())
def nums(s):
 s=s.translate(DIGITS).replace('٫','.').replace('٬','')
 if '=' in s:s=s.rsplit('=',1)[-1]
 return [float(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',s)]
def score(row,text):
 kind=row['kind'];expected=row['expected']
 if kind=='number':
  values=nums(text);return bool(values) and math.isclose(values[-1],expected,rel_tol=1e-5,abs_tol=.0051)
 if kind=='parts':
  v=nums(text);return len(v)>=2 and all(math.isclose(a,b,rel_tol=1e-5,abs_tol=.0051) for a,b in zip(v[-2:],expected))
 if kind=='abstain':return bool(re.search('قابل تأیید نیست|ناممکن|غیرممکن|اطلاعات کافی|cannot verify|could not verify|impossible|insufficient',text,re.I))
 if kind=='translation':return any(normalized(ref)==normalized(text) or normalized(ref) in normalized(text) for ref in expected)
 if kind=='contains':return all(x.lower() in text.lower() for x in expected)
 if kind=='code':
  blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
  if not blocks:return False
  code=blocks[0]
  try:
   for c in row['checks']:
    got=mod.SafePython().run(code,json.loads(json.dumps(c['args'])))
    if got!=c['expected']:return False
   return True
  except (ValueError,TypeError,KeyError,ArithmeticError,StopIteration,IndexError,SyntaxError):return False
 return False
rows=[json.loads(x) for x in CASES.read_text().splitlines()];results=[]
def timeout(*args):raise TimeoutError('request exceeded 20 seconds')
signal.signal(signal.SIGALRM,timeout)
start=time.perf_counter()
with TemporaryRuntime() as rt:
 startup=(time.perf_counter()-start)*1000
 rt.memory.set_setting('memory_enabled',False);rt.memory.set_setting('internet_enabled',False)
 for i,row in enumerate(rows):
  called=[]
  def blocked(*args,**kwargs):called.append(str(args[0]) if args else 'tool');raise AssertionError('external tools disabled')
  rt.tools.invoke=blocked;t=time.perf_counter();output='';error='';intent='';passed=False
  try:
   signal.alarm(20);reply=rt.agent.respond(row['question']);output=reply.text;intent=reply.intent;passed=score(row,output) and not called
  except Exception as e:error=type(e).__name__+': '+str(e)
  finally:signal.alarm(0)
  results.append({**row,'output':output,'passed':bool(passed),'intent':intent,'error':error,'tools_called':called,'latency_ms':(time.perf_counter()-t)*1000})
  if (i+1)%25==0:print(i+1,len(rows),sum(r['passed'] for r in results),flush=True);OUT.with_suffix('.partial.json').write_text(json.dumps(results,ensure_ascii=False))
summary={'project':str(ROOT),'benchmark_sha256':hashlib.sha256(CASES.read_bytes()).hexdigest(),'total':len(rows),'correct':sum(r['passed'] for r in results),'results':results,'category':{},'startup_ms':startup,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'duration_seconds':time.perf_counter()-start,'tools_enabled':False,'memory_enabled':False,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'jarvis').rglob('*.py')},'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*v21.npz')}}
for category in sorted({r['category'] for r in rows}):
 selected=[r for r in results if r['category']==category];summary['category'][category]={'total':len(selected),'correct':sum(r['passed'] for r in selected)}
OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('COMPLETE',summary['correct'],summary['category'],flush=True)
