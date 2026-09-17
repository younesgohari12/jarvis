"""Same frozen input, scoring and tool policy for any selected project root."""
from pathlib import Path
import sys,json,re,math,time,hashlib,signal,ast,subprocess,os,resource
ROOT=Path(sys.argv[1]).resolve();CASES=Path(sys.argv[2]).resolve();OUT=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(ROOT))
from tests.helpers import TemporaryRuntime
from jarvis.agent.deliberation import QueryAnalyzer
DIGITS=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')

def nums(text):
 t=text.translate(DIGITS).replace('٫','.').replace('٬','')
 t=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',t)
 if '=' in t:t=t.rsplit('=',1)[-1]
 return [float(x) for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',t)]

def code_behavior(text,row):
 blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
 source=blocks[0] if blocks else text
 try:tree=ast.parse(source)
 except SyntaxError:return False
 defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
 if not defs:return False
 fn=defs[0].name
 allowed={'sum','len','max','min','sorted','range','isinstance','int','float','str','abs','list','set','dict','ValueError','enumerate','zip','all','any',fn}
 allowed_attrs={'append','casefold','lower','isalnum','isalpha','strip','replace','join','values','items','get'}
 for node in ast.walk(tree):
  if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)):return False
  if isinstance(node,ast.Name) and node.id.startswith('__'):return False
  if isinstance(node,ast.Attribute) and node.attr not in allowed_attrs:return False
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id not in allowed:return False
 q=row['question'].casefold()
 if 'معکوس' in q or 'revers' in q: checks=[(['abCD'],'DCba'),([''],'')]
 elif 'پالیندروم' in q or 'palindrome' in q: checks=[(['level'],True),(['table'],False)]
 elif 'میانگین' in q or 'average' in q: checks=[([[2,3,7]],4),([[-4,2]],-1)]
 elif 'تکراری' in q:checks=[([[3,1,3,2]],[3,1,2])]
 elif 'positive' in q:checks=[([[-3,0,4,9]],[4,9])]
 elif 'largest' in q:checks=[([[-3,-8,-1]],-1)]
 else:checks=[([[2,-3,7]],6)]
 test='\n'.join(f'assert {fn}(*{args!r}) == {expected!r}' for args,expected in checks)
 try:
  p=subprocess.run([sys.executable,'-I','-S','-c',source+'\n'+test],capture_output=True,timeout=2,text=True)
  return p.returncode==0
 except (OSError,subprocess.TimeoutExpired):return False

def score(row,text):
 kind=row['kind'];expected=row['expected']
 if kind=='number':
  values=nums(text)
  return bool(values) and math.isclose(values[-1],expected,rel_tol=1e-4,abs_tol=.0051)
 if kind=='parts':
  values=nums(text)
  return len(values)>=2 and all(math.isclose(x,y,rel_tol=1e-4,abs_tol=.0051) for x,y in zip(values[-2:],expected))
 if kind=='contains':
  present=all(x.casefold() in text.casefold() for x in expected)
  return present and (code_behavior(text,row) if row['category']=='code_generation' else True)
 if kind=='constraints':
  sentences=[s for s in re.split(r'[.!?؟]+',text) if s.strip()]
  return len(sentences)==expected['sentences'] and expected['required'] in text and (bool(re.search('[آ-ی]',text)) if expected['language']=='fa' else not bool(re.search('[آ-ی]',text)))
 return False

rows=[json.loads(l) for l in CASES.read_text().splitlines()];results=[]
started=time.perf_counter();cold_start=time.perf_counter()
with TemporaryRuntime() as rt:
 startup_ms=(time.perf_counter()-cold_start)*1000
 rt.memory.set_setting('memory_enabled',False);rt.memory.set_setting('internet_enabled',False)
 for i,row in enumerate(rows):
  called=[]
  def blocked(tool,*args,**kwargs):called.append(str(tool));raise AssertionError('tools disabled for fresh benchmark')
  rt.tools.invoke=blocked
  tick=time.perf_counter();output='';passed=False;error='';intent='';data={}
  try:
   if row['kind']=='route':
    route=rt.agent.router.route(row['question'],{});output=route.intent;passed=output==row['expected'];intent=output
   elif row['kind']=='freshness':
    a=QueryAnalyzer().analyze(row['question']);observed=bool(a.needs_fresh_information or a.volatile)
    output=str(observed);passed=observed==row['expected'];intent='freshness_decision'
   else:
    reply=rt.agent.respond(row['question']);output=reply.text;intent=reply.intent;data=reply.data;passed=score(row,output) and not called
  except Exception as e:error=type(e).__name__+': '+str(e)
  elapsed=(time.perf_counter()-tick)*1000
  trace=getattr(rt.agent.reasoning.local_intelligence,'last_trace',None)
  result={**row,'passed':bool(passed),'output':output,'intent':intent,'latency_ms':elapsed,'tools_called':called,'error':error,'v20_ir':bool(trace),'reasoning_source':data.get('reasoning_source')}
  results.append(result)
  if (i+1)%25==0: print(i+1,'/',len(rows),'correct',sum(r['passed'] for r in results),flush=True)
  OUT.with_suffix('.partial.json').write_text(json.dumps(results,ensure_ascii=False))
summary={'project':str(ROOT),'benchmark_sha256':hashlib.sha256(CASES.read_bytes()).hexdigest(),'total':len(rows),'correct':sum(r['passed'] for r in results),'accuracy':sum(r['passed'] for r in results)/len(rows),'startup_ms':startup_ms,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'duration_s':time.perf_counter()-started,'web_tools_enabled':False,'memory_enabled':False,'category':{},'results':results}
for cat in sorted({r['category'] for r in results}):
 part=[r for r in results if r['category']==cat];summary['category'][cat]={'n':len(part),'correct':sum(r['passed'] for r in part)}
OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('COMPLETE',summary['correct'],summary['total'],flush=True)
