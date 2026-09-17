from pathlib import Path
import sys,time,json,statistics,resource,os,platform
ROOT=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();sys.path.insert(0,str(ROOT))
from tests.helpers import TemporaryRuntime
prompts=['525 را با نسبت 4 به 3 تقسیم کن','500 را 20 درصد کم کن و بعد 30 اضافه کن','احتمال موفقیت 30 درصد است؛ در 4 بار دقیقاً 2 موفقیت؟','دنباله هندسی از 4 با نسبت 3 داریم. جمله 6 چیست؟','7!']
starts=[];latencies=[];rss=[]
for repetition in range(3):
 start=time.perf_counter()
 with TemporaryRuntime() as rt:
  starts.append((time.perf_counter()-start)*1000)
  rt.memory.set_setting('memory_enabled',False);rt.memory.set_setting('internet_enabled',False)
  def blocked(*a,**k):raise AssertionError('tools disabled')
  rt.tools.invoke=blocked
  for q in prompts:rt.agent.respond(q)
  for _ in range(8):
   for q in prompts:
    now=time.perf_counter();rt.agent.respond(q);latencies.append((time.perf_counter()-now)*1000)
  rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
latencies.sort()
result={'startup_median_ms':statistics.median(starts),'startup_trials_ms':starts,'latency_p50_ms':statistics.median(latencies),'latency_p95_ms':latencies[int(.95*(len(latencies)-1))],'latency_max_ms':max(latencies),'requests':len(latencies),'peak_rss_mb':max(rss),'cpu_only':True,'gpu_tested':False,'python':platform.python_version(),'platform':platform.platform(),'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),'prompt_count':len(prompts),'notes':'Same established local prompts; no external tools; three in-process runtime initializations. Import cold-start excluded and measured separately by the fresh harness.'}
OUT.write_text(json.dumps(result,indent=2));print(result)
