"""Package a tested release, then verify CRC and every archived file hash."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve();OUT.mkdir(parents=True,exist_ok=True)
ZIP=OUT/'Jarvis_v0.11.0_Intelligence_v21_ARCHITECTURE_TRAINED_TESTED.zip'
exclude={'__pycache__','.pytest_cache','.git','.venv','node_modules'}
def included(p):
 return p.is_file() and not any(x in exclude for x in p.relative_to(ROOT).parts) and not p.name.endswith(('.pyc','.partial.json'))
def sha(data):return hashlib.sha256(data).hexdigest()
files=sorted(p for p in ROOT.rglob('*') if included(p) and p.name!='SHA256SUMS.json')
manifest={'algorithm':'sha256','scope':'all packaged files except this manifest','release':'v21 architecture trained tested with limitations','files':{p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files}}
mp=ROOT/'SHA256SUMS.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files+[mp]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None,'CRC failure'
 for rel,digest in manifest['files'].items():assert sha(z.read(ROOT.name+'/'+rel))==digest,rel
 assert len(z.namelist())==len(files)+1
result={'archive':ZIP.name,'size_bytes':ZIP.stat().st_size,'sha256':sha(ZIP.read_bytes()),'file_count':len(files)+1,'crc_verified':True,'all_manifest_hashes_verified':True,'release_gate':json.loads((ROOT/'RELEASE_V21.json').read_text())['label']}
(OUT/'Jarvis_v21_Archive_Check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
