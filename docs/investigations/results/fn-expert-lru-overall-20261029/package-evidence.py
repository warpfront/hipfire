from pathlib import Path
import json,gzip,hashlib,subprocess
F=Path(__file__).resolve().parent
P=Path('/home/husrcf/Code/hipfire-fn-expert-lru-beta-20261025')
D=P/'docs/investigations/results/fn-expert-lru-overall-20261029'
D.mkdir(exist_ok=False)
index=[]
def add(src,rel=None,compress=False):
    raw=src.read_bytes();dest=D/(rel or src.relative_to(F))
    if compress:dest=Path(str(dest)+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    public=gzip.compress(raw,mtime=0) if compress else raw
    dest.write_bytes(public)
    index.append({'artifact':str(dest.relative_to(D)),'original_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256(public).hexdigest(),'original_bytes':len(raw),'stored_bytes':len(public),'content_transformation':'gzip only' if compress else 'none'})
for src in sorted(F.iterdir()):
    if src.is_file() and (src.suffix in ('.json','.py') or src.suffix=='.log'):
        add(src,compress=src.suffix=='.log')
add(F/'runtime/MANIFEST.json')
for name in ['redline-candidate-trial01','serve-A1','serve-A1extra','serve-B1','serve-B2','serve-A2','serve-A3','serve-B3']:
    for src in sorted((F/name).iterdir()):
        if src.is_file() and src.suffix in ('.json','.jsonl','.log'):add(src,compress=src.suffix in ('.log','.jsonl'))
(D/'ARTIFACTS.json').write_text(json.dumps({'tested_commit':json.loads((F/'runtime/MANIFEST.json').read_text())['commit'],'source_tree_after_test':'Only this evidence directory is added after the tested commit; product/validation source remains identical.','path_policy':'Raw developer paths retained; generated public-repository fixtures only; no HOME configs or model bytes copied.','artifacts':index},indent=2)+'\n')
print(json.dumps({'files':len(index),'stored_bytes':sum(x['stored_bytes'] for x in index),'directory':str(D)}))
