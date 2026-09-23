#!/usr/bin/env python3
"""Serial, pinned-binary KLD measurement launcher. No kernel/quantizer implementation."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path
ROOT=Path('/home/kaden/ClaudeCode/warpfront/wt-mq4e8')
OUT=ROOT/'scratch-2026-09-17/Mq4e8Study'
BIN=OUT/'eval_hipfire.weight'
EXPECTED='8b2d99f611fd201dda6c0199f3ae3eba'
def md5(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'md5').hexdigest()
def run(model,variant,corpus,chunks):
    ref=Path(f'/home/kaden/kldrefs/qwen3.8-27b.ref_{corpus}.bin')
    bm=md5(BIN);assert bm==EXPECTED,(str(BIN),bm)
    log=OUT/f'eval_{variant}_{corpus}_{chunks}.log';kldseq=OUT/f'eval_{variant}_{corpus}_{chunks}.kldseq'
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('HIPFIRE_'):del env[k]
    pins={'HOME':'/home/kaden/.hipfire-homes/ab0','ROCR_VISIBLE_DEVICES':'0','HIPFIRE_MODELS_DIR':'/home/kaden/.hipfire/models','HIPFIRE_KERNEL_CACHE':'/home/kaden/.hipfire-homes/ab0/.hipfire_kernels','HIPFIRE_IU4_PREFILL':'0','HIPFIRE_IU4_XMASTER':'0','HIPFIRE_GRAPH':'0'}
    env.update(pins)
    cmd=[str(BIN),'--model',str(model),'--ref',str(ref),'--output',str(kldseq),'--kv-mode','q8','--kv-v','q8','--scoring-mode','prefill','--max-chunks',str(chunks)]
    meta={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'variant':variant,'corpus':corpus,'chunks':chunks,'model':str(model),'model_md5':md5(model),'reference':str(ref),'reference_md5':md5(ref),'binary':str(BIN),'binary_md5':bm,'env':pins,'command':cmd,'tree':'mq4e8-study @ b7169e4a9, tree build before activation-only changes'}
    print('START',json.dumps(meta),flush=True)
    t=time.time()
    with log.open('w',buffering=1) as f:
        f.write('PROVENANCE '+json.dumps(meta)+'\n');f.flush()
        p=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        f.write(f'EXIT_CODE {p.returncode}\nWALL_SECONDS {time.time()-t:.6f}\n')
    text=log.read_text();hits=re.findall(r'slice-mean KLD = ([0-9.eE+-]+)',text)
    if p.returncode or len(hits)!=1:raise RuntimeError(f'eval failed/incomplete: {log}')
    result={**meta,'kld':float(hits[0]),'log':str(log),'kldseq':str(kldseq),'kldseq_md5':md5(kldseq),'wall_seconds':time.time()-t}
    with (OUT/'weight_eval_results.jsonl').open('a',buffering=1) as f:f.write(json.dumps(result)+'\n')
    print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--variant',required=True);ap.add_argument('--corpus',choices=['wt2','ag'],default='wt2');ap.add_argument('--chunks',type=int,nargs='+',default=[1,2,24]);a=ap.parse_args()
    for n in a.chunks:run(a.model,a.variant,a.corpus,n)
