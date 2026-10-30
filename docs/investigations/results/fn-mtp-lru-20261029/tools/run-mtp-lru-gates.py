#!/usr/bin/env python3
"""Explicit no-retry runner; default dry plan. Caller owns the shared GPU lock."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verify_mtp_lru',HERE/'verify-mtp-lru.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',required=True,type=Path);p.add_argument('--publication-oracle',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--stage',choices=['no-model','models'],default='no-model')
    p.add_argument('--model',type=Path);p.add_argument('--prompt-files',nargs='+',type=Path)
    p.add_argument('--tokens',type=int,default=512);p.add_argument('--k',type=int,default=3);p.add_argument('--max-seq',type=int,default=16384)
    p.add_argument('--backend',choices=['hip','auto'],default='hip');p.add_argument('--timeout',type=int,default=1750)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    binaries={'oracle':{'path':str(a.oracle.resolve()),'sha256':v.sha(a.oracle)},'publication':{'path':str(a.publication_oracle.resolve()),'sha256':v.sha(a.publication_oracle)}}
    commands=[('cpu',[str(a.oracle),'--cpu-controls']),('collector',[str(a.oracle),'--execute','--phase','collector']),('publication',[str(a.publication_oracle),'--execute'])]
    if a.stage=='models':
        v.need(a.model and a.prompt_files,'models needs --model and --prompt-files')
        commands=[(label,[str(a.oracle),'--execute','--phase','model','--model',str(a.model),'--prompt-files',','.join(map(str,a.prompt_files)),'--tokens',str(a.tokens),'--k',str(a.k),'--max-seq',str(a.max_seq)]) for label in ['cache-off','cache-on']]
    common=os.environ.copy();common.update({'HIPFIRE_REPLAY_BACKEND':a.backend,'HIPFIRE_MTP_MODE':'on','HIPFIRE_MTP_NGRAM_COPY':'0','HIPFIRE_MTP_TRACE':'1','HIPFIRE_MTP_INCREMENTAL':'0','HIPFIRE_OFFLOAD_EXEC':'pcie','HIPFIRE_QWEN4_D2_DEFAULT':'0','HIPFIRE_QWEN4_DECODE_SHARE':'off','HIPFIRE_STATE_QUANT':'q8'})
    env_keys=['HIPFIRE_REPLAY_BACKEND','HIPFIRE_MTP_MODE','HIPFIRE_MTP_NGRAM_COPY','HIPFIRE_MTP_TRACE','HIPFIRE_MTP_INCREMENTAL','HIPFIRE_OFFLOAD_EXEC','HIPFIRE_QWEN4_D2_DEFAULT','HIPFIRE_QWEN4_DECODE_SHARE','HIPFIRE_STATE_QUANT','HIPFIRE_KV_MODE','HIPFIRE_PLE_FORMAT','HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB','HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT','HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT','HIPFIRE_KERNEL_CACHE','HIPFIRE_RAILGUN_CHECK']
    plan={'environment_snapshot':{k:common.get(k) for k in env_keys},'stage':a.stage,'binaries':binaries,'commands':commands,'execute':a.execute,'gpu_lock':'caller-held project lock; runner never retries a model','performance_claim':False}
    if not a.execute:print(json.dumps(plan,indent=2));return 0
    a.out.mkdir(parents=True,exist_ok=True)
    runfile=a.out/'RUN.json'
    if a.stage=='no-model':
        with open(runfile,'x') as f:json.dump({'schema':'mtp-lru-gate-run-v1','started_unix':time.time(),'binaries':binaries,'backend':a.backend,'environment_snapshot':plan['environment_snapshot'],'commands':commands,'performance_claim':False},f,indent=2);f.write('\n')
    else:
        old=v.load(runfile);v.need(old['binaries']==binaries,'binary changed after no-model gates; use a new run directory and rerun small gates')
        for name,check in [('cpu',v.verify_cpu),('collector',v.verify_collector),('publication',v.verify_publication)]:check(v.load(a.out/(name+'.json')))
        with open(a.out/'MODEL-PLAN.json','x') as f:json.dump(plan,f,indent=2);f.write('\n')
    for name,command in commands:
        v.need(v.sha(a.oracle)==binaries['oracle']['sha256'] and v.sha(a.publication_oracle)==binaries['publication']['sha256'],'frozen binary changed before phase '+name)
        env=common.copy()
        if name in ['cache-on','cache-off']:env['HIPFIRE_QWEN4_EXPERT_CACHE']='1' if name=='cache-on' else '0'
        raw=a.out/(name+'.stdout.log');err=a.out/(name+'.stderr.log');report=a.out/(name+'.json')
        v.need(not raw.exists() and not err.exists() and not report.exists(),'refusing to overwrite previous attempt '+name)
        started=time.time()
        try:
            with open(raw,'x') as out,open(err,'x') as errors:
                cp=subprocess.run(command,env=env,stdout=out,stderr=errors,timeout=a.timeout,check=False)
            lines=[x for x in raw.read_text().splitlines() if x.startswith('{')]
            v.need(len(lines)==1,'expected exactly one top-level report in '+name)
            data=json.loads(lines[0]);report.write_text(json.dumps(data,indent=2)+'\n')
            v.need(cp.returncode==0,'nonzero '+name+'; raw failure evidence retained')
            {'cpu':v.verify_cpu,'collector':v.verify_collector,'publication':v.verify_publication,'cache-on':lambda r:v.verify_model(r,True),'cache-off':lambda r:v.verify_model(r,False)}[name](data)
            receipt={'status':'PASS','name':name,'elapsed_seconds':time.time()-started,'returncode':cp.returncode,'stdout_sha256':v.sha(raw),'stderr_sha256':v.sha(err),'report_sha256':v.sha(report),'oracle_sha256':binaries['publication' if name=='publication' else 'oracle']['sha256']}
        except Exception as e:
            receipt={'status':'FAIL','name':name,'elapsed_seconds':time.time()-started,'error':repr(e),'retry':False}
            with open(a.out/(name+'.run.json'),'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
            print(json.dumps(receipt));return 1
        with open(a.out/(name+'.run.json'),'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    result=v.verify_directory(a.out,partial=a.stage=='no-model')
    with open(a.out/('NO-MODEL-VERIFIED.json' if a.stage=='no-model' else 'VERIFIED.json'),'x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
