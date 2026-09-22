#!/usr/bin/env python3
import json, os, re, subprocess, threading, time, urllib.request
from pathlib import Path
root=Path('/home/kaden/hipfire-gfx11slice/scratch-gfx11slice')
target=int(os.environ.get('TARGET_ROWS','8192'))
prefix=os.environ.get('OUTPUT_PREFIX','triple')
port=int(os.environ.get('PROBE_PORT','11741'))
prompt=Path('/home/kaden/hipfire-gfx11stack-profile/benchmarks/prompts/pp8192.txt').read_text()+' red'*(target-8192)
assert prompt.startswith('The quick brown fox')
variants=['A swift orange fox'+prompt[len('The quick brown fox'):], prompt, 'A swift orange fox'+prompt[len('The quick brown fox'):]]
url=f'http://127.0.0.1:{port}/v1/chat/completions'
for index,text in enumerate(variants,1):
    body={'model':'/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq',
          'messages':[{'role':'user','content':text}], 'max_tokens':1,'temperature':0,
          'top_p':1,'top_k':1,'reasoning_effort':'none','max_think_tokens':1,
          'stream':True,'stream_options':{'include_usage':True}}
    done=threading.Event(); result={}
    def send():
        try:
            req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
            chunks=[]
            with urllib.request.urlopen(req,timeout=1200) as response:
                for line in response:
                    if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
                        chunks.append(json.loads(line[6:]))
            result['usage']=next((c['usage'] for c in reversed(chunks) if c.get('usage')),None)
            result['timings']=next((c['timings'] for c in chunks if c.get('timings')),None)
        except Exception as exc: result['error']=repr(exc)
        finally: done.set()
    start=time.monotonic_ns(); worker=threading.Thread(target=send); worker.start()
    readings=[]
    while not done.is_set():
        smi=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True,check=True)
        match=re.search(r'GPU\[0\]\s+: VRAM Total Used Memory \(B\): (\d+)',smi.stdout)
        if match: readings.append(int(match.group(1)))
        done.wait(.05)
    worker.join(); end=time.monotonic_ns()
    usage=result.get('usage')
    if usage and (usage['prompt_tokens']!=target or usage.get('prompt_tokens_details',{}).get('cached_tokens',0)!=0):
        result['error']=f'request {index} not cold pp{target}: {usage}'
    record={'request':index,'start_ns':start,'end_ns':end,'wall_ms':(end-start)/1e6,
            'vram_samples':len(readings),'vram_first_bytes':readings[0],
            'vram_peak_observed_bytes':max(readings),'vram_last_bytes':readings[-1],
            'usage':usage,'timings':result.get('timings'),'error':result.get('error')}
    (root/f'{prefix}-{index}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if result.get('error'): raise RuntimeError(result['error'])
