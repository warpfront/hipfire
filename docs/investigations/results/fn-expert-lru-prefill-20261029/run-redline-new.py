from pathlib import Path
import os,sys,subprocess,time,json,threading,hashlib
repo=Path('/home/husrcf/Code/hipfire-fn-lru-prefill-20261029')
a=Path('/home/husrcf/Code/hipfireQC/tmpOutput/expert-lru-beta-20261025/prefill-restore-20261029')
case=a/'redline-new-trial01';case.mkdir(exist_ok=False)
home=case/'home';home.mkdir(exist_ok=True)
(home/'config.toml').write_text('[speculation]\nmode = "off"\nmtp = "off"\ndflash = "off"\nngram = "off"\n\n[memory]\noffload_exec = "pcie"\nple_format = "q4"\n')
env=dict(os.environ,HIPFIRE_HOME=str(home),HIPFIRE_MTP_MODE='off',HIPFIRE_QWEN4_EXPERT_CACHE='1',HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT='50',HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT='64',HIPFIRE_QWEN4_DECODE_SHARE='off',HIPFIRE_QWEN4_D2_DEFAULT='0',HIPFIRE_PLE_FORMAT='q4',HIPFIRE_PREFILL_CHUNK_ROWS='4096',HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB='20000',HIPFIRE_DN_STATE_EF='1',HIPFIRE_OFFLOAD_EXEC='pcie',HIPFIRE_KERNEL_CACHE='/home/husrcf/Code/hipfireQC/tmpOutput/expert-lru-beta-20261025/update-f85ea28/shared-kernel-cache')
cmd=['/usr/bin/python3','scripts/redline_daemon_harness.py','--model','/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts','--daemon',str(a/'runtime-new'/'daemon_lru_new'),'--qwen4','--pm4','--skip-prefill','--decode-context','512','--kv-mode','fp8','--max-seq','16384','--capture-repeats','3','--measure-repeats','3','--decode-iterations','16','--shadow-iterations','16','--timeout','1800','--out',str(case/'report.json'),'--log',str(case/'daemon.log')]
receipt={'runtime_manifest':json.loads((a/'runtime-new/MANIFEST.json').read_text()),'base':'6c99bef7f8fcb966c4104aab2583e894f20c9d10','command':cmd,'env':{k:env[k] for k in env if k.startswith('HIPFIRE_')},'model_sha256_from_verified_hotset':'8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972','memory_sampling_interval_seconds':0.1,'memory_sampling_claim':'observed sampled peak, not exact allocator peak'}
(case/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
stop=threading.Event();peaks={}
def sample():
 with (case/'memory.jsonl').open('w') as out:
  while not stop.is_set():
   row={'time_monotonic':time.monotonic()}
   for name in ['vram','gtt']:
    try:row[name+'_used_bytes']=int(Path('/sys/class/drm/card1/device/mem_info_'+name+'_used').read_text())
    except OSError:pass
   try:
    row['mem_available_bytes']=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
   except (OSError,StopIteration):pass
   for k,v in row.items():
    if k.endswith('_used_bytes'):peaks[k]=max(peaks.get(k,0),v)
   out.write(json.dumps(row)+'\n');out.flush();stop.wait(0.1)
thread=threading.Thread(target=sample,daemon=True);thread.start();start=time.monotonic()
try:
 with (case/'harness.log').open('w') as out:r=subprocess.run(cmd,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT)
finally:stop.set();thread.join()
receipt.update(exit_code=r.returncode,elapsed_seconds=time.monotonic()-start,observed_sampled_peaks=peaks)
(case/'run.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(r.returncode)
