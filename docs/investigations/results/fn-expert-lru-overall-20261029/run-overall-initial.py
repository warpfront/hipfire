"""Fixed developer-only validation: ABBAAB fresh servers, identical warmup cells.
Defer only successful warmup cleanup. Keep startup retry cleanup and all gates.
"""
from pathlib import Path
import os,sys,json,time,threading,subprocess,importlib.util,traceback,hashlib
ROOT=Path('/home/husrcf/Code/hipfire-fn-expert-lru-beta-20261025')
OUT=Path(__file__).resolve().parent
MODEL=Path('/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts')
ORDER=[('A1','0'),('B1','1'),('B2','1'),('A2','0'),('A3','0'),('B3','1')]
COMMON={'HIPFIRE_CLI_BIN':str(OUT/'runtime/hipfire'),'HIPFIRE_DAEMON_BIN':str(OUT/'runtime/daemon_lruq'),'HIPFIRE_MODELS_DIR':str(MODEL.parent),'HIPFIRE_MTP_MODE':'off','HIPFIRE_OFFLOAD_EXEC':'pcie','HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT':'50','HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT':'64','HIPFIRE_QWEN4_DECODE_SHARE':'off','HIPFIRE_QWEN4_D2_DEFAULT':'0','HIPFIRE_PLE_FORMAT':'q4','HIPFIRE_PREFILL_CHUNK_ROWS':'4096','HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB':'20000','HIPFIRE_DN_STATE_EF':'1','HIPFIRE_KERNEL_CACHE':str(OUT.parent/'shared-kernel-cache')}

def worker(name,cache):
    case=OUT/('serve-'+name);case.mkdir(exist_ok=False)
    for k in list(os.environ):
        if k.startswith('HIPFIRE_') or k in ('ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES'):del os.environ[k]
    explicit=dict(COMMON,HIPFIRE_QWEN4_EXPERT_CACHE=cache,HIPFIRE_HOME=str(case/'parent-home'))
    os.environ.update(explicit)
    spec=importlib.util.spec_from_file_location('canonical_serve_harness',ROOT/'scripts/serve_harness.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    orig_spawn,orig_kill,orig_run=m.spawn_serve,m._kill_serve,m.run
    launch_offset=None;observed=[]
    def keep_after_success(cfg,home,log):
        nonlocal launch_offset
        offset=orig_spawn(cfg,home,log)
        if offset is not None:
            launch_offset=offset;m._kill_serve=lambda:None
        return offset
    def checked_run(cfg,args):
        assert launch_offset is not None
        text=m._serve_log_text(args.serve_log,launch_offset)
        m.apply_observed_kv_backend(cfg,text)
        m._assert_serve_path_proofs(cfg,args.serve_log,offset=launch_offset)
        phase_offset=Path(args.serve_log).stat().st_size
        cfg['_serve_log_offset']=phase_offset
        rows=orig_run(cfg,args)
        m._assert_dflash_request_proofs(cfg,rows,args.serve_log,offset=phase_offset)
        observed.append({'phase':Path(args.out).stem,'kv_backend':cfg.get('kv_backend_effective'),'kv_backend_legacy':cfg.get('kv_backend_legacy'),'row_backend_fields':[{k:r.get(k) for k in ('kv_mode','kv_backend','kv_backend_legacy','kv_backend_reason')} for r in rows]})
        return rows
    m.spawn_serve=keep_after_success;m.run=checked_run
    prompts=[{'genre':'dirty-cache-before-staged-prefill','prompt':(ROOT/'benchmarks/prompts/fiction_lighthouse.txt').read_text()}, {'genre':'staged-prefill','prompt':(ROOT/'benchmarks/prompts/cohere2moe_long.txt').read_text()}]
    (case/'staged-prompts.json').write_text(json.dumps(prompts,ensure_ascii=False)+'\n')
    phases=[('warmup','battery',16,[]),('battery','battery',512,[])]
    if name in ('A1','B1'):
        phases += [('chain','chain',512,[]),('session','session',512,[]),('staged','battery',128,['--prompts-file',str(case/'staged-prompts.json')]),('longdecode','battery',1024,['--prompt-file',str(ROOT/'benchmarks/prompts/online_tune/code_parser_rust.txt')])]
    receipt={'frozen_manifest':json.loads((OUT/'runtime/MANIFEST.json').read_text()),'name':name,'cache':cache,'env':explicit,'phase_order':[x[0] for x in phases],'phase_results':[],'memory_sampling_interval_seconds':0.1,'memory_sampling_claim':'observed sampled peak, not exact allocator peak','staged_prompt_md5s':[hashlib.md5(x['prompt'].encode()).hexdigest() for x in prompts]}
    stop=threading.Event();peaks={}
    def sample():
        with (case/'memory.jsonl').open('w') as out:
            while not stop.is_set():
                row={'time_monotonic':time.monotonic()}
                for kind in ('vram','gtt'):
                    try:row[kind+'_used_bytes']=int(Path('/sys/class/drm/card1/device/mem_info_'+kind+'_used').read_text())
                    except OSError:pass
                for k,v in row.items():
                    if k.endswith('_used_bytes'):peaks[k]=max(peaks.get(k,0),v)
                out.write(json.dumps(row)+'\n');out.flush();stop.wait(.1)
    thread=threading.Thread(target=sample,daemon=True);thread.start();start=time.monotonic();error=None
    try:
        for i,(phase,mode,cap,extra) in enumerate(phases):
            args=['scripts/serve_harness.py','--model',str(MODEL),'--tag','qwen3.8:flash-next','--registry',str(ROOT/'registry/models.json'),'--speculation','off','--thinking','off','--max-tokens',str(cap),'--max-seq','16384','--sampling','greedy','--seed','123','--mode',mode,'--port','11588','--home',str(case/'home'),'--serve-warm-timeout-secs','1800','--serve-log',str(case/'serve.log'),'--out',str(case/(phase+'.json')),'--replay-route-proof-log']+extra
            if i:args+=['--no-spawn']
            if name=='B1' and phase!='warmup':args+=['--compare-transcript',str(OUT/'serve-A1'/(phase+'.json'))]
            t=time.monotonic();sys.argv=args
            print(json.dumps({'case':name,'phase':phase,'event':'start'}),flush=True)
            with (case/(phase+'.log')).open('w',buffering=1) as log:
                prevout,preverr=sys.stdout,sys.stderr;sys.stdout=sys.stderr=log
                try:m.main()
                finally:sys.stdout,sys.stderr=prevout,preverr
            m._kill_serve=orig_kill
            rows=json.loads((case/(phase+'.json')).read_text())
            for idx,r in enumerate(rows):
                assert not r.get('attractor') and not r.get('empty'),(name,phase,idx,'bad text')
                assert r.get('finish') in ('stop','length','eos','tool_calls'),(name,phase,idx,r.get('finish'))
                assert not r.get('stream_error'),(name,phase,idx,'stream error',r.get('stream_error'))
                assert r.get('terminal_count',1)==1,(name,phase,idx,'terminal count')
            result={'phase':phase,'arguments':args,'elapsed_seconds':time.monotonic()-t,'rows':len(rows),'status':'PASS'}
            receipt['phase_results'].append(result)
            (case/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({'case':name,'phase':phase,'event':'pass','rows':len(rows),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as e:
        error=repr(e);receipt['failure']=error;(case/'driver-error.log').write_text(traceback.format_exc());raise
    finally:
        m._kill_serve=orig_kill;orig_kill();stop.set();thread.join()
        receipt.update(elapsed_seconds=time.monotonic()-start,observed_sampled_peaks=peaks,observed_backend_receipts=observed,status='FAIL' if error else 'PASS')
        (case/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[1],sys.argv[2])
    else:
        results=[]
        for name,cache in ORDER:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),name,cache],cwd=ROOT)
            results.append({'case':name,'cache':cache,'exit_code':r.returncode})
            (OUT/'overall-matrix.json').write_text(json.dumps(results,indent=2)+'\n')
            if r.returncode:raise SystemExit(r.returncode)
