from pathlib import Path
import os, subprocess, json, datetime
repo=Path('/home/husrcf/Code/hipfire-fn-expert-lru-beta-20261025')
art=Path('/home/husrcf/Code/hipfireQC/tmpOutput/expert-lru-beta-20261025')
env=dict(os.environ,PATH='/home/husrcf/.cargo/bin:/usr/local/bin:/usr/bin:/bin',CARGO_TARGET_DIR='/home/husrcf/wk/hipfire-psr/target',CARGO_BUILD_JOBS='6',ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1')
jobs=[('clippy-final',['/home/husrcf/.cargo/bin/cargo','clippy','-p','rdna-compute','--features','lab','--example','fn_expert_table_indirection','--locked']),('precommit-final',['bash','scripts/pre-commit-ci.sh','9f71309bfc7542a78e748e478bc751d4111c2845']),('diff-final',['git','diff','--check']),('staged-diff-final',['git','diff','--cached','--check'])]
report={'scope':'required precommit and targeted lab-example check; GPU hidden; no model','base':'9f71309bfc7542a78e748e478bc751d4111c2845','results':[]}
for name,cmd in jobs:
 row={'name':name,'command':cmd,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 with (art/(name+'.log')).open('w') as log:
  row['exit_code']=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 row['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();report['results'].append(row)
 (art/'final-cpu-checks.json').write_text(json.dumps(report,indent=2)+'\n')
report['passed']=all(x['exit_code']==0 for x in report['results'])
(art/'final-cpu-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'checks':len(report['results'])}))
raise SystemExit(0 if report['passed'] else 1)
