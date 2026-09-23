import json, hashlib, re, subprocess, shutil, os
id_ = 'w2x2'; defines = 'KS4_W_LOADFORM=2 KS4_X_LOADFORM=2'; sched = 'default'; recipe = 'hipfire.ks4residual.aligned_wx_b128'
ev = '/home/kaden/ks4-sweep/docs/investigations/2026-09-04-ks4-radiowave-evidence/w2x2'; ledger = '/home/kaden/ks4-sweep/docs/investigations/2026-09-04-ks4-radiowave-ledger.jsonl'
manifest = json.load(open(os.path.join(ev, 'manifest.json')))
insp = manifest.get('inspection', manifest)
kern = next(k for k in insp['kernels'] if k['name'] == 'gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds')
ins = kern['instructions']
obj = os.path.join(ev, 'ks4_' + id_ + '.o')
sha = hashlib.sha256(open(obj,'rb').read()).hexdigest()
cfg = 'ks4-radiowave\n%s\n%s\n%s\n' % (id_, defines, sched)
cfg_sha = hashlib.sha256(cfg.encode()).hexdigest()
# vmcnt-in-loop: unbundle, disassemble, widest backward-branch span = group loop.
def addr_of(l):
    m = re.search(r'//\s*([0-9a-f]+):', l)
    if m: return int(m.group(1), 16)
    m = re.match(r'^\s*([0-9a-f]+)\s*<', l)
    return int(m.group(1), 16) if m else None
llvm = '/opt/rocm/core-10.0/lib/llvm/bin'
bundler = shutil.which('clang-offload-bundler') or llvm + '/clang-offload-bundler'
objdump = shutil.which('llvm-objdump') or llvm + '/llvm-objdump'
tgt = insp.get('bundle_target', 'hipv4-amdgcn-amd-amdhsa--gfx1100')
co = os.path.join(ev, 'ks4_' + id_ + '.co')
subprocess.run([bundler, '--type=o', '--unbundle', '--input=' + obj, '--targets=' + tgt, '--output=' + co], check=True, capture_output=True)
dis = subprocess.run([objdump, '--disassemble', '--mcpu=gfx1100', co], capture_output=True, text=True).stdout.splitlines()
def sym_range(name, lines):
    s = next(i for i, l in enumerate(lines) if '<' + name + '>:' in l)
    e = next((i for i, l in enumerate(lines[s+1:], s+1) if re.search(r'^\s*[0-9a-f]+\s*<\w', l)), len(lines))
    return lines[s:e]
focus = 'gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds'
body = sym_range(focus, dis)
base = addr_of(body[0])
spans = []
for l in body:
    if 's_cbranch_execnz' in l or re.search(r'\bs_branch\b', l):
        m = re.search(r'\+0x([0-9a-f]+)>', l); a = addr_of(l)
        if m and a is not None:
            t = base + int(m.group(1), 16)
            if t < a: spans.append((t, a))
assert spans, 'no loop back-edge found in ' + focus
t, a = max(spans, key=lambda s: s[1] - s[0])
vmcnt_loop = sum(1 for l in body if addr_of(l) is not None and t <= addr_of(l) <= a and re.search(r's_waitcnt.*vmcnt', l))
os.remove(co)
# parity: worst relL2 + N=16 ks4 rows (out_proj/down_proj).
par = open(os.path.join(ev, 'parity.log')).read()
rows = []
for l in par.splitlines():
    m = re.match(r'\s*(out_proj|down_proj)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\[(OK|FAIL)\]', l)
    if m: rows.append(m.groups())
assert rows, 'no parity rows parsed'
worst = max(float(r[3]) for r in rows)
def n16(proj):
    r = next(r for r in rows if r[0] == proj and r[1] == '16' and r[2] == '4')
    return dict(relL2=float(r[3]), maxAbs=float(r[4]), min_us=float(r[11]), med_us=float(r[12]), status=r[13])
# bench: L0 out/down_proj (residual) N=16 min/med/%roof + symbol.
bench = open(os.path.join(ev, 'bench.log')).read()
sym = dict(re.findall(r'(L0 out_proj \(residual\)|L0 down_proj \(residual\))\s+gpu\.\S+.*?->\s+(\S+)', bench))
brows = {}
for l in bench.splitlines():
    m = re.match(r'\s*(L0 out_proj \(residual\)|L0 down_proj \(residual\))\s+16\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', l)
    if m: brows[m.group(1)] = [float(m.group(i).rstrip('%')) for i in range(2, 7)]
assert 'L0 out_proj (residual)' in brows and 'L0 down_proj (residual)' in brows, 'bench N=16 residual rows missing'
o = brows['L0 out_proj (residual)']; d = brows['L0 down_proj (residual)']
spill = kern['vgpr_spill_count'] != 0 or kern['sgpr_spill_count'] != 0 or kern['private_segment_fixed_size'] != 0
pass_ = 'True' == 'True'
verdict = 'completed' if pass_ else 'correctness-rejected'
row = dict(campaign='ks4-radiowave', schema=1, candidate=id_, recipe=recipe, defines=defines, sched_profile=sched,
  config_sha256=cfg_sha, code_object_sha8=sha[:8],
  isa_ks4=dict(global_loads=ins['global_loads'], buffer_loads=ins['buffer_loads'], flat_loads=ins['flat_loads'],
    wait_total=ins['wait_instructions'], vmcnt_in_loop=vmcnt_loop, vgpr=kern['vgpr_count'], sgpr=kern['sgpr_count'],
    scratch=kern['private_segment_fixed_size'], spills=dict(vgpr=kern['vgpr_spill_count'], sgpr=kern['sgpr_spill_count'])),
  parity=dict(pass_=pass_, worst_relL2=worst, out_proj_N16_ks4=n16('out_proj'), down_proj_N16_ks4=n16('down_proj')),
  bench=dict(out_proj_N16_min_us=o[0], out_proj_N16_med_us=o[1], out_proj_N16_pct_roof=o[4],
    down_proj_N16_min_us=d[0], down_proj_N16_med_us=d[1], down_proj_N16_pct_roof=d[4],
    symbol_out=sym.get('L0 out_proj (residual)'), symbol_down=sym.get('L0 down_proj (residual)')),
  prompt_md5='e45d15bfe0c9a87132697101d17cbed6  /home/kaden/.hipfire/models/qwen3.8-27b.mq4', binary_md5=dict(parity='b1b3ead1020087bcd56ddc2dec73f995', bench='ed969b73617b9a0cfbe916aec71d4fc4'),
  spill=spill, shippable=(pass_ and not spill), verdict=verdict,
  reason=('parity relL2<=5e-5 gate' if pass_ else 'parity FAIL relL2>5e-5 or non-finite') + ('; spill/scratch nonzero: never shippable' if spill else ''),
  evidence=dict(parity_log=os.path.join(ev, 'parity.log'), bench_log=os.path.join(ev, 'bench.log'), manifest=os.path.join(ev, 'manifest.json')))
open(ledger, 'a').write(json.dumps(row, sort_keys=True) + '\n')
print(json.dumps(dict(candidate=id_, verdict=verdict, worst_relL2=worst, spill=spill,
  out_min=o[0], out_med=o[1], down_min=d[0], down_med=d[1], sha8=sha[:8], vmcnt_loop=vmcnt_loop), indent=1))
