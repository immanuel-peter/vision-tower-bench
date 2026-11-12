import collections, datetime, gzip, hashlib, json, re, shutil, subprocess
from pathlib import Path

E = Path('/tmp/vtb-delivery/issues/22')
V = Path('results/evidence/qwen38-flash-next')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git', *args])
base = 'a2277c8dd9d9f81bbf29cc4a5c545f8fd6513902'
assert git('merge-base', 'HEAD', 'origin/main').decode().strip() == base
old_index = json.loads(git('show', base + ':results/index.json'))
index = json.loads(Path('results/index.json').read_text())
for name, value in old_index['cohorts'].items(): assert index['cohorts'][name] == value
old_diag = json.loads(git('show', base + ':results/diagnostics/mimo26-tower-outlier-channel/label_budget_standardized.json'))
diag = json.loads(Path('results/diagnostics/mimo26-tower-outlier-channel/label_budget_standardized.json').read_text())
for name, value in old_diag.items():
    if name != 'runs': assert diag[name] == value
for name, value in old_diag['runs'].items(): assert diag['runs'][name] == value
added = set(diag['runs']) - set(old_diag['runs'])
assert added == {f'qwen38_flash_next_probe_attention_matched_{tag}' for tag in ['f001','f005','f020','f10']}
allowed = {'results/README.md','results/index.json','results/history/roster-run.md',
           'results/diagnostics/mimo26-tower-outlier-channel/label_budget_standardized.json'}
preserved = []
for line in git('ls-tree','-r',base,'results').decode().splitlines():
    meta, name = line.split('\t'); expected = meta.split()[2]
    if name in allowed: continue
    p = Path(name); data = p.read_bytes()
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected, name
    preserved.append(name)
for name in ['qwen-sibling-intervals','qwen-weight-drift']:
    shutil.copyfile(Path('docs/figures')/(name+'.svg'),V/(name+'.svg'))
p=V/'benchmark-summary.md';p.write_text(p.read_text().replace('../../../docs/figures/qwen-sibling-intervals.svg','qwen-sibling-intervals.svg'))
figures = {p.name:sha(p) for p in Path('docs/figures').glob('*.svg')}
assert len(figures) == 12
for name in ['tests-final.log','tests-final-corrected.log','tests-final-complete.log','tests-qwen-weights-final.log',
             'tests-split-regression.log','tests-conditioning.log','tests-standardized.log','figures-final.log']:
    shutil.copyfile(E/name,V/name.replace('.log','.txt'))
assert '156 passed, 50 skipped' in (E/'tests-final-complete.log').read_text()
assert '7 passed' in (E/'tests-qwen-weights-final.log').read_text()
paired = [p for p in Path('results/bootstrap').glob('*.json') if 'qwen38_flash_next' in p.name or 'flash-next-vs-27b' in p.name]
assert len(paired) == 33
paired_sources = {}
for p in paired:
    data = json.loads(p.read_text())
    for name,digest in data.get('sources',{}).items(): assert sha(Path(name)) == digest, (p,name)
    paired_sources[str(p)] = sha(p)
license_path = Path('hf/Qwen3.8-Flash-Next-Vision/LICENSE')
assert sha(license_path) == 'a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08'
assert sha(V/'reducer-channel-standardization.patch') == 'f5a2bc1dcda0e6be9a2d6abea9c1bcc9cb5817282ff6406489b5022dc65e330a'
for name,digest in [('suite-geometry-62bd0d3d6f35.py','62bd0d3d6f354d7bb47b5b717b4224700f81128aad5aea036add529d9db03629'),
                    ('suite-geometry-b4a6cd33a021.py','b4a6cd33a02176f745d45f4978eec661bbbf056491758ef5e37a89adbf7bd148'),
                    ('suite-final-e2823d501cbc.py','e2823d501cbc906df289916d4335c8a8634fbf10ba6d32d39eae19b0f4edbd5d')]:
    assert sha(V/name) == digest
# The copied license, recovered patch, frozen runners, and generated SVGs retain their source bytes.
whitespace = subprocess.run(['git','-c','core.whitespace=-blank-at-eof,cr-at-eol','diff',base,'--check','--','.',
                            ':(exclude)hf/Qwen3.8-Flash-Next-Vision/LICENSE',
                            ':(exclude)*.svg',':(exclude)results/evidence/qwen38-flash-next/*.py',
                            ':(exclude)results/evidence/qwen38-flash-next/*.patch'],capture_output=True,text=True)
assert whitespace.returncode == 0, whitespace.stdout
paths = set(git('ls-files','-z').decode().split('\0')) | set(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))
patterns = [rb'hf_[A-Za-z0-9]{30,}',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',
            rb'AKIA[0-9A-Z]{16}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
findings = []
for name in paths:
    p=Path(name)
    if not p.is_file(): continue
    data=p.read_bytes()
    if p.suffix=='.gz': data=gzip.decompress(data)
    if any(re.search(pattern,data) for pattern in patterns): findings.append(name)
assert not findings, findings
report = {'completed_at':datetime.datetime.now(datetime.UTC).isoformat(),'base':base,
          'unchanged_historical_result_files':len(preserved),'historical_cohorts_exactly_preserved':True,
          'existing_standardized_runs_exactly_preserved':len(old_diag['runs']), 'added_standardized_runs':sorted(added),
          'primary_paired_files':paired_sources,'generated_figures':figures,
          'figure_source_selection_tests':'all 12 render; deterministic deepest-stage sources versus original full sweeps; existing standardized label-budget sources; each of 20 sibling intervals including SPair class-macro checked against its paired source',
          'tests':'156 passed, 50 expected weight-dependent skips; 14 existing TorchScript deprecation warnings',
          'weighted_qwen_tests':'7 passed, 16 deselected', 'secrets_scan_findings':findings,
          'whitespace_review':'Source/license bytes, recovered patch, frozen runner snapshots, autogenerated SVG whitespace and blank EOF retained. Other trailing-whitespace check passed.',
          'cuda_after_release':False,'limitations':['Low-clean primary Tower/merged perturbation readouts do not support intrinsic robustness claims.',
            'Standardization changes scaling and the retained PCA subspace; training cause and any common cause with MiMo remain untested.',
            'Hardware-specific throughput and one KITTI transfer column; paired reruns retain their own points.']}
(V/'final-validation.json').write_text(json.dumps(report,indent=2)+'\n')
(E/'final-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['primary_paired_files','generated_figures']},indent=2))
