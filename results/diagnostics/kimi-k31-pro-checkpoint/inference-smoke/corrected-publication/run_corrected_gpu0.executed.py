from pathlib import Path
import csv,datetime,fcntl,hashlib,json,os,shlex,subprocess,time
from huggingface_hub import snapshot_download

base=Path(__file__).parent
pub=json.loads((base/'corrected-publication/publication.json').read_text())
out=base/'corrected-publication/smoke';out.mkdir(exist_ok=True)
cache=out/'fresh-cache';assert not cache.exists()
snapshot=Path(snapshot_download(pub['repo'],revision=pub['revision'],cache_dir=str(cache),max_workers=2))
assert hashlib.sha256((snapshot/'examples/inference.py').read_bytes()).hexdigest()==pub['new_example_sha256']
grant=out/'gpu0-grant.json'
g={'active':True,'physical_gpu':0,'phase':'corrected-published-example-smoke','threads':2,'batch_size':1,'revision':pub['revision'],'benchmark_grant_repurposed':False}
grant.write_text(json.dumps(g,indent=2)+'\n')
image='/tmp/vtb-delivery/data/imagenet100/validation/validation_000000.jpg'
work='/home/shadeform/.t3/worktrees/vision-tower-bench/t3-edc83c25'
def state():
    row=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    uuid,mem=next(csv.reader([row]));uuid=uuid.strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    return {'uuid':uuid,'memory_mib':int(mem.strip()),'cuda_processes':[r for r in rows if uuid in r]}
lock=open('/tmp/vtb-delivery/locks/gpu0.lock','a')
os.dup2(lock.fileno(),9);os.set_inheritable(9,True)
fcntl.flock(9,fcntl.LOCK_EX)
try:
    before=state();assert before['memory_mib']==0 and not before['cuda_processes']
    prefix='source '+work+'/.venv/bin/activate\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
    commands=[('standalone','cd '+shlex.quote(str(snapshot))+'\npython examples/inference.py '+image),('numerical','cd '+work+'\nPYTHONPATH=. python '+shlex.quote(str(base/'compare_published_example_adapter.py'))+' --snapshot '+shlex.quote(str(snapshot))+' --revision '+pub['revision']+' --physical-gpu 0 --grant '+shlex.quote(str(grant))+' --lock-fd 9 --image '+image+' --out '+shlex.quote(str(out/'numerical-comparison.json')))]
    runs={}
    for label,command in commands:
        start=time.monotonic();run=subprocess.run(['bash','-ic',prefix+command],text=True,capture_output=True,pass_fds=(9,))
        (out/(label+'.stdout.txt')).write_text(run.stdout);(out/(label+'.stderr.txt')).write_text(run.stderr)
        runs[label]={'command':command,'exit_code':run.returncode,'seconds':time.monotonic()-start}
        assert run.returncode==0,(label,run.stderr)
    report={'passed':True,'revision':pub['revision'],'snapshot':str(snapshot),'physical_gpu':0,'fresh_download':True,'source_sha256':pub['new_example_sha256'],'runs':runs,'numerical':json.loads((out/'numerical-comparison.json').read_text()),'timestamp':datetime.datetime.now(datetime.UTC).isoformat()}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='numerical'},indent=2),flush=True)
finally:
    g['active']=False;g['released']=True;grant.write_text(json.dumps(g,indent=2)+'\n')
    fcntl.flock(9,fcntl.LOCK_UN);fcntl.flock(9,fcntl.LOCK_EX)
    after=state();assert after['memory_mib']==0 and not after['cuda_processes']
    (out/'gpu0-release.json').write_text(json.dumps({'passed':True,'physical_gpu':0,**after,'lock_reacquired':True,'lock_released_after_check':True},indent=2)+'\n')
    fcntl.flock(9,fcntl.LOCK_UN);os.close(9);lock.close()
