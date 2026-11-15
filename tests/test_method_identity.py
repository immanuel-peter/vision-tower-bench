import json
import sys
from types import SimpleNamespace

import pytest
import torch

from vtb import perturbation_run
from vtb.cross_run_bootstrap import perturbation
from vtb import methods, probe, probe_run


def test_real_perturbation_producer_rejects_mixed_methods(tmp_path, monkeypatch):
    ids=[str(i) for i in range(20)]
    labels=tmp_path/'labels.json';labels.write_text(json.dumps({'labels':{i:int(i)%2 for i in ids}}))
    manifest=tmp_path/'transforms.json'
    manifest.write_text(json.dumps({'images':20,'conditions':['identity','scale_l1'],'factors':{'scale':[1,.5]}}))
    features=torch.arange(20*2*4).reshape(20,2,4).float()
    monkeypatch.setattr(perturbation_run.cache,'slices',lambda _: [('tower',1)])
    monkeypatch.setattr(perturbation_run.cache,'load',lambda *_: (features,ids,{'model_id':'fixture','relative_depth':1.0}))
    monkeypatch.setattr(probe_run,'train_once',lambda *_: (None,.5,.5))
    monkeypatch.setattr(perturbation_run,'correctness',lambda model,x,y,index,*args: torch.ones(len(index)))
    outputs=[]
    for diagnostic in (False,True):
        out=tmp_path/f'{diagnostic}.json'
        argv=['producer','--clean-run',str(tmp_path),'--features-root',str(tmp_path),'--model-dir','fixture',
              '--labels',str(labels),'--transform-manifest',str(manifest),'--out',str(out),
              '--device','cpu','--width','2','--epochs','1','--seeds','1','--learning-rates','.01','--resamples','20']
        monkeypatch.setattr(sys,'argv',argv+(['--standardize-channels'] if diagnostic else []))
        perturbation_run.main()
        outputs.append(json.loads(out.read_text()))
    args=SimpleNamespace(stage='tower',first_stage=None,second_stage=None,resamples=20,confidence=.95,bootstrap_seed=0)
    assert outputs[0]['protocol']['method']!=outputs[1]['protocol']['method']
    with pytest.raises(ValueError,match='methods differ'):
        perturbation(outputs[0],outputs[1],('first','second'),args)
    for payload in outputs:
        assert 'clean' in perturbation(payload,payload,('first','second'),args)


def test_missing_identity_is_not_implicitly_primary():
    with pytest.raises(ValueError,match='unknown'):
        methods.require_same({'protocol':{}},{'protocol':{}})


def test_legacy_certificate_is_bound_to_exact_payload(monkeypatch):
    payload={'images':20,'protocol':{'seeds':3},'cells':[]}
    monkeypatch.setattr(methods,'legacy_registry',lambda:{methods.payload_hash(payload):{'method':methods.descriptor(),'evidence':['pinned producer']}})
    assert methods.require_same(payload,payload)==methods.descriptor()
    with pytest.raises(ValueError,match='unknown'):
        methods.identify(payload|{'images':21})


def test_common_reducer_fits_only_train_and_preserves_rng():
    torch.manual_seed(53);tokens=torch.randn(12,3,9);tokens[:,:,0]=1
    split=probe_run.Split(torch.arange(7),torch.arange(7,9),torch.arange(9,12))
    state=torch.random.get_rng_state().clone()
    fit=probe_run.fit_reducer(tokens,split,4,standardize_channels=True)
    assert isinstance(fit,probe.StandardizedReducer)
    assert torch.equal(state,torch.random.get_rng_state())
    changed=tokens.clone();changed[7:]*=10000
    other=probe_run.fit_reducer(changed,split,4,standardize_channels=True)
    assert torch.equal(fit.basis,other.basis) and torch.equal(fit.mean,other.mean) and torch.equal(fit.scale,other.scale)
    assert type(probe_run.fit_reducer(tokens,split,4)) is probe.Reducer
    with pytest.raises(TypeError):probe_run.fit_reducer(tokens,split,4,True)


def test_real_recognition_producer_records_its_actual_method(tmp_path, monkeypatch):
    ids=[str(i) for i in range(20)]
    labels=tmp_path/'labels.json';labels.write_text(json.dumps({'labels':{i:int(i)%2 for i in ids}}))
    features=torch.arange(20*2*4).reshape(20,2,4).float()
    monkeypatch.setattr(probe_run,'selected_slices',lambda *_:[('tower',1)])
    monkeypatch.setattr(probe_run.cache,'load',lambda *_:(features,ids,{'model_id':'fixture','relative_depth':1.0}))
    monkeypatch.setattr(probe_run,'run_cell',lambda *args:{'test_accuracy':.5,'test_std':0.,'learning_rate':.01})
    outputs=[]
    for standardized in (False,True):
        out=tmp_path/f'recognition-{standardized}.json'
        argv=['producer','--run',str(tmp_path),'--labels',str(labels),'--out',str(out),
              '--device','cpu','--match-capacity','--width','2']
        monkeypatch.setattr(sys,'argv',argv+(['--standardize-channels'] if standardized else []))
        probe_run.main();payload=json.loads(out.read_text());outputs.append(payload)
        assert methods.identify(payload)==methods.identify(payload['cells'][0])
    with pytest.raises(ValueError,match='methods differ'):
        methods.require_same(*outputs)
