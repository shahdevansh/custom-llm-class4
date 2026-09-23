"""Independent read-only verification of original notebook and learning artifacts.
Writes only this directory; reconstructs both original 3,000-step experiments in memory.
"""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, math, random, re, subprocess, sys
import nbformat
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
COMMIT='56f610f7e4ae183db78bf9201cf9d543dfde654a'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('independent_nanogpt',ROOT/'nanogpt_model.py')
ng=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ng;spec.loader.exec_module(ng)
torch.set_num_threads(4)
report={'commit':COMMIT,'checks':[], 'runs':{}, 'notebooks':{},'file_hashes':{}}
def check(name,actual,expected,tol=None):
    ok=abs(actual-expected)<=tol if tol is not None else actual==expected
    report['checks'].append({'name':name,'pass':bool(ok),'actual':actual,'expected':expected})
    if not ok: print('FAIL',name,actual,expected,flush=True)
def load(path):return json.loads((ROOT/path).read_text())
def original(path):
    p=ROOT/path;data=p.read_bytes(); committed=subprocess.check_output(['git','show',f'{COMMIT}:{path}'],cwd=ROOT)
    check(f'{path} identical to original commit',data==committed,True)
    report['file_hashes'][str(path)]=hashlib.sha256(data).hexdigest()
    return data
for p in ['custom_llm.ipynb','custom_llm_expanded.ipynb','source/course_starter.ipynb','nanogpt_model.py','custom_llm.py','docs/learning_evidence.md','README.md','experiment_runs.json']:
    original(p)
base=nbformat.read(ROOT/'source/course_starter.ipynb',as_version=4)
ledger=load('experiment_runs.json')
for name,entry in ledger.items():
    path=entry['notebook'];n=nbformat.read(ROOT/path,as_version=4);nbformat.validate(n)
    codes=[(i,c) for i,c in enumerate(n.cells) if c.cell_type=='code']
    check(name+' code execution counts',[c.execution_count for _,c in codes],list(range(1,len(codes)+1)))
    errs=[{'cell':i,'error':o} for i,c in codes for o in c.get('outputs',[]) if o.output_type=='error']
    check(name+' stored notebook errors',errs,[])
    for i,c in codes:ast.parse(c.source,filename=f'{path}:cell{i}')
    changed=[]
    for j,c in enumerate(base.cells):
        k=j if j<24 else j+3
        if c.source!=n.cells[k].source:changed.append(j)
    check(name+' original cells changed only title/settings/prediction',changed,[0,1,2])
    check(name+' additional chat cells source',all(n.cells[i].source==base.cells[23].source.replace('CHAT_PROMPT = "the customer"',f'CHAT_PROMPT = {json.dumps(prompt)}') for i,prompt in [(24,'the nurse'),(25,'the small cups'),(26,'explain quantum teleportation')]),True)
    report['notebooks'][name]={'total_cells':len(n.cells),'code_cells':len(codes),'all_sources_syntax_valid':True,'stored_error_count':len(errs),'original_cell_changes':changed,'final_reflection_cell':n.cells[-1].source}

def tokens(text):return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]",text.lower(),flags=re.UNICODE)
def encode(text,vocab):
    idx={x:i for i,x in enumerate(vocab)};return [idx.get(x,0) for x in tokens(text)]
def batch(docs,vocab):
    ss=[[1]+encode(d,vocab)+[2] for d in docs];length=max(len(s)-1 for s in ss)
    x=torch.full((len(ss),length),2,dtype=torch.long);y=torch.full_like(x,-1)
    for i,s in enumerate(ss):x[i,:len(s)-1]=torch.tensor(s[:-1]);y[i,:len(s)-1]=torch.tensor(s[1:])
    return x,y
@torch.no_grad()
def panel_loss(m,docs,vocab):
    m.eval();x,y=batch(docs,vocab);logits,loss=m(x,y)
    # independent nonpadding reduction via observed-token log probabilities
    logp=F.log_softmax(logits,dim=-1);mask=y!=-1;manual=-logp[mask].gather(1,y[mask,None]).mean().item()
    return loss.item(),manual,int(mask.sum())
@torch.no_grad()
def probs(m,text,vocab):
    m.eval();return F.softmax(m(torch.tensor([[1]+encode(text,vocab)]))[0][0,-1],dim=-1).tolist()
@torch.no_grad()
def samples(m,vocab,temp=.8):
    m.eval();g=torch.Generator(device='cpu').manual_seed(2026);result=[]
    for _ in range(4):
        ids=[1];out=[]
        for _ in range(32):
            logits=m(torch.tensor([ids[-48:]]))[0][0,-1]
            nxt=torch.multinomial(F.softmax(logits/temp,dim=-1),1,generator=g).item()
            if nxt==2:break
            ids.append(nxt);out.append(vocab[nxt])
        result.append(' '.join(out))
    return result
@torch.no_grad()
def attention(m,text,vocab):
    ids=torch.tensor([[1]+encode(text,vocab)]);n=ids.shape[1]
    x=m.transformer.wte(ids)+m.transformer.wpe(torch.arange(n));b=m.transformer.h[0]
    q,k,_=b.attn.c_attn(b.ln_1(x)).split(64,dim=-1)
    q=q.reshape(1,n,4,16).transpose(1,2);k=k.reshape(1,n,4,16).transpose(1,2)
    score=q@k.transpose(-2,-1)/4
    score.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf'))
    return score.softmax(-1)[0,0].tolist()
for name,entry in ledger.items():
    run=entry['run_dir'];rd=ROOT/run;cfg=load(f'{run}/config.json');sp=load(f'{run}/split.json');tok=load(f'{run}/tokenization.json');ins=load(f'{run}/inspection.json');hist=load(f'{run}/history.json');cp=load(f'{run}/checkpoint.json');vocab=tok['vocabulary']
    for fn in ['config.json','split.json','tokenization.json','inspection.json','history.json','checkpoint.json','model.pt','model_untrained.pt','training.csv','temperature_comparison.json','training_summary.json','samples/step_0000.txt','samples/step_1500.txt','samples/step_3000.txt']:
        original(f'{run}/{fn}')
    initial=torch.load(rd/'model_untrained.pt',map_location='cpu',weights_only=True);final=torch.load(rd/'model.pt',map_location='cpu',weights_only=True)
    for stage,ck in [('initial',initial),('final',final)]:
        check(name+' '+stage+' model vocabulary',ck['vocabulary'],vocab)
    check(name+' initial completed_steps',initial['completed_steps'],0);check(name+' final completed_steps',final['completed_steps'],3000)
    check(name+' tokenization example',tok['example'],sp['train'][0]);expected=[1]+encode(tok['example'],vocab)+[2]
    check(name+' tokenization IDs',tok['ids'],expected);check(name+' shifted inputs',tok['inputs'],expected[:-1]);check(name+' shifted targets',tok['targets'],expected[1:])
    check(name+' customer ID',vocab.index('customer'),ins['token_id'])
    check(name+' checkpoint initial table',cp['initial_embeddings'],initial['model']['transformer.wte.weight'].tolist())
    check(name+' checkpoint final table',cp['weights']['wte'],final['model']['transformer.wte.weight'].tolist())
    check(name+' before embedding',ins['embedding_before'],initial['model']['transformer.wte.weight'][ins['token_id']].tolist())
    check(name+' after embedding',ins['embedding_after'],final['model']['transformer.wte.weight'][ins['token_id']].tolist())
    torch.manual_seed(42);m=ng.GPT(ng.GPTConfig(**initial['model_args']))
    check(name+' saved initial tensors equal fresh seed42 model',all(torch.equal(x,initial['model'][k]) for k,x in m.state_dict().items()),True)
    check(name+' parameter count',sum(p.numel() for p in m.parameters()),cfg['parameters'])
    csvhist=[{'step':int(r['step']),'training_loss':float(r['training_loss']),'validation_loss':float(r['validation_loss'])} for r in csv.DictReader((rd/'training.csv').open())]
    check(name+' CSV/history equal',csvhist,hist)
    check(name+' baseline probabilities',probs(m,ins['prefix'],vocab),ins['probabilities_before'])
    replay=[]
    def record(step):
        r={'step':step};counts={}
        for label,docs in [('training',sp['evaluation_train']),('validation',sp['evaluation_validation'])]:
            value,manual,n=panel_loss(m,docs,vocab);r[label+'_loss']=value;counts[label]=n
            check(f'{name} step{step} {label} loss equals manual nonpadding target mean',value,manual,1e-6)
        replay.append(r)
        check(f'{name} step{step} loss history',r,next(x for x in hist if x['step']==step))
        check(f'{name} step{step} all four generation samples',samples(m,vocab),(rd/f'samples/step_{step:04d}.txt').read_text().split('\n'))
        print(name,'verified',r,'target_counts',counts,flush=True)
        return counts
    counts=record(0)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);rng=random.Random(43);first={}
    for step in range(3000):
        lr=.001*min(1,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/2900)))
        for group in opt.param_groups:group['lr']=lr
        m.train();opt.zero_grad(set_to_none=True);_,loss=m(*batch(rng.choices(sp['train'],k=32),vocab));loss.backward()
        if step==0:first={'token':'customer','coordinate':0,'before':m.transformer.wte.weight[ins['token_id'],0].item(),'gradient':m.transformer.wte.weight.grad[ins['token_id'],0].item(),'learning_rate':lr}
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if step==0:
            first['after']=m.transformer.wte.weight[ins['token_id'],0].item();check(name+' actual first gradient/update replay',first,ins['first_update']);first_norm=float(norm)
        if step+1 in [1500,3000]:record(step+1)
    check(name+' all final tensors equal complete independent training replay',all(torch.equal(x,final['model'][k]) for k,x in m.state_dict().items()),True)
    check(name+' final probabilities',probs(m,ins['prefix'],vocab),ins['probabilities_after']);check(name+' first-head causal attention',attention(m,ins['prefix'],vocab),ins['attention_rows'])
    for t,texts in load(f'{run}/temperature_comparison.json').items():check(name+f' all temperature {t} samples',samples(m,vocab,float(t)),texts)
    report['runs'][name]={'run_dir':run,'model_args':initial['model_args'],'replayed_loss_history':replay,'nonpadding_panel_targets':counts,'first_update':first,'first_global_gradient_norm':first_norm,'parameter_count':cfg['parameters'],'final_tensor_comparison':'exact equality','before_embedding_shape':list(initial['model']['transformer.wte.weight'].shape),'after_embedding_shape':list(final['model']['transformer.wte.weight'].shape)}
# Keep result compact by representing long equal arrays only by hashes/counts.
for c in report['checks']:
    for key in ['actual','expected']:
        s=json.dumps(c[key],sort_keys=True)
        if len(s)>1000:c[key]={'serialized_sha256':hashlib.sha256(s.encode()).hexdigest(),'serialized_characters':len(s)}
report['summary']={'total_checks':len(report['checks']),'passed':sum(c['pass'] for c in report['checks']),'failed':sum(not c['pass'] for c in report['checks'])}
(OUT/'notebook_learning_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['summary']),flush=True)
