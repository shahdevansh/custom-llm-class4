import datetime,hashlib,html,json,os,pathlib,sys,time
R=pathlib.Path(__file__).resolve().parents[2];Q=R/'qa/independent_review_20260923';os.chdir(R);sys.path.insert(0,str(R))
import pexpect
from run_evals import load_model,model_hash
model_path=pathlib.Path('tuning_runs/20260923T065357_257422Z/runs/balanced_lr0.003_seed42/model.pt')
prompt_path=Q/'own_prompts.json';original_path=Q/'own_chat_transcript.json';review_path=Q/'review.md'
immutable={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prompt_path,original_path,review_path]}
model,vocab,saved=load_model(R/model_path);state_before=model_hash(model);checkpoint_before=hashlib.sha256((R/model_path).read_bytes()).hexdigest()
out=Q/'selected_chat_transcript.json'
if out.exists():raise RuntimeError('Selected transcript exists; refusing overwrite')
command=f'source .venv/bin/activate\npython chat.py --model {model_path} --transcript {out.relative_to(R)}'
start=time.monotonic();events=[]
class Capture:
 def write(self,text):events.append([round(time.monotonic()-start,6),'o',text])
 def flush(self):pass
child=pexpect.spawn('/bin/zsh',['-c',command],cwd=str(R),encoding='utf-8',timeout=45,dimensions=(32,140));child.logfile_read=Capture();child.expect_exact('You: ')
plan=json.loads(prompt_path.read_text())
for p in plan['prompts']:
 child.sendline(p['prompt']);child.expect_exact('You: ')
child.sendline('/quit');child.expect(pexpect.EOF);child.close()
raw=''.join(e[2] for e in events);(Q/'selected_chat_terminal.txt').write_text(raw)
cast={'version':2,'width':140,'height':32,'timestamp':int(time.time()),'env':{'SHELL':'/bin/zsh','TERM':'xterm-256color'},'title':'Selected tuned checkpoint: same eight independent prompts'}
(Q/'selected_chat.cast').write_text('\n'.join(json.dumps(x) for x in [cast,*events])+'\n')
event_js=json.dumps(events).replace('</','<\\/')
(Q/'selected_chat_recording.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Selected checkpoint: actual terminal recording</title><style>body{margin:32px;background:#111827;color:#e5e7eb;font:16px system-ui}button{padding:10px;margin-right:8px}pre{background:#020617;white-space:pre-wrap;padding:24px;border:1px solid #475569;border-radius:10px;font:15px/1.55 monospace}.note{max-width:1000px;color:#cbd5e1}</style><h1>Selected checkpoint — actual terminal recording</h1><p class="note">Real PTY run of chat.py with balanced_lr0.003_seed42/model.pt, tuning run 20260923T065357_257422Z. The same eight prompts and seeds as the original independent review. Captured terminal output only; this playback is not a live model.</p><button id="play">Replay</button><button id="all">Show complete session</button><span id="status"></span><pre id="terminal"></pre><script>const events='''+event_js+''';const el=document.getElementById('terminal');let timers=[];function clear(){timers.forEach(clearTimeout);timers=[];el.textContent='';}function all(){clear();el.textContent=events.map(e=>e[2]).join('');document.getElementById('status').textContent='Complete captured session';}document.getElementById('all').onclick=all;document.getElementById('play').onclick=()=>{clear();for(const e of events){timers.push(setTimeout(()=>{el.textContent+=e[2];document.getElementById('status').textContent=e[0].toFixed(2)+' s';},e[0]*1000));}};all();</script></html>''')
new=json.loads(out.read_text());old=json.loads(original_path.read_text());checks={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'exit_code':child.exitstatus,'duration_seconds':time.monotonic()-start,'selected_checkpoint':str(model_path),'checkpoint_file_sha256':checkpoint_before,'checkpoint_unchanged':hashlib.sha256((R/model_path).read_bytes()).hexdigest()==checkpoint_before,'model_state_sha256':state_before,'transcript_hash_matches_checkpoint':new['model_sha256']==state_before,'completed_steps':saved.get('completed_steps'),'vocabulary_size':len(vocab),'model_args':saved['model_args'],'turn_count':len(new['turns']),'same_frozen_prompts':[t['prompt'] for t in new['turns']]==[p['prompt'] for p in plan['prompts']],'same_prompts_and_seeds_as_original':[(t['prompt'],t['seed']) for t in new['turns']]==[(t['prompt'],t['seed']) for t in old['turns']],'same_generation_settings':all(new[k]==old[k] for k in ['temperature','max_tokens','fresh_context_per_prompt']),'preserved_original_evidence':{p:hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in immutable.items()},'transcript_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
checks['rows']=[{'id':p['id'],'prompt':n['prompt'],'seed':n['seed'],'original_response':o['response'],'selected_response':n['response'],'original_unknown':o['unknown_prompt_words'],'selected_unknown':n['unknown_prompt_words'],'original_truncated':o['prompt_truncated'],'selected_truncated':n['prompt_truncated']} for p,o,n in zip(plan['prompts'],old['turns'],new['turns'])]
(Q/'selected_chat_checks.json').write_text(json.dumps(checks,indent=2)+'\n');(Q/'selected_chat_command_log.json').write_text(json.dumps({'command':command,'exit_code':child.exitstatus,'pty':True,'duration_seconds':checks['duration_seconds'],'stdout_captured':raw},indent=2)+'\n')
print(raw);print(json.dumps({k:v for k,v in checks.items() if k!='rows'},indent=2))
