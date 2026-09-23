import csv, datetime, hashlib, html, importlib.metadata, json, os, pathlib, platform, re, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
Q=ROOT/'qa/independent_review_20260923'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
import pexpect, torch
from run_evals import load_model,model_hash,suite_hash,load_suite
runs={'starter':'20260923T043154_967139Z','expanded':'20260923T062639_403952Z'}
commands=[]
checks={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__},'evaluations':{}}
for variant,run in runs.items():
 for stage,filename in [('untrained','model_untrained.pt'),('final','model.pt')]:
  label=f'{variant}-{stage}'; saved_dir=ROOT/'llm_runs'/run/'language_evals'/stage; out=Q/'replay'/label
  cmd=[str(ROOT/'.venv/bin/python'),'run_evals.py','--model',f'llm_runs/{run}/{filename}','--output',str(out)]
  start=time.monotonic(); result=subprocess.run(cmd,text=True,capture_output=True)
  entry={'command':cmd,'exit_code':result.returncode,'duration_seconds':time.monotonic()-start,'stdout':result.stdout,'stderr':result.stderr}
  commands.append(entry); (Q/f'{label}.log').write_text(result.stdout+result.stderr)
  print(label,entry['exit_code'],result.stdout,flush=True)
  if result.returncode: continue
  model,vocab,saved=load_model(ROOT/'llm_runs'/run/filename)
  rows=json.loads((out/'eval_results.json').read_text()); expected=json.loads((saved_dir/'eval_results.json').read_text())
  differences=[]
  for a,b in zip(rows,expected):
   for key in a:
    if key!='stage' and a[key]!=b[key]: differences.append({'id':a['id'],'field':key,'replayed':a[key],'saved':b.get(key)})
  csv_rows=list(csv.DictReader((saved_dir/'eval_results.csv').open()))
  csv_differences=[]
  for a,b in zip(expected,csv_rows):
   for key,value in a.items():
    expected_text=json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else '' if value is None else str(value)
    if b[key]!=expected_text: csv_differences.append({'id':a['id'],'field':key})
  summary=json.loads((out/'eval_summary.json').read_text()); saved_summary=json.loads((saved_dir/'eval_summary.json').read_text())
  summary_difference_keys=[k for k in summary if k!='stage' and summary[k]!=saved_summary[k]]
  record={'run_id':run,'checkpoint':filename,'completed_steps':saved.get('completed_steps'),'model_hash':model_hash(model),'checkpoint_file_sha256':hashlib.sha256((ROOT/'llm_runs'/run/filename).read_bytes()).hexdigest(),'vocabulary_size':len(vocab),'parameter_count':sum(p.numel() for p in model.parameters()),'model_args':saved['model_args'],'replay_case_count':len(rows),'saved_json_case_count':len(expected),'saved_csv_case_count':len(csv_rows),'rows_identical_except_stage':not differences,'row_differences':differences,'summary_identical_except_stage':not summary_difference_keys,'summary_difference_keys':summary_difference_keys,'csv_matches_json':not csv_differences,'csv_differences':csv_differences,'suite_matches_current':json.loads((out/'eval_cases.json').read_text())==load_suite() and json.loads((saved_dir/'eval_cases.json').read_text())==load_suite(),'stage_from_documented_command':summary['stage'],'saved_stage':saved_summary['stage'],'summary':summary}
  checks['evaluations'][label]=record
# Exact documented command with a fresh review output destination; actual PTY, not fabricated input/output.
transcript=Q/'own_chat_transcript.json'
command=f'source .venv/bin/activate\npython chat.py --model llm_runs/{runs["expanded"]}/model.pt --transcript {transcript.relative_to(ROOT)}'
start=time.monotonic(); events=[]
class Capture:
 def write(self,text): events.append([round(time.monotonic()-start,6),'o',text])
 def flush(self): pass
child=pexpect.spawn('/bin/zsh',['-c',command],cwd=str(ROOT),encoding='utf-8',timeout=45,dimensions=(32,140))
child.logfile_read=Capture()
child.expect_exact('You: ')
plan=json.loads((Q/'own_prompts.json').read_text())
for row in plan['prompts']:
 child.sendline(row['prompt']); child.expect_exact('You: ')
child.sendline('/quit'); child.expect(pexpect.EOF); child.close()
commands.append({'command':command,'pty':True,'exit_code':child.exitstatus,'duration_seconds':time.monotonic()-start,'prompts_file':'own_prompts.json'})
raw=''.join(e[2] for e in events); (Q/'own_chat_terminal.txt').write_text(raw)
cast={'version':2,'width':140,'height':32,'timestamp':int(time.time()),'env':{'SHELL':'/bin/zsh','TERM':'xterm-256color'},'title':'Independent actual saved-model chat: expanded original run'}
(Q/'own_chat.cast').write_text('\n'.join(json.dumps(x) for x in [cast,*events])+'\n')
script_events=json.dumps(events).replace('</','<\\/')
(Q/'own_chat_recording.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent saved-model chat recording</title><style>body{margin:32px;background:#111827;color:#e5e7eb;font:16px system-ui}button{padding:10px;margin-right:8px}pre{background:#020617;white-space:pre-wrap;padding:24px;border:1px solid #475569;border-radius:10px;font:15px/1.55 monospace}.note{max-width:1000px;color:#cbd5e1}</style><h1>Actual terminal recording — independent review</h1><p class="note">Captured from a real pseudo-terminal running the README chat command against original expanded run 20260923T062639_403952Z. Prompts were frozen in own_prompts.json before opening evaluation result files. This player replays captured terminal output; it does not generate text.</p><button id="play">Replay</button><button id="all">Show complete session</button><span id="status"></span><pre id="terminal"></pre><script>const events='''+script_events+'''; const el=document.getElementById('terminal');let timers=[]; function clear(){timers.forEach(clearTimeout);timers=[];el.textContent='';}function all(){clear();el.textContent=events.map(e=>e[2]).join('');document.getElementById('status').textContent='Complete captured session';}document.getElementById('all').onclick=all;document.getElementById('play').onclick=()=>{clear();for(const e of events){timers.push(setTimeout(()=>{el.textContent+=e[2];document.getElementById('status').textContent=e[0].toFixed(2)+' s';},e[0]*1000));}};all();</script></html>''')
chat=json.loads(transcript.read_text()); checks['chat']={'command_exit_code':child.exitstatus,'turn_count':len(chat['turns']),'prompts_match_frozen_plan':[r['prompt'] for r in plan['prompts']]==[t['prompt'] for t in chat['turns']],'model_hash_matches_replayed_expanded_final':chat['model_sha256']==checks['evaluations']['expanded-final']['model_hash'],'transcript':chat,'recording_files':['own_chat_terminal.txt','own_chat.cast','own_chat_recording.html']}
checks['suite_sha256']=suite_hash(load_suite())
(Q/'checks.json').write_text(json.dumps(checks,indent=2)+'\n'); (Q/'command_log.json').write_text(json.dumps(commands,indent=2)+'\n')
print(raw,flush=True)
print('Wrote checks.json and command_log.json',flush=True)
