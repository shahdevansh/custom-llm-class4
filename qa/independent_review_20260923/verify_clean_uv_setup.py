import datetime, json, os, pathlib, subprocess, tempfile, time
R=pathlib.Path(__file__).resolve().parents[2];Q=R/'qa/independent_review_20260923';os.chdir(R)
prompts=['the gardener feels','the client discussed the','last week the musicians']
(Q/'clean_setup_prompts.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prompts':prompts,'purpose':'Frozen additional clean-environment smoke prompts before this remedy test; compare identical seeds across environments, not an independent quality benchmark.'},indent=2)+'\n')
log=[];result={'route':'Proposed uv setup remedy; not original README instructions','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prompts':prompts}
def run(cmd,**kwargs):
 start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,**kwargs);e={'command':cmd,'exit_code':p.returncode,'duration_seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr};log.append(e);(Q/'clean_uv_setup_commands.json').write_text(json.dumps(log,indent=2)+'\n');print(json.dumps(e),flush=True);return p
try:
 with tempfile.TemporaryDirectory(prefix='independent_custom_llm_') as temp:
  env=pathlib.Path(temp)/'venv';py=env/'bin/python'
  p=run(['/opt/homebrew/bin/uv','venv','--python','3.12','--seed',str(env)])
  if p.returncode:raise RuntimeError('uv venv failed')
  p=run(['/opt/homebrew/bin/uv','pip','install','--python',str(py),'-r','requirements-lock.txt'])
  if p.returncode:raise RuntimeError('dependency install failed')
  run([str(py),'-m','pip','check'])
  versions=run([str(py),'-c','import sys,torch,numpy,pypdf;print(sys.version);print("torch",torch.__version__,"numpy",numpy.__version__,"pypdf",pypdf.__version__)'])
  transcripts={}
  for label,exe in [('clean',str(py)),('existing',str(R/'.venv/bin/python'))]:
   out=Q/f'clean_setup_{label}_chat.json';p=run([exe,'chat.py','--model','llm_runs/20260923T062639_403952Z/model.pt','--transcript',str(out)],input='\n'.join(prompts+['/quit'])+'\n')
   if p.returncode:raise RuntimeError(label+' chat failed')
   transcripts[label]=json.loads(out.read_text())
  result['chat_transcripts_identical']=transcripts['clean']==transcripts['existing'];result['three_replies_present']=len(transcripts['clean']['turns'])==3 and all(x['response'] for x in transcripts['clean']['turns']);result['model_sha256']=transcripts['clean']['model_sha256'];result['fresh_environment_path']=str(env)
  p=run([str(py),'run_evals.py','--model','llm_runs/20260923T062639_403952Z/model.pt','--output',str(Q/'clean_setup_eval')])
  if p.returncode:raise RuntimeError('clean eval failed')
  result['clean_eval_identical_to_original']=json.loads((Q/'clean_setup_eval/eval_results.json').read_text())==json.loads((R/'llm_runs/20260923T062639_403952Z/language_evals/final/eval_results.json').read_text());result['success']=result['chat_transcripts_identical'] and result['three_replies_present'] and result['clean_eval_identical_to_original']
 result['temporary_environment_removed']=not pathlib.Path(result.get('fresh_environment_path','/unlikely')).exists()
except Exception as e:result['success']=False;result['error']=str(e)
finally:
 result['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(Q/'clean_uv_setup_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
