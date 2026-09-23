import datetime, hashlib, importlib.metadata, json, os, pathlib, re, subprocess, sys, zipfile
R=pathlib.Path(__file__).resolve().parents[2];Q=R/'qa/independent_review_20260923';os.chdir(R)
checks={}; commands=[]
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True); commands.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});return p
# Inspect existing upstream Git objects, no network, no checkout mutation.
base='9e04ddb6aacb8efcb790e70c62550ca55e0f2a75';checks['upstream_commit']=base; checks['upstream_integrity']={}
for path in ['evals/language_evals.json','run_evals.py','chat.py','nanogpt_model.py','test_language_evals.py']:
 p=subprocess.run(['git','show',f'{base}:{path}'],capture_output=True)
 checks['upstream_integrity'][path]={'upstream_exists':p.returncode==0,'bytes_identical':p.stdout==(R/path).read_bytes(),'current_sha256':hashlib.sha256((R/path).read_bytes()).hexdigest(),'upstream_sha256':hashlib.sha256(p.stdout).hexdigest()}
setup=run(['/bin/zsh','-c','python3.12 -m venv qa/independent_review_20260923/clean_setup_venv'])
pip=run([sys.executable,'-m','pip','check'])
checks['setup']={'literal_python312_command_exit_code':setup.returncode,'literal_command_stderr':setup.stderr,'existing_environment_pip_check_exit_code':pip.returncode,'existing_environment_pip_check_stderr':pip.stderr,'boundary':'Fresh dependency installation not attempted because documented Python3.12 launcher is absent; saved-model inference tested with pre-existing .venv.'}
versions=[]
for line in (R/'requirements-lock.txt').read_text().splitlines():
 if '==' not in line: continue
 name,want=line.split('==',1)
 try: actual=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: actual=None
 versions.append({'package':name,'locked':want,'installed':actual,'matches':actual==want})
checks['locked_environment_matches']=all(x['matches'] for x in versions);checks['locked_environment_versions']=versions
checks['zip_archives']={}
for run_id in ['20260923T043154_967139Z','20260923T062639_403952Z']:
 with zipfile.ZipFile(R/'llm_runs'/f'{run_id}.zip') as z:
  mismatches=[]; absent=[]
  for f in z.namelist():
   if f.endswith('/'): continue
   target=R/'llm_runs'/run_id/f
   if not target.exists(): absent.append(f)
   elif z.read(f)!=target.read_bytes(): mismatches.append(f)
  checks['zip_archives'][run_id]={'entry_count':len(z.namelist()),'testzip':z.testzip(),'not_found_at_expected_relative_path':absent,'content_mismatches':mismatches}
# Existing evidence transcript is verified by replay, not by trusting the screenshot.
sys.path.insert(0,str(R)); from run_evals import load_model,generate_reply,model_hash
model,vocab,saved=load_model(R/'llm_runs/20260923T062639_403952Z/model.pt')
original=json.loads((R/'evidence/chat_transcript.json').read_text()); t=[]
for row in original['turns']:
 actual=generate_reply(model,vocab,row['prompt'],seed=row['seed'])
 t.append({'prompt':row['prompt'],'exact_response_match':actual['response']==row['response'],'unknown_words_match':actual['unknown_prompt_words']==row['unknown_prompt_words']})
checks['existing_chat_evidence']={'run_completed_steps':saved['completed_steps'],'model_hash_matches':model_hash(model)==original['model_sha256'],'turn_count':len(t),'replayed_turns':t}
# README links to local files should resolve. Ignore anchors/network links.
links=re.findall(r'\]\(([^)]+)\)',(R/'README.md').read_text()); missing=[]
for x in links:
 if x.startswith(('https://','http://','#')):continue
 path=x.split('#')[0]
 if not (R/path).exists():missing.append(x)
checks['readme_local_links']={'checked':sum(not x.startswith(('https://','http://','#')) for x in links),'missing':missing}
checks['timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(Q/'additional_checks.json').write_text(json.dumps(checks,indent=2)+'\n');(Q/'additional_command_log.json').write_text(json.dumps(commands,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k!='locked_environment_versions'},indent=2))
