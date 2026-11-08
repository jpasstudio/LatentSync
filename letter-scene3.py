from pathlib import Path
import importlib.util,sys,json,time,threading,subprocess,uuid,hashlib
root=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011')
app=Path('/home/oneflow-ai-workstation/apps/LatentSync-letter-lab')
out=root/'word-lipsync-v1';run=out/'scene3-attempt1';run.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('worker','/home/oneflow-ai-workstation/Projects/worktrees/doraflim-object-lab-20261010/packages/avideo/src/extender-workbench/native-video-worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
video=root/'media/ff476e98-43e1-4e06-8790-1608b3b0ea62.mp4';audio=out/'scene3-guide.wav'
receipt={'schema':'doraflim.lipsync-canary/v1','projectId':'3a7e2a3d-a67b-40f3-bb8a-5b44a3b6d8a8','shotId':'SB_8b326176-238d-4cdf-8c62-9182274c7765','sourceVideoSha256':w.digest(video),'sourceAudioSha256':w.digest(audio),'engineCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=app,text=True).strip(),'engine':'LatentSync-1.6','status':'preparing','qualityApproved':False,'productionConnected':False,'seed':1247,'inferenceSteps':20,'guidanceScale':1.5,'placement':{'startFrame':18,'fps':24,'frames':124}}
def save():w.write(run/'receipt.json',receipt)
save()
leaseid=None;stop=threading.Event();thread=None;process=None
try:
 deadline=time.monotonic()+1200
 while not w.native_queue_idle(w.request('http://127.0.0.1:8197','/queue')):
  if time.monotonic()>deadline:raise RuntimeError('Native queue busy')
  time.sleep(5)
 if w.broker_idle(w.broker('/status')):w.request('http://127.0.0.1:8197','/free',{'unload_models':True,'free_memory':True},allow_empty=True)
 lease=w.broker('/acquire',{'jobKind':'other','estVramMib':24000,'priority':1,'tenantId':'869a5079-a1df-4123-9c09-9d154b185af9','brandId':'doraflim','accountRef':'creator-'+hashlib.sha256(b'a62439d2-9810-4229-be89-dcb928a7694e').hexdigest()[:32],'requestId':str(uuid.uuid4()),'label':'Letter scene3 dedicated lip sync Lab'})
 leaseid=lease['leaseId'];receipt['leaseId']=leaseid;receipt['status']='waiting-for-gpu';save()
 while lease['status']!='granted':
  active=next((x for x in w.broker('/status')['jobs'] if x['id']==leaseid),None)
  if active and active['status']=='active':break
  if not active or time.monotonic()>deadline:raise RuntimeError('GPU lease not granted')
  time.sleep(5)
 errors=[]
 def pulse():
  while not stop.wait(25):
   try:
    if not w.broker('/heartbeat',{'leaseId':leaseid}).get('ok'):errors.append('heartbeat rejected')
   except Exception:errors.append('heartbeat failed')
 thread=threading.Thread(target=pulse,daemon=True);thread.start()
 receipt['status']='running';save()
 cmd=[str(app/'.venv/bin/python'),'-m','scripts.inference','--unet_config_path','configs/unet/stage2_512.yaml','--inference_ckpt_path','checkpoints/latentsync_unet.pt','--video_path',str(video),'--audio_path',str(audio),'--video_out_path',str(run/'result-25fps.mp4'),'--temp_dir',str(run/'temp'),'--inference_steps','20','--guidance_scale','1.5','--seed','1247']
 receipt['command']=cmd;save()
 with (run/'inference.log').open('w') as log:
  process=subprocess.Popen(cmd,cwd=app,stdout=log,stderr=subprocess.STDOUT)
  while process.poll() is None:
   if errors:process.terminate();raise RuntimeError(errors[0])
   time.sleep(2)
  if process.returncode:raise RuntimeError('Inference failed: '+str(process.returncode))
 receipt['status']='rendered-awaiting-review';receipt['outputSha256']=w.digest(run/'result-25fps.mp4');save()
except Exception as e:
 receipt['status']='failed';receipt['error']=str(e);save();raise
finally:
 if process is not None and process.poll() is None:
  process.terminate();process.wait(timeout=30)
 stop.set()
 if thread:thread.join(timeout=5)
 if leaseid:receipt['leaseReleased']=w.broker('/release',{'leaseId':leaseid});save()
