from pathlib import Path
import sys,json,subprocess,cv2,numpy as np,torch
from scipy.io import wavfile
import python_speech_features
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.syncnet.syncnet_eval import calc_pdist
p=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011');out=p/'word-lipsync-v2';torch.set_num_threads(4)
scene=int(sys.argv[1]);version=sys.argv[2];source=p/f'word-lipsync-{version}/scene{scene}-refined/result-24fps.mp4'
run=out/f'metric-scene{scene}-{version}-refined-24';run.mkdir(exist_ok=False)
# Identical context crops improve detection of the distant face without altering time.
crop='crop=280:330:155:180,scale=560:660' if scene==1 else 'crop=380:450:100:100,scale=760:900'
subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(source),'-i',str(p/f'word-lipsync-v1/scene{scene}-guide.wav'),'-map','0:v','-map','1:a','-vf',crop+',fps=25','-c:v','libx264','-crf','16','-c:a','aac',str(run/'context.mp4')],check=True)
detector=SyncNetDetector(device='cpu',detect_results_dir=str(run/'detect'));detector(str(run/'context.mp4'),min_track=20)
files=list((run/'detect/crop').glob('*.mp4'));assert len(files)==1,files
f=files[0];cap=cv2.VideoCapture(str(f));frames=[]
while True:
 ret,im=cap.read()
 if not ret:break
 frames.append(cv2.resize(im,(224,224)))
cap.release();subprocess.run(['ffmpeg','-v','error','-n','-i',str(f),'-ac','1','-ar','16000',str(run/'audio.wav')],check=True)
sr,audio=wavfile.read(run/'audio.wav');mfcc=python_speech_features.mfcc(audio,sr).T
im=torch.from_numpy(np.stack(frames).transpose(3,0,1,2)[None].astype(np.float32));au=torch.from_numpy(mfcc[None,None].astype(np.float32));length=min(len(frames),len(audio)//640)-5
model=SyncNetEval(device='cpu');model.loadParameters('checkpoints/auxiliary/syncnet_v2.model');model.__S__.eval();vf=[];af=[]
with torch.no_grad():
 for i in range(0,length,10):
  inds=range(i,min(i+10,length));vf.append(model.__S__.forward_lip(torch.cat([im[:,:,j:j+5] for j in inds])));af.append(model.__S__.forward_aud(torch.cat([au[:,:,:,j*4:j*4+20] for j in inds])))
vf=torch.cat(vf);af=torch.cat(af);zero=torch.nn.functional.pairwise_distance(vf,af).numpy();d=torch.stack(calc_pdist(vf,af,vshift=10)).numpy();times=(np.arange(len(zero))+2)/25
words={1:[('พ่อ',.9,1.24),('ซ่อน',1.24,1.46),('อะไร',1.46,1.68),('ไว้',1.68,1.96),('เนี่ย',1.96,2.16)],3:[('วันนี้',.92,1.2),('หนู',1.2,1.34),('จะ',1.34,1.46),('เริ่ม',1.46,1.68),('ใหม่',1.68,1.84),('แล้ว',1.84,1.98),('นะ',1.98,2.1)]}[scene]
sel=(times>=words[0][1])&(times<words[-1][2]);means=d[sel].mean(axis=0);best=int(np.argmin(means));report={'scene':scene,'version':version,'method':'SyncNet v2 CPU; 5-frame windows over ASR-estimated word intervals; comparative evidence, not Thai phoneme acceptance','speechZeroOffsetDistance':float(zero[sel].mean()),'bestOffsetFrames25fps':10-best,'bestDistance':float(means[best]),'confidence':float(np.median(means)-means[best]),'wordMetrics':[]}
for word,start,end in words:
 sel=(times>=start)&(times<end);vals=zero[sel];md=d[sel].mean(axis=0);idx=int(np.argmin(md));report['wordMetrics'].append({'word':word,'start':start,'end':end,'zeroOffsetDistance':float(vals.mean()),'windows':int(len(vals)),'bestOffsetFrames25fps':10-idx,'confidence':float(np.median(md)-md[idx])})
(run/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));np.savez(run/'features.npz',video=vf.numpy(),audio=af.numpy(),times=times,distances=d);print(json.dumps(report,ensure_ascii=False),flush=True)
