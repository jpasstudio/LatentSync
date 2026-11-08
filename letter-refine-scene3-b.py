from pathlib import Path
import cv2,numpy as np,subprocess,json,hashlib
p=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011');out=p/'word-lipsync-v2/scene3-refined-b';out.mkdir(exist_ok=False)
a=p/'word-lipsync-v1/scene3-attempt1/result-24fps.mp4';b=p/'word-lipsync-v2/scene3-attempt1/result-24fps.mp4';ca=cv2.VideoCapture(str(a));cb=cv2.VideoCapture(str(b));weights=[]
cmd=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','576x1024','-r','24','-i','pipe:0','-i',str(p/'word-lipsync-v1/scene3-guide.wav'),'-map','0:v','-map','1:a','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(124/24),str(out/'result-24fps.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
for i in range(124):
 ra,fa=ca.read();rb,fb=cb.read();assert ra and rb
 t=i/24;weight=max(0,min(1,(t-1.16)/.08,(2.00-t)/.08));weight*=min(1,max(0,(1.34-t)/.06,(t-1.46)/.08));weights.append(weight)
 im=np.rint(fa.astype(float)*(1-weight)+fb.astype(float)*weight).clip(0,255).astype('uint8');proc.stdin.write(im.tobytes())
proc.stdin.close();assert proc.wait()==0;ca.release();cb.release()
r={'method':'Same-frame blend, retain v1 before 1.16 s; blend in 1.16-1.24 s and blend out 1.92-2.00 s; retain v1 around ja: blend out 1.28-1.34 and back 1.46-1.54; audio timing unchanged','sources':{str(f):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in [a,b]},'weights':weights,'fps':24,'frames':124,'qualityApproved':False};(out/'refinement.json').write_text(json.dumps(r,indent=2))
