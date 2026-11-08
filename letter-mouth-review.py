from pathlib import Path
import cv2,mediapipe as mp,json,numpy as np
p=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011');out=p/'word-lipsync-v1'
words={1:[('พ่อ',.90,1.24),('ซ่อน',1.24,1.46),('อะไร',1.46,1.68),('ไว้',1.68,1.96),('เนี่ย',1.96,2.16)],3:[('วันนี้',.92,1.20),('หนู',1.20,1.34),('จะ',1.34,1.46),('เริ่ม',1.46,1.68),('ใหม่',1.68,1.84),('แล้ว',1.84,1.98),('นะ',1.98,2.10)]}
report={'method':'MediaPipe lip aperture at estimated Thai word intervals; no phoneme or human quality acceptance','qualityApproved':False,'scenes':{}}
for scene in [1,3]:
 f=out/f'scene{scene}-attempt1/result-24fps.mp4';cap=cv2.VideoCapture(str(f));frames=[]
 with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.4,min_tracking_confidence=.4) as mesh:
  while True:
   ret,im=cap.read()
   if not ret:break
   if scene==3: im=cv2.resize(im[100:550,100:480],(760,900))
   r=mesh.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ratio=None
   if r.multi_face_landmarks:
    l=r.multi_face_landmarks[0].landmark;xy=lambda i:np.array([l[i].x*im.shape[1],l[i].y*im.shape[0]])
    ratio=float(np.linalg.norm(xy(13)-xy(14))/max(np.linalg.norm(xy(61)-xy(291)),1))
   frames.append({'frame':len(frames),'time':len(frames)/24,'lipAperture':ratio})
 cap.release();rw=[]
 for label,a,b in words[scene]:
  values=[x['lipAperture'] for x in frames if a<=x['time']<b and x['lipAperture'] is not None]
  rw.append({'word':label,'estimatedStart':a,'estimatedEnd':b,'faceFrames':len(values),'minAperture':min(values) if values else None,'maxAperture':max(values) if values else None,'quality':'pending-audiovisual-review'})
 report['scenes'][str(scene)]={'wordReview':rw,'frames':frames,'faceDetectedFrames':sum(x['lipAperture'] is not None for x in frames)}
(out/'word-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print({s:{'detected':v['faceDetectedFrames'],'words':len(v['wordReview'])} for s,v in report['scenes'].items()})
