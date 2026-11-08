from pathlib import Path
import subprocess
p=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011');out=p/'word-lipsync-v2';font='/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf'
for scene,folder in [(1,'scene1-refined'),(3,'scene3-refined-b')]:
 a=p/f'word-lipsync-v1/scene{scene}-attempt1/result-24fps.mp4';b=out/folder/'result-24fps.mp4';crop='crop=280:330:155:180' if scene==1 else 'crop=240:280:170:135'
 def chain(idx,label):return f'[{idx}:v]{crop},scale=560:660,drawtext=fontfile={font}:text=\'{label} · ฉาก {scene}\':fontcolor=white:fontsize=26:x=16:y=16:box=1:boxcolor=black@0.65[v{idx}]'
 filt=chain(0,'ก่อนปรับ')+';'+chain(1,'ปรับรายคำ')+';[v0][v1]hstack[v]'
 subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(a),'-i',str(b),'-i',str(p/f'word-lipsync-v1/scene{scene}-guide.wav'),'-filter_complex',filt,'-map','[v]','-map','2:a','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t',str(124/24),str(out/f'comparison-scene{scene}.mp4')],check=True)
subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(out/'comparison-scene1.mp4'),'-i',str(out/'comparison-scene3.mp4'),'-filter_complex','[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]','-map','[v]','-map','[a]','-c:v','libx264','-crf','18','-c:a','aac','-b:a','192k',str(out/'word-review-before-after.mp4')],check=True)
