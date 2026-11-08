from PIL import Image,ImageDraw
from pathlib import Path
import subprocess,io,sys
p=Path('/home/oneflow-ai-workstation/oneflow-work/doraflim-project-workspaces/letter-20261011');scene=int(sys.argv[1]);times=[.96,1.08,1.28,1.40,1.52,1.64,1.80,1.96,2.08,2.24]
crop='crop=280:330:155:180,scale=200:236' if scene==1 else 'crop=240:280:170:135,scale=200:236'
out=Image.new('RGB',(2000,522),'#111827');d=ImageDraw.Draw(out)
for row,version in enumerate(['v1','v2']):
 folder=f'scene{scene}-refined' if scene==1 else 'scene3-refined-b'
 f=(p/f'word-lipsync-v1/scene{scene}-attempt1/result-24fps.mp4') if row==0 else p/'word-lipsync-v2'/folder/'result-24fps.mp4'
 for col,t in enumerate(times):
  raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(t),'-i',str(f),'-frames:v','1','-vf',crop,'-f','image2pipe','-vcodec','png','-'])
  out.paste(Image.open(io.BytesIO(raw)),(col*200,row*261+25));d.text((col*200+4,row*261+5),f'{version} {t:.2f}s',fill='white')
out.save(p/f'word-lipsync-v2/scene{scene}-final-comparison.jpg')
