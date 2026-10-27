# synth_pass.py out.mp4 [seed] [птиц] — синтетика для ожидания размера: крупный дрон-крыло пролетает мимо камеры: маленький -> крупный -> маленький
import sys, json, subprocess, numpy as np, cv2
out=sys.argv[1]; rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 1)
W,H,N=1920,1080,330
# небо: градиент + облака (размытый шум)
yy=np.linspace(0,1,H)[:,None]
sky=(150+60*yy)*np.ones((H,W))
cl=cv2.GaussianBlur(rng.normal(0,1,(H//8,W//8)).astype(np.float32),(0,0),6)
cl=cv2.resize(cl,(W,H)); sky+=np.clip(cl*90,-10,25)
def plane(img,cx,cy,size,ang,val):
    s=size/2.0; pts=[]
    body=[(-0.08,-0.9),(0.08,-0.9),(0.08,0.9),(-0.08,0.9)]
    wing=[(-1,0.05),(1,0.05),(1,0.22),(-1,0.22)]
    tail=[(-0.35,0.75),(0.35,0.75),(0.35,0.85),(-0.35,0.85)]
    ca,sa=np.cos(ang),np.sin(ang)
    for poly in (body,wing,tail):
        p=np.array([[cx+s*(x*ca-y*sa), cy+s*(x*sa+y*ca)] for x,y in poly],np.float32)
        cv2.fillPoly(img,[np.round(p*16).astype(np.int32)],val,lineType=cv2.LINE_AA,shift=4)
p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{W}x{H}','-r','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
gt=[]
birds=[(rng.uniform(200,1700),rng.uniform(150,900),rng.uniform(-6,6),rng.uniform(-2,2)) for _ in range(int(sys.argv[3]) if len(sys.argv)>3 else 0)]
for f in range(N):
    u=(f-N/2)/55.0
    x=960+880*np.tanh(u*0.8); y=520+120*np.tanh(u*0.5)
    size=22+380/(1+u*u)
    img=sky.copy()
    m=np.zeros((H,W),np.float32)
    plane(m,x,y,size,0.25*np.tanh(u)+np.pi/2,1.0)
    img=img*(1-m)+70*m
    for i,(bx,by,vx,vy) in enumerate(birds):
        bx2=bx+vx*f; by2=by+vy*f+3*np.sin(f*0.9+i)
        if 0<bx2<W and 0<by2<H: cv2.circle(img,(int(bx2),int(by2)),3,90,-1)
    img+=rng.normal(0,2,(H,W))
    p.stdin.write(np.clip(img,0,255).astype(np.uint8).tobytes())
    gt.append([{'x':float(x),'y':float(y),'w':float(size),'h':float(size*0.9)}])
p.stdin.close(); p.wait()
json.dump({'w':W,'h':H,'frames':gt},open(out.replace('.mp4','.gt.json'),'w'))
