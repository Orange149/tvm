from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from strong_baseline import ranking as w
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Apply the frozen 13FPS complete formula to YOLO own full-domain profiles')
    p.add_argument('command',choices=('prepare','rank','validate'));a=p.parse_args()
    print(json.dumps(getattr(w,a.command)(),ensure_ascii=False,indent=2))
