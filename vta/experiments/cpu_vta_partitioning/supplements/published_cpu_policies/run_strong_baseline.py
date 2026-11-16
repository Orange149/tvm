from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Freeze the actual 13FPS full-domain baseline before further revision')
    p.add_argument('command',choices=('audit',));a=p.parse_args()
    from strong_baseline.audit import audit
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
