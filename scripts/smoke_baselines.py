"""Forward/backward checks on official architectures, no patient data required."""
import argparse,json,time,gc
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.baselines import MODEL_REGISTRY,build_model,prediction_logits,segmentation_loss
from losses import CE_GeneralizedSoftDiceLoss

def main():
 p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=[n for n in MODEL_REGISTRY if n not in ('Wave','WaveDe')]); p.add_argument('--report',default='output/baseline_smoke.json'); args=p.parse_args()
 torch.set_num_threads(2); results=[]
 for name in args.models:
  start=time.time(); model=None
  try:
   model=build_model(name)
   spatial=(32,32,32) if MODEL_REGISTRY[name]=='3d' else ((224,224) if name in ('MT-UNet','TransUNet','Mamba-UNet') else (64,64))
   x=torch.randn(1,1,*spatial); y=torch.randint(0,4,(1,*spatial))
   model.train(); out=model(x); loss=segmentation_loss(CE_GeneralizedSoftDiceLoss(),out,y)
   assert torch.isfinite(loss); loss.backward()
   grads=[p.grad for p in model.parameters() if p.grad is not None]
   assert grads and all(torch.isfinite(g).all() for g in grads)
   del out,loss,grads
   model.eval()
   with torch.no_grad(): pred=model(x)
   assert pred.shape==(1,4,*spatial) and torch.isfinite(pred).all()
   result=dict(model=name,status='passed',shape=list(pred.shape),seconds=round(time.time()-start,2))
  except RuntimeError as exc:
   if name=='Mamba-UNet' and 'selective_scan' in str(exc): result=dict(model=name,status='blocked_missing_cuda_operator',detail=str(exc))
   else: raise
  results.append(result); print(json.dumps(result),flush=True)
  del model; gc.collect()
 dest=Path(args.report); dest.parent.mkdir(exist_ok=True,parents=True); dest.write_text(json.dumps(results,indent=2))
if __name__=='__main__': main()
