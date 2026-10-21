import urllib.request,json,pathlib,concurrent.futures
repos={'mamba_unet':('ziyangwang007/Mamba-UNet',('code/networks/mamba_sys.py','code/networks/vision_mamba.py','code/configs/vmamba_tiny.yaml')), 'emcad':('SLDGroup/EMCAD',('lib/',)), 'cascade':('SLDGroup/CASCADE',('lib/',)), 'effidec3d':('SLDGroup/EffiDec3D',('networks/MedNeXt/mednextv1/','main_train_BTCV_TU.py')), 'efficientmednext':('SLDGroup/EfficientMedNeXt',('networks/MedNeXt/mednextv1/','main_train_efficientmednext.py')), 'mt_unet':('Dootmaan/MT-UNet',('model/MTUNet.py',)), 'transunet':('Beckschen/TransUNet',('networks/',))}
def request(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AttCo-baselines'}),timeout=90).read()
def fetch(item):
 name,(repo,prefixes)=item
 manifest=pathlib.Path('third_party')/name/'UPSTREAM.json'
 sha=json.loads(manifest.read_text())['commit'] if manifest.exists() else json.loads(request('https://api.github.com/repos/'+repo+'/commits/HEAD'))['sha']
 tree=json.loads(request('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1'))['tree']
 dest=pathlib.Path('third_party')/name; dest.mkdir(exist_ok=True)
 paths=[x['path'] for x in tree if x['type']=='blob' and (any(x['path'].startswith(p) for p in prefixes) or x['path'].lower() in ('license','readme.md','requirements.txt')) and (x['path'].endswith(('.py','.yaml','.md','.txt')) or x['path'].lower()=='license')]
 def save(path):
  target=dest/path; target.parent.mkdir(exist_ok=True,parents=True)
  if not target.exists(): target.write_bytes(request(f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool: list(pool.map(save,paths))
 (dest/'UPSTREAM.json').write_text(json.dumps({'repository':'https://github.com/'+repo,'commit':sha},indent=2))
 return name,len(paths)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  for result in pool.map(fetch,repos.items()): print(result,flush=True)

 # Reapply local compatibility patches after restoring upstream files.
 import subprocess,sys
 subprocess.run([sys.executable,'scripts/patch_baselines.py'],check=True)
