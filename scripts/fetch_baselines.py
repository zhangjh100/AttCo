import urllib.request, json, pathlib, concurrent.futures
repos={'mamba_unet':'ziyangwang007/Mamba-UNet','emcad':'SLDGroup/EMCAD','cascade':'SLDGroup/CASCADE','effidec3d':'SLDGroup/EffiDec3D','efficientmednext':'SLDGroup/EfficientMedNeXt','mt_unet':'Dootmaan/MT-UNet','transunet':'Beckschen/TransUNet'}
def get(item):
 name,repo=item
 req=lambda url: urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AttCo-baselines'}),timeout=60).read()
 info=json.loads(req('https://api.github.com/repos/'+repo+'/commits/HEAD'))
 sha=info['sha']; tree=json.loads(req('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1'))
 dest=pathlib.Path('third_party')/name; dest.mkdir(exist_ok=True)
 entries=[x['path'] for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith('.py') or 'license' in x['path'].lower() or x['path'].lower().endswith(('.yaml','.yml')) or x['path'].lower() in ('readme.md','requirements.txt'))]
 for path in entries:
  target=dest/path; target.parent.mkdir(exist_ok=True,parents=True)
  target.write_bytes(req(f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'))
 (dest/'UPSTREAM.json').write_text(json.dumps({'repository':'https://github.com/'+repo,'commit':sha},indent=2))
 return name,len(entries)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 for result in pool.map(get,repos.items()): print(result,flush=True)
