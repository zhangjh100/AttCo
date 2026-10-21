"""Apply documented import isolation and optional-weight support to vendored sources."""
from pathlib import Path
import re
for repo in ('emcad','cascade','effidec3d','efficientmednext'):
 for path in (Path('third_party')/repo).rglob('*.py'):
  source=path.read_text(encoding='utf-8-sig')
  source=re.sub(r'(?m)^from (lib|networks)(\.| import)',rf'from third_party.{repo}.\1\2',source)
  if repo=='cascade' and path.name=='networks.py' and "        path = './pretrained_pth/pvt/pvt_v2_b2.pth'" in source:
   source=source.replace('def __init__(self, n_class=1):','def __init__(self, n_class=1, pretrained_path=None):',1)
   start=source.index("        path = './pretrained_pth/pvt/pvt_v2_b2.pth'")
   end=source.index('        # decoder initialization',start)
   source=source[:start]+'''        if pretrained_path is not None:
            save_model = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            model_dict = self.backbone.state_dict()
            model_dict.update({k: v for k, v in save_model.items() if k in model_dict})
            self.backbone.load_state_dict(model_dict)

'''+source[end:]
  if repo=='cascade' and path.name=='decoders.py':
   source='\n'.join(line for line in source.splitlines() if line not in ('from PIL import Image','import cv2','import seaborn as sns','import matplotlib.pyplot as plt','import scipy.misc     '))+'\n'
  path.write_text(source,encoding='utf-8')

p=Path('third_party/mt_unet/model/MTUNet.py')
s=p.read_text(encoding='utf-8-sig').replace('torch.zeros((b, c, right_size, right_size))','x.new_zeros((b, c, right_size, right_size))').replace('.cuda()', '.to(x.device)')
p.write_text(s,encoding='utf-8')
