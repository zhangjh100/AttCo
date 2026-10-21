"""Package the current training implementation for transfer to a Linux server."""
from pathlib import Path
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
def main():
    names = ['main.py','main_single.py','test.py','dataset.py','comparison_data.py',
             'transforms.py','losses.py','metrics.py','utils.py','requirements.txt',
             'requirements-baselines.txt','ACDC_Training_5folds.csv','BASELINES.md']
    files = [ROOT / name for name in names]
    for folder in ('models','third_party','tests'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts
                     and (p.suffix in ('.py','.json','.yaml','.yml') or p.name.lower()=='license'))
    files.append(ROOT / 'scripts' / 'smoke_baselines.py')
    files = sorted(set(files))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    destination = ROOT / 'output' / 'attco_training_fix.tar.gz'
    destination.parent.mkdir(exist_ok=True)
    with tarfile.open(destination,'w:gz') as archive:
        for p in files:
            archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        content = json.dumps(manifest,indent=2).encode()
        info = tarfile.TarInfo('TRAINING_RELEASE_SHA256.json'); info.size=len(content)
        archive.addfile(info,io.BytesIO(content))
    with tarfile.open(destination,'r:gz') as archive:
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected
    print(f'Archive: {destination}\nFiles: {len(files)}\nSize: {destination.stat().st_size} bytes')
    print('SHA256: '+hashlib.sha256(destination.read_bytes()).hexdigest())
if __name__=='__main__': main()
