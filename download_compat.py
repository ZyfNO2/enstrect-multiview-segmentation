import urllib.request
import json
import os

dest = r'G:\Zed\ENSTRECTtest\wheels'

def download_version(pkg, version):
    url = f'https://pypi.org/pypi/{pkg}/json'
    print(f'Downloading {pkg}=={version}...')
    resp = urllib.request.urlopen(url, timeout=15)
    data = json.loads(resp.read())
    files = data['releases'].get(version, [])
    
    for f in files:
        fname = f['filename']
        if not fname.endswith('.whl'):
            continue
        if 'cp38-win' in fname or ('cp38' in fname and 'win' in fname):
            print(f'  Exact match: {fname}')
        elif 'py3-none-any' in fname or 'py2.py3-none-any' in fname:
            print(f'  Any platform: {fname}')
        elif 'cp38' in fname:
            print(f'  cp38: {fname}')
        else:
            continue
        
        fpath = os.path.join(dest, fname)
        with open(fpath, 'wb') as fout:
            fout.write(urllib.request.urlopen(f['url'], timeout=120).read())
        print(f'  OK ({os.path.getsize(fpath)} bytes)')
        return True
    
    available = [f['filename'] for f in files if f['filename'].endswith('.whl')]
    print(f'  No suitable whl. Available: {available[:5]}')
    return False

targets = [
    ('gdown', '5.2.0'),
    ('ezdxf', '1.3.5'),
]

for pkg, ver in targets:
    download_version(pkg, ver)

print('\nDone!')
