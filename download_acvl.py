import urllib.request
import json
import os

dest = r'G:\Zed\ENSTRECTtest\wheels'
os.makedirs(dest, exist_ok=True)

def download_package(pkg_name, version=None):
    url = f'https://pypi.org/pypi/{pkg_name}/json'
    print(f'Checking {pkg_name}...')
    resp = urllib.request.urlopen(url, timeout=15)
    data = json.loads(resp.read())
    
    if version:
        files = data['releases'].get(version, [])
    else:
        versions = list(data['releases'].keys())
        # 找 Python 3.8 兼容的版本
        for ver in reversed(versions):
            files = data['releases'][ver]
            whls = [f for f in files if f['filename'].endswith('.whl')]
            for w in whls:
                if 'py3' in w['filename'] or 'py2.py3' in w['filename']:
                    print(f'  Found version {ver}')
                    version = ver
                    break
            if version:
                break
    
    for f in files:
        fname = f['filename']
        if fname.endswith('.whl') and ('py3' in fname or 'py2.py3' in fname):
            print(f'  Downloading: {fname}')
            fpath = os.path.join(dest, fname)
            with open(fpath, 'wb') as fout:
                fout.write(urllib.request.urlopen(f['url'], timeout=120).read())
            print(f'  OK ({os.path.getsize(fpath)} bytes)')
            return True
    
    print(f'  No suitable wheel found')
    return False

# 下载 acvl-utils 和 batchgenerators
download_package('acvl-utils')
download_package('batchgenerators')

print('\nDone!')
