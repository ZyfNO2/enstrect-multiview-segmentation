import urllib.request
import json
import os

packages = ['gdown', 'addict', 'alphashape', 'pc-skeletor', 'pyntcloud', 'ezdxf', 'tensordict']
index_url = 'https://pypi.org/pypi/{}/json'
dest = r'G:\Zed\ENSTRECTtest\wheels'
os.makedirs(dest, exist_ok=True)

for pkg in packages:
    try:
        url = index_url.format(pkg)
        print(f'Checking {pkg}...')
        resp = urllib.request.urlopen(url, timeout=15)
        data = json.loads(resp.read())
        version = list(data['releases'].keys())[-1]
        print(f'  Latest: {version}')

        files = data['releases'][version]
        whl = [f for f in files if f['filename'].endswith('.whl') and ('cp38-win_amd64' in f['filename'] or 'cp38-none' in f['filename'])]
        if not whl:
            whl = [f for f in files if f['filename'].endswith('.whl') and 'py3-none-any' in f['filename']]
        if not whl:
            whl = [f for f in files if f['filename'].endswith('.whl') and 'any' in f['filename']]
        
        if whl:
            dl = whl[0]
            fname = dl['filename']
            print(f'  Downloading: {fname}')
            fpath = os.path.join(dest, fname)
            with open(fpath, 'wb') as fout:
                fout.write(urllib.request.urlopen(dl['url'], timeout=120).read())
            print(f'  OK: {fname} ({os.path.getsize(fpath)} bytes)')
        else:
            available = [f['filename'] for f in files if f['filename'].endswith('.whl')]
            print(f'  No suitable wheel. Available cp38: {[a for a in available if "cp38" in a][:3]}')
    except Exception as e:
        print(f'  ERROR {pkg}: {type(e).__name__}: {e}')

print('\nDone! Wheels saved to:', dest)
