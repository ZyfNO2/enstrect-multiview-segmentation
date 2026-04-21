import urllib.request
import json
import os

dest = r'G:\Zed\ENSTRECTtest\wheels'
os.makedirs(dest, exist_ok=True)

packages = [('nibabel', '4.0.2'), ('nibabel', '3.2.2')]

for pkg, ver in packages:
    url = f'https://pypi.org/pypi/{pkg}/json'
    print(f'Checking {pkg}=={ver}...')
    resp = urllib.request.urlopen(url, timeout=15)
    data = json.loads(resp.read())
    
    files = data['releases'].get(ver, [])
    for f in files:
        fname = f['filename']
        if fname.endswith('.whl') and ('py3-none-any' in fname or 'py2.py3-none-any' in fname):
            print(f'  Found: {fname}')
            fpath = os.path.join(dest, fname)
            with open(fpath, 'wb') as fout:
                fout.write(urllib.request.urlopen(f['url'], timeout=120).read())
            print(f'  Downloaded ({os.path.getsize(fpath)} bytes)')
            break

print('\nDone!')
