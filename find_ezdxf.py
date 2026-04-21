import urllib.request
import json
import os

dest = r'G:\Zed\ENSTRECTtest\wheels'
url = 'https://pypi.org/pypi/ezdxf/json'
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())

targets = ['1.1.0', '1.0.3', '0.18.0', '0.17.2', '0.16.0']

for ver in targets:
    files = data['releases'].get(ver, [])
    whls = [f for f in files if f['filename'].endswith('.whl')]
    
    for w in whls:
        fname = w['filename']
        if 'py3-none-any' in fname or 'py2.py3-none-any' in fname:
            print(f'Found {ver}: {fname}')
            fpath = os.path.join(dest, fname)
            with open(fpath, 'wb') as fout:
                fout.write(urllib.request.urlopen(w['url'], timeout=60).read())
            print(f'Downloaded: {os.path.getsize(fpath)} bytes')
            exit(0)

print('No any-compatible version found')
