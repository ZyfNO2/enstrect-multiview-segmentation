import urllib.request
import json
import os

dest = r'G:\Zed\ENSTRECTtest\wheels'
os.makedirs(dest, exist_ok=True)

# SimpleITK 需要平台特定的 wheel
url = 'https://pypi.org/pypi/SimpleITK/json'
print('Checking SimpleITK...')
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())

# 找 Windows cp38 版本
versions = list(data['releases'].keys())
for ver in reversed(versions):
    files = data['releases'][ver]
    for f in files:
        fname = f['filename']
        # 找 Windows Python 3.8 64位版本
        if fname.endswith('.whl') and 'cp38' in fname and 'win_amd64' in fname:
            print(f'  Found {ver}: {fname}')
            fpath = os.path.join(dest, fname)
            with open(fpath, 'wb') as fout:
                fout.write(urllib.request.urlopen(f['url'], timeout=180).read())
            print(f'  Downloaded ({os.path.getsize(fpath)} bytes)')
            exit(0)

print('  No suitable wheel found')
