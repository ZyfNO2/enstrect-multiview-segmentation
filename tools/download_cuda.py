import urllib.request
import os

url = "https://developer.download.nvidia.com/compute/cuda/12.8.1/network/cuda_12.8.1_windows_network.exe"
dest = r"C:\Users\ZYF\Downloads\cuda_12.8.1_windows_network.exe"

print(f"Downloading CUDA 12.8.1 Network Installer...")
print(f"URL: {url}")
print(f"Destination: {dest}")

def progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    if percent % 10 == 0:
        print(f"Progress: {percent}%", end="\r")

try:
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\nDownload complete!")
    print(f"File size: {os.path.getsize(dest) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"\nDownload failed: {e}")
