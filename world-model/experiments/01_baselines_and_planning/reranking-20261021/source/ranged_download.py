"""Task-local verified range transfer for the already pinned official asset."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

spec = json.loads(Path('manifests/official_dp.json').read_text(encoding='utf-8'))['checkpoint']
root = Path('checkpoints/official-dp')
prefix = root/'original.ckpt.part'
target = root/'original.ckpt.ranged'
final = root/'original.ckpt'
size = spec['bytes']
start = prefix.stat().st_size
assert 0 < start < size and not target.exists() and not final.exists()
with prefix.open('rb') as src, target.open('xb') as dst:
    shutil.copyfileobj(src, dst, 8*1024**2)
    dst.truncate(size)
lock = threading.Lock()
completed = start
chunk = 16*1024**2
def transfer(offset):
    global completed
    end = min(size, offset+chunk)-1
    for attempt in range(3):
        try:
            req = urllib.request.Request(spec['url'], headers={'Range':f'bytes={offset}-{end}'})
            with urllib.request.urlopen(req,timeout=60) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f'bytes {offset}-{end}/{size}'
                data = response.read(end-offset+2)
            assert len(data) == end-offset+1
            with lock:
                with target.open('r+b') as stream:
                    stream.seek(offset)
                    stream.write(data)
                completed += len(data)
                if (completed-start)//(128*1024**2) != (completed-start-len(data))//(128*1024**2) or completed==size:
                    print(json.dumps({'downloaded_bytes':completed,'total_bytes':size}),flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)
print(json.dumps({'resuming_bytes':start,'chunk_bytes':chunk,'workers':6}),flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:
    for future in as_completed([pool.submit(transfer,offset) for offset in range(start,size,chunk)]):
        future.result()
digest=hashlib.sha256()
with target.open('rb') as stream:
    for data in iter(lambda:stream.read(8*1024**2),b''):
        digest.update(data)
assert digest.hexdigest()==spec['sha256'],digest.hexdigest()
target.rename(final)
print(json.dumps({'status':'verified','bytes':size,'sha256':digest.hexdigest()}),flush=True)
