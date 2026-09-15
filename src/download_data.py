"""Download source bytes, preserve hashes, and extract only known dataset members."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, shutil, urllib.request, zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
SOURCES = [
    ('ibm_telco.csv', 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv', 'IBM/telco-customer-churn-on-icp4d', 'Repository Apache-2.0; retain source attribution'),
    ('iranian_churn.zip', 'https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip', 'https://doi.org/10.24432/C5JW3Z', 'CC BY 4.0'),
    ('online_retail_ii.zip', 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip', 'https://doi.org/10.24432/C5CG6D', 'CC BY 4.0'),
]

def download(item):
    name, url, source, license_note = item
    path = RAW / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as f:
            shutil.copyfileobj(response, f)
        tmp.replace(path)
    result = dict(file=name, url=url, source=source, license=license_note,
                  bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if name.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            wanted = [n for n in z.namelist() if n.lower().endswith(('.csv', '.xlsx')) and not n.startswith('__MACOSX')]
            result['members'] = wanted
            for member in wanted:
                dest = RAW / ('iranian_churn.csv' if member.lower().endswith('.csv') else 'online_retail_ii.xlsx')
                with z.open(member) as f, dest.open('wb') as out:
                    shutil.copyfileobj(f, out)
                result.setdefault('extracted', []).append(dict(file=dest.name, bytes=dest.stat().st_size,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print(name, result['bytes'], flush=True)
    return result

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(download, SOURCES))
    manifest = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sources': rows}
    (ROOT / 'data/source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
