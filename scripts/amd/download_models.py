#!/usr/bin/env python3
"""Resume pinned inventory downloads and verify every byte before publication."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time


def verify(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes']:
        return False
    algorithm = 'sha256' if row.get('sha256') else 'sha1'
    digest = hashlib.new(algorithm)
    if algorithm == 'sha1':
        digest.update(f"blob {row['bytes']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest() == (row.get('sha256') or row['git_blob'])


def fetch(row):
    start = time.monotonic()
    target = Path(row['destination'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify(target, row):
        return dict(file=row['file'], destination=str(target), status='verified-existing', bytes=row['bytes'])
    if target.exists():
        raise RuntimeError(f'Existing file fails checksum: {target}')
    partial = target.with_name(target.name + '.partial')
    # curl resumes with HTTP Range. On failure retain the partial and report it.
    command = ['curl', '--fail', '--location', '--silent', '--show-error',
               '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
               '--speed-limit', '1024', '--speed-time', '120',
               '--continue-at', '-', '--output', str(partial), row['url']]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"curl {result.returncode}: {result.stderr[-600:]}")
    if not verify(partial, row):
        raise RuntimeError(f'Checksum/size mismatch: {partial}')
    partial.replace(target)
    return dict(file=row['file'], destination=str(target), status='downloaded-verified',
                bytes=row['bytes'], seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--exclude-prepared', action='store_true')
    parser.add_argument('--only-prepared', action='store_true')
    parser.add_argument('--only-sampling', action='store_true')
    args = parser.parse_args()
    inventory = json.loads(args.manifest.read_text())
    rows = [r for r in inventory['files'] if not (args.exclude_prepared and r.get('prepared'))
            and (not args.only_prepared or r.get('prepared'))
            and (not args.only_sampling or r.get('sampling_file'))]
    # Publish small configs quickly, then overlap the large independent components.
    rows.sort(key=lambda r: r['bytes'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(manifest=str(args.manifest), start=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  state='running', selected_files=len(rows), files=[])
    def save():
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.out)
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(file=row['file'], destination=row['destination'], status='failed', error=str(error))
            report['files'].append(result)
            save()
            print(json.dumps(dict(completed=len(report['files']), total=len(rows), **result)), flush=True)
    report['state'] = 'failed' if any(r['status']=='failed' for r in report['files']) else 'complete'
    report['end'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()
    return report['state'] != 'complete'


if __name__ == '__main__':
    raise SystemExit(main())
