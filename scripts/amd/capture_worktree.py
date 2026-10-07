#!/usr/bin/env python3
"""Preserve source hashes, tracked patch and new files for an experiment stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(os.environ.get('FV_STORAGE_ROOT', '/dc1/zihaomu/free_token_mapping'))/'experiments/freevideo-r9700'


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name.replace('-', '').replace('_', '').isalnum():
        parser.error('Use letters, digits, underscores or hyphens')
    reports = EXPERIMENT/'reports'
    receipt = reports/f'{args.name}-code-receipt.json'
    patch = reports/f'{args.name}-tracked-changes.patch'
    archive = reports/f'{args.name}-worktree-files.zip'
    if any(p.exists() for p in (receipt, patch, archive)):
        raise FileExistsError('Snapshot name already exists')
    names = sorted(set(git('ls-files', '-z', '--modified', '--others', '--exclude-standard').decode().split('\0'))-{'', '.gitignore'})
    report = dict(created=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                  head=git('rev-parse', 'HEAD').decode().strip(),
                  branch=git('branch', '--show-current').decode().strip(),
                  excluded_preexisting_change='.gitignore',
                  source_files={}, profiles={}, archive=str(archive), tracked_patch=str(patch))
    patch.write_bytes(git('diff', '--', '.', ':(exclude).gitignore'))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for name in names:
            path = ROOT/name
            if path.is_file():
                contents = path.read_bytes()
                report['source_files'][name] = dict(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
                output.writestr(name, contents)
        for path in sorted((EXPERIMENT/'prepared/profiles').glob('*.json')):
            contents = path.read_bytes()
            report['profiles'][path.name] = dict(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
            output.writestr('experiment-profiles/'+path.name, contents)
    report['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    report['tracked_patch_sha256'] = hashlib.sha256(patch.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(receipt=str(receipt), source_files=len(report['source_files']), profiles=len(report['profiles']))))


if __name__ == '__main__':
    main()
