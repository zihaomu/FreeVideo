#!/usr/bin/env python3
"""Create a pinned H3 download inventory and check space; never download weights."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path,
                        default=Path(os.environ.get('FV_STORAGE_ROOT', '/dc1/zihaomu/free_token_mapping')) / 'models')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('per_tensor', 'rowwise'), default='per_tensor')
    parser.add_argument('--all-quality-levels', action='store_true')
    args = parser.parse_args()
    from freevideo_engine.prepared_model import select, files
    from freevideo_engine.install_tuning import required_models
    from freevideo_engine.bootstrap import model_target
    from freevideo_engine.sampling_assets import install_files

    models = args.models.expanduser().absolute()
    models.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / 'freevideo_engine/model_files.json').read_text(encoding='utf-8'))
    prepared = select((12, 0), models, scale_granularity=args.variant)
    rows = required_models(catalog, prepared) + files(prepared) + install_files(args.all_quality_levels)
    for row in rows:
        row['destination'] = str(model_target(row, models / 'vdn', models / 'encoder', Path(prepared['directory'])))
        row['url'] = f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{quote(row['file'], safe='/')}"
    components = defaultdict(int)
    for row in rows:
        component = ('sampling_tables' if row.get('sampling_file') else 'prepared_transformer'
                     if row.get('prepared') else 'latent_upscaler' if row.get('role') == 'latent_upscaler'
                     else 'text_encoder' if row['repo'].startswith('t8star/') else 'base_decoders_and_configs')
        components[component] += row['bytes']
    total = sum(row['bytes'] for row in rows)
    free = shutil.disk_usage(models).free
    required_free = total + 15 * 2**30
    report = dict(
        freevideo_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        model_root=str(models), resolved_model_root=str(models.resolve()),
        variant=args.variant, variant_is_provisional=True,
        variant_note='Selected explicitly for investigation; AMD compute and scale policy must be validated before weight download',
        prepared=prepared, components_bytes=dict(components), file_count=len(rows),
        total_bytes=total, total_gib=total / 2**30, free_bytes=free,
        recommended_free_bytes=required_free, storage_ready=free >= required_free,
        weights_downloaded=False, files=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
