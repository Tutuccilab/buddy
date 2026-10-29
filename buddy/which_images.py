"""
Before a batch step: skip images that were already processed with the same settings, and ask before
overwriting images that were processed with other settings.

    todo = select_images(exp, 'read')       # step 1, from exp.images        -> list of image dicts
    todo = select_images(exp, 'segment')    # step 2, from exp.json_files    -> list of img.json paths
    todo = select_images(exp, 'spots')      # step 3, from exp.json_files    -> list of img.json paths

Each image falls in one of three groups:
    new                       not processed by this step yet                      -> processed
    same settings             processed with the settings in the config file      -> skipped
    other settings            processed with different settings                   -> what changed is listed,
                                                                                     then overwritten or skipped
What to do with the last group: "if_already_processed" in the config file (top level):
    "ask" (default)  one yes/no question for all of them
    "skip"           never overwrite (e.g. an unattended run)
    "overwrite"      always overwrite

The settings a step used are recorded in img.json (results -> settings -> step) by Image.record_settings().
For images processed by an older buddy version they are reconstructed from img.json where possible.

Re-running step 2 (segmentation) makes new cell masks, so the image's corrected/ folder (aligned masks,
bud masks and hand corrections from step 4) no longer fits; it is moved to corrected_replaced_<date-time>/
and rebuilt by step 3.
"""
import json
import time
from pathlib import Path

from buddy.config import POLICIES

STAGE_OF_STEP = {'read': 'configure', 'segment': 'segment', 'spots': 'spots'}
STAGE_ORDER = ['configure', 'segment', 'spots']
STEP_NAMES = {'read': 'step 1 (read)', 'segment': 'step 2 (segment)', 'spots': 'step 3 (spots)'}


# ---------------------------------------------------------------------------------------------
# the settings that change the result of each step
# ---------------------------------------------------------------------------------------------
def _segmentation_changes(p, seg):
    """The Omnipose part of the "segmentation" block where it changes the result compared to the defaults."""
    from buddy.config import SEGMENTATION_DEFAULTS, omnipose_overrides
    changed = {k: v for k, v in seg.items() if k not in ('cells', 'nuclei', 'use_preset') and v != SEGMENTATION_DEFAULTS.get(k)}
    if seg and p.get('organism'):
        cells, nuclei = omnipose_overrides(p, seg)
        changed.update({k: v for k, v in (('cells', cells), ('nuclei', nuclei)) if v})
    return changed


def _decomposition_changes(p):
    from buddy.config import DECOMPOSITION_DEFAULTS
    d = p.get('decomposition') or {}
    return {k: v for k, v in d.items() if v != DECOMPOSITION_DEFAULTS.get(k)}


def step_settings(exp, step):
    """Settings in the config file that change the result of a step (GPU use does not)."""
    p = exp.parameters
    if step == 'read':
        return {'channels': {ch: c.get('filter') for ch, c in exp.channels.items()}}
    if step == 'segment':
        seg = getattr(exp, 'segmentation', None) or {}
        settings = {'organism': p.get('organism'), 'min_size': p.get('min_size'),
                    'nuclei_min_size': p.get('nuclei_min_size', 10), 'min_cell_area': p.get('min_cell_area', 20),
                    'expand_px': p.get('expand_px', 0), 'remove_edge_cells': bool(p.get('remove_edge_cells', False))}
        # the "segmentation" block (buddy >= 0.5.0): only recorded when it differs from the defaults, so images
        # processed by older versions are not reported as processed with other settings
        changed = _segmentation_changes(p, seg)
        if changed:
            settings['segmentation'] = changed
        return settings
    if step == 'spots':
        bud = getattr(exp, 'bud_classification', {}) or {}
        bud_model = 'off'
        if bud.get('enabled'):
            bud_model = bud.get('model')
            if bud_model == 'with_ASH1CLB2':
                bud_model += f" ({bud.get('ash_channel')})"
        settings = {'sigma': list(p.get('sigma', [])), 'scale': list(p.get('scale', [])),
                    'spot_radius': list(p.get('spot_radius', [])), 'patch_size': list(p.get('patch_size', [])),
                    'thresholds': {ch: ('auto' if c.get('threshold') is None else c.get('threshold'))
                                   for ch, c in exp.channels.items() if ch != 'DAPI'},
                    'bud_model': bud_model}
        if _decomposition_changes(p):        # only when changed: images of older versions are not reported as different
            settings['decomposition'] = _decomposition_changes(p)
        return settings
    raise ValueError(f"unknown step {step!r} (use 'read', 'segment' or 'spots')")


def _recorded_settings(meta, step, folder, exp):
    """Settings an image was processed with: recorded in img.json, or reconstructed for older images."""
    recorded = meta.get('results', {}).get('settings', {}).get(step)
    if recorded is not None:
        return recorded, False
    p = meta.get('parameters', {})
    if step == 'read':        # older images: which channel files exist (filter names were not recorded)
        current = step_settings(exp, 'read')['channels']
        return {'channels': {ch: f for ch, f in current.items() if (folder / f'{ch}.tif').exists()}}, True
    if step == 'segment':
        seg = getattr(exp, 'segmentation', None) or {}
        settings = {'organism': p.get('organism'), 'min_size': p.get('min_size'),
                    'nuclei_min_size': p.get('nuclei_min_size', 10), 'min_cell_area': p.get('min_cell_area', 20),
                    'expand_px': p.get('expand_px', 0), 'remove_edge_cells': bool(p.get('remove_edge_cells', False))}
        # the "segmentation" block (buddy >= 0.5.0): only recorded when it differs from the defaults, so images
        # processed by older versions are not reported as processed with other settings
        changed = _segmentation_changes(p, seg)
        if changed:
            settings['segmentation'] = changed
        return settings, True
    results = meta.get('results', {})
    bud = results.get('bud_classification')
    return {'sigma': list(p.get('sigma', [])), 'scale': list(p.get('scale', [])),
            'spot_radius': list(p.get('spot_radius', [])), 'patch_size': list(p.get('patch_size', [])),
            'thresholds': {ch: v.get('threshold') for ch, v in results.items()
                           if isinstance(v, dict) and 'colormap' in v},
            'bud_model': bud.get('model', 'off') if isinstance(bud, dict) else 'off'}, True


def _differences(old, new, prefix=''):
    """List of 'name: old -> new' for every setting that differs."""
    out = []
    for key in sorted(set(old) | set(new), key=str):
        a, b = old.get(key, '-'), new.get(key, '-')
        if isinstance(a, dict) and isinstance(b, dict):
            out += _differences(a, b, prefix=f'{prefix}{key} ')
        elif a != b:
            out.append(f'{prefix}{key}: {a} → {b}')
    return out


# ---------------------------------------------------------------------------------------------
# choose the images to process
# ---------------------------------------------------------------------------------------------
def _json_of(exp, item, step):
    if step == 'read':
        return Path(exp.outputdir) / exp.stem_of(item['vsi_file']) / 'img.json'
    return Path(item)


def _has_corrections(folder):
    return (folder / 'corrected').is_dir() and any((folder / 'corrected').iterdir())


def _ask(question):
    try:
        return input(question).strip().lower() in ('y', 'yes', 'j', 'ja')
    except (EOFError, OSError):          # no one to answer (e.g. a script): do not overwrite
        print('  (no answer possible - not overwriting)')
        return False


def select_images(exp, step, candidates=None, ask=None):
    """
    The images a batch step should process (see the module docstring).
    candidates: default exp.images.values() for 'read', exp.json_files for 'segment' / 'spots'.
    ask: how to ask the overwrite question - None = typed y/n answer; or a function that gets the
         text (what changed + the question) and returns True / False (e.g. a yes/no dialog in napari).
    """
    if candidates is None:
        candidates = list(exp.images.values()) if step == 'read' else list(exp.json_files)
    policy = getattr(exp, 'if_already_processed', 'ask')
    if policy not in POLICIES:
        raise ValueError(f'"if_already_processed" must be one of {POLICIES}, not {policy!r}')
    current = step_settings(exp, step)
    stage = STAGE_ORDER.index(STAGE_OF_STEP[step])

    new, same, changed = [], [], []           # changed: (item, folder, differences, reconstructed)
    for item in candidates:
        path = _json_of(exp, item, step)
        if not path.is_file():
            new.append(item)
            continue
        meta = json.load(open(path))
        last = meta.get('last_run')
        if last not in STAGE_ORDER or STAGE_ORDER.index(last) < stage:
            new.append(item)
            continue
        recorded, reconstructed = _recorded_settings(meta, step, path.parent, exp)
        diff = _differences(recorded, current)
        (changed.append((item, path.parent, diff, reconstructed)) if diff else same.append(item))

    name = STEP_NAMES[step]
    lines = [f'{name}: {len(new)} new, {len(same)} already processed with these settings (skipped), '
             f'{len(changed)} processed with other settings']
    overwrite = []
    if changed:
        all_diffs = sorted({d for _, _, diff, _ in changed for d in diff})
        lines.append('  what changed: ' + '; '.join(all_diffs[:12]) + (' ...' if len(all_diffs) > 12 else ''))
        if any(r for *_, r in changed):
            lines.append('  (some images were processed by an older buddy version: their settings are partly reconstructed)')
        if step == 'read':
            lines.append('  re-reading resets these images: steps 2 and 3 run again for them, and their corrected/ '
                         'masks are moved aside in step 2')
        elif step == 'segment':
            n = sum(_has_corrections(folder) for _, folder, _, _ in changed)
            if n:
                lines.append(f'  {n} of them have corrected masks (step 3 alignment / bud model / your step-4 '
                             f'corrections): these no longer fit new cell masks and are moved to '
                             f'corrected_replaced_<date-time>/')
        elif step == 'spots':
            lines.append('  hand-corrected masks from step 4 are kept')
    print('\n'.join(lines))
    if changed:
        question = f'Overwrite these {len(changed)} image(s)?'
        if policy == 'overwrite':
            yes = True
        elif policy == 'skip':
            yes = False
        elif ask is None:
            yes = _ask(f'  {question} [y/N] ')
        else:
            yes = bool(ask('\n'.join(lines) + '\n\n' + question))
        if yes:
            overwrite = [item for item, *_ in changed]
            print(f'  -> overwriting {len(overwrite)} image(s)')
        else:
            print('  -> skipping them (to overwrite: answer yes, or set "if_already_processed": "overwrite")')

    todo = new + overwrite
    if step == 'segment':                 # new cell masks: the old corrected/ masks no longer fit
        stamp = time.strftime('%Y%m%d-%H%M%S')
        for item in todo:
            folder = _json_of(exp, item, step).parent
            if _has_corrections(folder):
                (folder / 'corrected').rename(folder / f'corrected_replaced_{stamp}')
                print(f'  moved {folder.name}/corrected -> corrected_replaced_{stamp}')
    print(f'  processing {len(todo)} image(s)')
    return todo
