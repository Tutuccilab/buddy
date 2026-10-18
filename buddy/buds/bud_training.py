"""
Training the bud model (used by 05_train_bud_model.ipynb).

The model ("model D", see bud_model.py) learns from images you corrected by hand in step 4 of the bulk-processing
notebook:
  corrected/DIC_corrected_masks.tif   cells, mother + bud share one label
  corrected/bud_corrected_masks.tif   1 = bud
together with the Omnipose cells (DIC_masks_pp.tif, and the raw DIC_masks.tif when present), the DIC image, the
DAPI stack and (for the 'with_ASH1CLB2' model) the ASH1/CLB2 stack of each image folder.

One training example = one Omnipose cell: the crop around it with a soft dot on its centre; the model has to outline
the cell's body, its bud (if any) and say whether it is unbudded / budded / itself a bud. Small raw Omnipose objects
that were not curated but touch the cell are "ignore regions": they may be buds the curation missed, so the bud map
is not graded there.

Three steps:
  1. prepare_image(folder)  -> one training file per image (aligned, normalised channels + a table with
                               one row per Omnipose cell: unbudded / budded / bud / uncurated / fragment)
  2. train(...)             -> network weights (.pt) + <name>.json with the settings
  3. evaluate(...)          -> pairing scores on images that were not used for training
"""
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io

from buddy.buds.bud_model import (
    CLASSES,
    CROP,
    MODELS,
    build_network,
    centre_dot,
    crop_around,
    n_input_channels,
    normalise,
    normalise_ash,
    pad_to,
    pick_device,
)
from buddy.buds.bud_tools import BUD
from buddy.buds.dic_orientation import TRANSFORMS, dic_light_direction, lighting_preserving_transforms, rotation_to_reference

DEFAULT_PIXEL_NM = 65.0   # pixel size when an image's img.json does not record it ("parameters" -> "scale")
PX_UM2 = (DEFAULT_PIXEL_NM / 1000) ** 2   # pixel area (um2) at that pixel size

log = logging.getLogger("buddy")
MIN_CELL_PX = 20          # smaller Omnipose objects are not used as training examples
UNCERTAIN_UM2 = 4.0       # raw Omnipose objects below this, uncurated and touching a cell -> ignore regions
TOUCH_PX = 3              # "touching": within this many pixels
LATE_M_ASH = 8.0          # late-M: bud with its own full nucleus AND ASH1/CLB2 >= 8x the median unbudded cell
SMALL_BUD_PX = 700        # buds smaller than this get more weight in the loss (up to 2x)
DOT_JITTER = 2            # px, training only: the dot is not always exactly on the centre
CELL_LOSS_WEIGHT = 0.5    # weight of the cell-map loss
DRAW_SHARE = {"budded": 0.40, "not mine": 0.25, "bud": 0.15, "unbudded": 0.20}
# strains left out per model by default: she2/she3 deletions, in which ASH1 mRNA is not transported to the bud
EXCLUDED_STRAINS = {"DIC_DAPI_only": (), "with_ASH1CLB2": ("YET563", "YET572")}


# ---------------------------------------------------------------------------------------------
# 1. preparing the training data
# ---------------------------------------------------------------------------------------------
def find_shift(moving, fixed, max_shift=80):
    """(dy, dx) that best moves the foreground of `moving` onto the foreground of `fixed` (FFT cross-correlation)."""
    a = np.zeros(fixed.shape, np.float32)
    a[:min(moving.shape[0], fixed.shape[0]), :min(moving.shape[1], fixed.shape[1])] = \
        (moving[:fixed.shape[0], :fixed.shape[1]] > 0)
    b = (fixed > 0).astype(np.float32)
    corr = np.real(np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))))
    corr = np.fft.fftshift(corr)
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    window = corr[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
    dy, dx = np.unravel_index(np.argmax(window), window.shape)
    return int(dy - max_shift), int(dx - max_shift)


def move(image, shape, dy, dx):
    """Image placed in an array of `shape`, moved by (dy, dx); also returns where it has data."""
    out = np.zeros(shape, image.dtype)
    valid = np.zeros(shape, bool)
    h, w = image.shape
    y0, x0, y1, x1 = max(0, dy), max(0, dx), min(shape[0], h + dy), min(shape[1], w + dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        valid[y0:y1, x0:x1] = True
    return out, valid


def late_m_cells(folder, ash_channel):
    """Curated cells whose bud already has its own full-size nucleus and high ASH1/CLB2 (late mitosis). Only recorded."""
    csv = Path(folder) / f"{ash_channel}.csv"
    if not csv.is_file():
        return set()
    t = pd.read_csv(csv)
    needed = {"area_bud", "total_RNAs", "nuclei_bud_area", "nuclei_bud_mean_intensity", "mother_nuclei_area", "mother_nuclei_mean_intensity"}
    if not needed <= set(t.columns):
        return set()
    unbudded = t[t.area_bud == 0]
    if not len(unbudded):
        return set()
    ash_ref = max(unbudded.total_RNAs.median(), 1)
    nuc_area, nuc_int = unbudded.mother_nuclei_area.median(), unbudded.mother_nuclei_mean_intensity.median()
    own_nucleus = (t.nuclei_bud_area >= 0.5 * nuc_area) & (t.nuclei_bud_mean_intensity >= 0.7 * nuc_int)
    return set(t.image_cell_id[own_nucleus & (t.area_bud > 0) & (t.total_RNAs >= LATE_M_ASH * ash_ref)].astype(int))


def _object_pixels(label_image):
    """{label: sorted flat pixel indices}"""
    W = label_image.shape[1]
    out = {}
    for label, box in enumerate(ndi.find_objects(label_image), start=1):
        if box is not None:
            yy, xx = np.nonzero(label_image[box] == label)
            out[label] = np.sort((yy + box[0].start) * W + (xx + box[1].start))
    return out


def label_cells(omni, cells, bud, late_m, px_um2=PX_UM2):
    """
    One row per Omnipose cell, matched against your corrections:
      budded / unbudded : >= 50 % of the object lies on one curated cell and the outlines agree (IoU >= 0.5)
      bud               : >= 50 % of the object lies on a curated bud (it is a bud of another cell)
      fragment          : on a curated cell, but outlines disagree (over- / under-segmented)
      uncurated         : < 20 % on curated cells (not a cell, or a cell you removed)
    Only budded / unbudded / bud are trained on.
    """
    shape = cells.shape
    is_bud = bud
    body = np.where(is_bud, 0, cells)
    body_area = np.bincount(body.ravel(), minlength=cells.max() + 1)
    cell_area = np.bincount(cells.ravel(), minlength=cells.max() + 1)
    bud_px = np.bincount(np.where(is_bud, cells, 0).ravel(), minlength=cells.max() + 1)
    bud_of = np.where(is_bud, cells, 0)
    rows = []
    for label, box in enumerate(ndi.find_objects(omni), start=1):
        if box is None:
            continue
        me = omni[box] == label
        n = int(me.sum())
        if n < MIN_CELL_PX:
            continue
        on_cells = cells[box][me]
        on_curated, on_bud = float((on_cells > 0).mean()), float(is_bud[box][me].mean())
        cell = int(np.bincount(on_cells[on_cells > 0]).argmax()) if (on_cells > 0).any() else 0
        kind, iou = "fragment", 0.0
        if on_bud >= 0.5:
            kind = "bud"
        elif on_curated >= 0.5 and cell:
            inter = int(((body[box] == cell) & me).sum())
            iou_body = inter / max(n + body_area[cell] - inter, 1)
            inter_whole = int(((cells[box] == cell) & me).sum())             # an object covering mother + bud
            iou_whole = inter_whole / max(n + int(cell_area[cell]) - inter_whole, 1)
            iou = max(iou_body, iou_whole)
            if iou >= 0.5:
                kind = "budded" if bud_px[cell] > 0 else "unbudded"
        elif on_curated < 0.2:
            cell, kind = 0, "uncurated"
        # "not mine": the object touches the curated bud of another cell
        wide = tuple(slice(max(s.start - 6, 0), s.stop + 6) for s in box)
        m = omni[wide] == label
        ring = ndi.binary_dilation(m, iterations=5) & ~m
        neighbour_bud = bool(set(np.unique(bud_of[wide][ring]).tolist()) - {0, cell})
        edge = box[0].start == 0 or box[1].start == 0 or box[0].stop >= shape[0] or box[1].stop >= shape[1]
        yy, xx = np.nonzero(me)
        rows.append(dict(label=label, cy=int(yy.mean() + box[0].start), cx=int(xx.mean() + box[1].start), area_um2=n * px_um2,
                         kind=kind, cell=cell, iou=round(iou, 3), bud_px=int(bud_px[cell]) if kind == "budded" else 0,
                         touches_other_bud=neighbour_bud, late_m=cell in late_m, edge=edge))
    return pd.DataFrame(rows)


def uncertain_objects(raw, cells, px_um2=PX_UM2):
    """
    Raw Omnipose objects that buddy's clean-up did not keep as cells and that you did not curate, but which touch a
    curated cell and are small (< UNCERTAIN_UM2): maybe a bud the curation missed. The bud map is not graded there.
    Returns the raw label image with only these objects kept.
    """
    if raw is None or not (raw > 0).any():
        return np.zeros(cells.shape, np.uint16)
    keep = np.zeros(raw.max() + 1, bool)
    grown = ndi.grey_dilation(cells, size=2 * TOUCH_PX + 1)
    for label, box in enumerate(ndi.find_objects(raw), start=1):
        if box is None:
            continue
        m = raw[box] == label
        n = int(m.sum())
        if n * px_um2 >= UNCERTAIN_UM2 or n < MIN_CELL_PX:
            continue
        if (cells[box][m] > 0).mean() >= 0.2:            # curated (part of a cell you kept)
            continue
        if (grown[box][m] > 0).any():                     # touches a curated cell
            keep[label] = True
    return np.where(keep[raw], raw, 0).astype(np.uint16)


def pixel_area_um2(meta):
    """Pixel area in um2 from an image's img.json (\"parameters\" -> \"scale\" = [z, y, x] in nm); DEFAULT_PIXEL_NM when missing."""
    scale = (meta.get("parameters") or {}).get("scale")
    if not scale or len(scale) < 3:
        return PX_UM2
    return (float(scale[1]) / 1000) * (float(scale[2]) / 1000)


def prepare_image(folder, out_dir, ash_channel="ASH1CLB2"):
    """
    Turn one corrected image folder into a training file <out_dir>/<name>.npz + <name>.csv.
    Everything is put in the frame of the corrected masks (= fluorescence frame).
    """
    folder, out_dir = Path(folder), Path(out_dir)
    day = folder.parents[2].name if len(folder.parents) > 2 else ""      # <day>/<strain replicate>/output/<image>
    name = f"{day}__{folder.parent.parent.name}__{folder.name}".replace(" ", "")   # the day too: names repeat between days
    if (out_dir / f"{name}.npz").exists():
        return name
    cells = io.imread(folder / "corrected" / "DIC_corrected_masks.tif").astype(np.int64)
    bud = io.imread(folder / "corrected" / "bud_corrected_masks.tif") == BUD
    shape = cells.shape
    omni = io.imread(folder / "DIC_masks_pp.tif")
    if not (omni > 0).any():
        log.warning(f"{folder}: no Omnipose cells - skipped"); return None

    meta = json.load(open(folder / "img.json")) if (folder / "img.json").exists() else {}
    shift = meta.get("results", {}).get("alignment", {}).get("mask_shift")
    if shift is None:
        shift = find_shift(omni, cells)                  # older folders: find it from the masks
    dy, dx = shift
    omni, _ = move(omni.astype(np.int64), shape, dy, dx)
    raw_file = folder / "DIC_masks.tif"
    raw = move(io.imread(raw_file).astype(np.int64), shape, dy, dx)[0] if raw_file.is_file() else None
    dic = io.imread(folder / "DIC.tif"); dic = dic[dic.shape[0] // 2] if dic.ndim == 3 else dic
    dic, valid = move(dic.astype(np.float32), shape, dy, dx)
    dapi = pad_to(io.imread(folder / "DAPI.tif").max(0), shape)
    x = np.stack([normalise(dic, valid), normalise(dapi, valid)])
    x[:, ~valid] = 0
    ash_file = folder / f"{ash_channel}.tif"
    ash = pad_to(normalise_ash(io.imread(ash_file).max(0)), shape) if ash_file.exists() else np.zeros(shape, np.float32)
    light_angle, light_consistency, n_cells = dic_light_direction(x[0], omni)
    rotation = rotation_to_reference(light_angle)              # measured per image, as for the shipped models
    if not light_consistency >= 0.9:
        log.warning(f"{name}: DIC lighting direction unclear (angle {light_angle:.0f} deg, consistency "
                    f"{light_consistency:.2f}, {n_cells} cells) - turned by {rotation} x 90 deg; check this image")

    px_um2 = pixel_area_um2(meta)
    table = label_cells(omni, cells, bud, late_m_cells(folder, ash_channel), px_um2)
    table.insert(0, "image", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", x=x.astype(np.float16), ash=ash.astype(np.float16), omni=omni.astype(np.uint16),
                        uncertain=uncertain_objects(raw, cells, px_um2), bud_cells=np.where(bud, cells, 0).astype(np.uint16),
                        cells=cells.astype(np.uint16), rotation=rotation,
                        light_angle=light_angle, light_consistency=light_consistency)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    return name


def load_table(data_dir, images):
    return pd.concat([pd.read_csv(Path(data_dir) / f"{n}.csv") for n in images], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# 2. training
# ---------------------------------------------------------------------------------------------
class CellCrops:
    """Training examples: one Omnipose cell -> (input crop, bud target, cell target, ignore mask, class, weight)."""

    def __init__(self, table, data_dir, uses_ash, augment=True, seed=0, crop=CROP):
        self.table = table.reset_index(drop=True)
        self.data_dir, self.uses_ash, self.augment, self.crop = Path(data_dir), uses_ash, augment, crop
        self.rng = np.random.default_rng(seed)
        self.images = {}
        self.ops = lighting_preserving_transforms()

    def image(self, name):
        if name not in self.images:
            self.images[name] = dict(np.load(self.data_dir / f"{name}.npz"))
        return self.images[name]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        import torch
        r = self.table.iloc[i]
        f = self.image(r.image)
        c = self.crop
        x, (y0, x0) = crop_around(f["x"].astype(np.float32), r.cy, r.cx, c)
        omni, _ = crop_around(f["omni"], r.cy, r.cx, c)
        cells, _ = crop_around(f["cells"], r.cy, r.cx, c)
        bud_cells, _ = crop_around(f["bud_cells"], r.cy, r.cx, c)
        uncertain, _ = crop_around(f["uncertain"], r.cy, r.cx, c)
        me = omni == r.label
        yy, xx = np.nonzero(me)
        cy, cx = (yy.mean(), xx.mean()) if len(yy) else (r.cy - y0, r.cx - x0)
        if self.augment:
            cy, cx = cy + self.rng.integers(-DOT_JITTER, DOT_JITTER + 1), cx + self.rng.integers(-DOT_JITTER, DOT_JITTER + 1)
        channels = [x[0], x[1], centre_dot(cy, cx, c)]
        if self.uses_ash:
            channels.append(crop_around(f["ash"].astype(np.float32), r.cy, r.cx, c)[0])
        # targets
        is_bud = bud_cells > 0
        bud = ((bud_cells == r.cell) & (r.kind == "budded")).astype(np.float32)
        if r.kind == "bud":
            cell = ((cells == r.cell) & is_bud).astype(np.float32)             # the bud itself
        else:
            cell = ((cells == r.cell) & ~is_bud).astype(np.float32)            # the cell body (without its bud)
        near = ndi.binary_dilation(me, iterations=TOUCH_PX)
        touching = set(np.unique(uncertain[near]).astype(int).tolist()) - {0}
        ignore = np.isin(uncertain, list(touching)) if touching else np.zeros(me.shape, bool)
        ignore &= (bud == 0) & (cell == 0)
        stack = np.stack(channels + [bud, cell, ignore.astype(np.float32)])
        k = int(f["rotation"])
        if k:
            stack = np.rot90(stack, k, (1, 2))
        if self.augment:
            stack = TRANSFORMS[self.ops[self.rng.integers(len(self.ops))]](stack).copy()
            stack[0] = stack[0] * self.rng.uniform(0.8, 1.25) + self.rng.uniform(-0.1, 0.1)
            stack[1] = stack[1] * self.rng.uniform(0.8, 1.25)
            stack[0] += self.rng.normal(0, self.rng.uniform(0, 0.05), stack[0].shape).astype(np.float32)
        weight = float(np.clip(np.sqrt(SMALL_BUD_PX / max(r.bud_px, 1)), 1, 2)) if r.kind == "budded" else 1.0
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        n = len(channels)
        return (t(stack[:n]), t(stack[n]), t(stack[n + 1]), t(stack[n + 2]), CLASSES.index(r.kind),
                torch.tensor(weight, dtype=torch.float32))


def training_rows(table, model_name, exclude_strains=None, exclude_late_m=False):
    """
    Which cells to learn from: unbudded / budded / bud cells away from the image edge.
    exclude_strains: strain names whose images are left out; None = the model's default (EXCLUDED_STRAINS: she2/she3
    for with_ASH1CLB2, none for DIC_DAPI_only). A strain matches as a whole name in the image name, any case
    (YET563 matches "240909__YET563BR3__..." and "yET563-...", but not YET5630).
    Late-M cells are kept unless exclude_late_m (keeping them scored better).
    """
    if exclude_strains is None:
        exclude_strains = EXCLUDED_STRAINS[model_name]
    t = table[table.kind.isin(CLASSES) & ~table.edge.astype(bool)]
    if exclude_late_m:
        t = t[~t.late_m.astype(bool)]
    for strain in exclude_strains:
        whole_name = rf"(?<![A-Z0-9]){re.escape(strain.upper())}(?![0-9])"
        t = t[~t.image.str.upper().str.contains(whole_name, regex=True)]
    return t


def sampling_weights(t):
    """Draw 40% budded cells, 25% cells touching another cell's bud ('not mine'), 15% buds, 20% other unbudded."""
    group = np.where(t.kind == "budded", "budded", np.where(t.touches_other_bud, "not mine", np.where(t.kind == "bud", "bud", "unbudded")))
    counts = pd.Series(group).value_counts()
    return np.array([DRAW_SHARE[g] / counts[g] for g in group])


def map_loss(logit, target, keep):
    """Binary cross-entropy (x10) + dice on the graded pixels (keep = 1 - ignore)."""
    import torch
    import torch.nn.functional as F
    bce = (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * keep).sum((1, 2, 3)) / keep.sum((1, 2, 3)).clamp(min=1)
    p = torch.sigmoid(logit) * keep
    has = (target.sum((1, 2, 3)) > 0).float()
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce * 10 + dice * has


def loss_function(maps, cell_class, bud, cell, ignore, kind, weight):
    import torch
    import torch.nn.functional as F
    up = lambda a: F.interpolate(a[:, None], scale_factor=2, mode="nearest")
    bud_loss = (weight * map_loss(maps[:, :1], up(bud), 1 - up(ignore))).mean()
    cell_loss = map_loss(maps[:, 1:], up(cell), torch.ones_like(up(cell))).mean()
    class_loss = F.cross_entropy(cell_class, kind, label_smoothing=0.05)
    return bud_loss + CELL_LOSS_WEIGHT * cell_loss + 0.5 * class_loss


def train(table, data_dir, model_name, out_file, epochs=30, samples_per_epoch=3000, batch_size=16, lr=3e-4, use_gpu=True,
          save_every=2, pretrained_encoder=True, crop=None):
    """
    Train one model; the weights of the last epoch are saved to out_file (+ every `save_every` epochs, for
    comparison; the shipped models use the last epoch, 30). pretrained_encoder: start from ImageNet weights
    (downloaded once by segmentation-models-pytorch). crop: crop size in px (default: the model's, 192).
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    device = pick_device(use_gpu)
    uses_ash = MODELS[model_name]["uses_ash"]
    crop = crop or MODELS[model_name]["crop_px"]
    sampler = WeightedRandomSampler(torch.tensor(sampling_weights(table), dtype=torch.double), samples_per_epoch, replacement=True)
    loader = DataLoader(CellCrops(table, data_dir, uses_ash, crop=crop), batch_size=batch_size, sampler=sampler, drop_last=True)
    net = build_network(n_input_channels(model_name), pretrained_encoder=pretrained_encoder).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    out_file = Path(out_file); out_file.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(model=model_name, model_type="D", inputs=["DIC", "DAPI", "centre dot"] + (["ASH1CLB2"] if uses_ash else []),
                    crop_px=crop, epochs=epochs, samples_per_epoch=samples_per_epoch, batch_size=batch_size, lr=lr,
                    n_cells=int(len(table)), n_budded=int((table.kind == "budded").sum()), late_m="included")
    json.dump(settings, open(out_file.with_suffix(".json"), "w"), indent=1)
    print(f"training '{model_name}' on {len(table)} cells ({settings['n_budded']} budded), crop {crop} px - {device}")
    for epoch in range(1, epochs + 1):
        net.train(); t0 = time.time(); total = 0
        for x, bud, cell, ignore, kind, weight in loader:
            x, bud, cell, ignore, kind, weight = (a.to(device) for a in (x, bud, cell, ignore, kind, weight))
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                maps, cell_class = net(x)
            loss = loss_function(maps.float(), cell_class.float(), bud, cell, ignore, kind, weight)
            scaler.scale(loss).backward(); scaler.step(optimiser); scaler.update(); schedule.step()
            total += loss.item()
        print(f"epoch {epoch}/{epochs}  loss {total / len(loader):.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if epoch % save_every == 0 and epoch < epochs:
            torch.save(net.state_dict(), out_file.with_name(f"{out_file.stem}_epoch{epoch:02d}.pt"))
    torch.save(net.state_dict(), out_file)
    print(f"saved {out_file}")
    return out_file


# ---------------------------------------------------------------------------------------------
# 3. evaluation on images not used for training
# ---------------------------------------------------------------------------------------------
def evaluate(model, data_dir, images, skip_late_m=False):
    """
    Pairing scores against your corrections. For every curated budded cell: is its mother's Omnipose cell
    paired with the right bud (overlap >= 50% of your bud and >= 30% of the predicted bud)?
    False pair = a bud given to a cell you curated as unbudded. Uses the model's snap setting.
    """
    rows = []
    for name in images:
        f = np.load(Path(data_dir) / f"{name}.npz"); t = pd.read_csv(Path(data_dir) / f"{name}.csv")
        omni = f["omni"].astype(np.int64); bud_cells = f["bud_cells"].astype(np.int64)
        x = f["x"].astype(np.float32)
        # the prepared channels are already normalised: feed them through unchanged
        cells = model.predict_cells_normalised(x[0], x[1], omni, f["ash"].astype(np.float32) if model.uses_ash else None)
        accepted = model.assign(cells, omni)
        if model.snap:
            model.snap_claims(accepted, omni, model._pixels_of)
        bud_size = np.bincount(bud_cells.ravel())
        mothers = t[(t.kind == "budded") & ~t.edge.astype(bool)]
        if skip_late_m:
            mothers = mothers[~mothers.late_m.astype(bool)]
        for r in mothers.itertuples():
            claim = accepted[r.label]["claim"] if r.label in accepted else np.array([], np.int64)
            overlap = int((bud_cells.ravel()[claim] == r.cell).sum()) if len(claim) else 0
            ok = len(claim) and overlap >= 0.5 * bud_size[r.cell] and overlap >= 0.3 * len(claim)
            rows.append(dict(image=name, outcome="correct" if ok else ("wrong bud" if len(claim) else "missed")))
        unbudded = set(t.label[(t.kind == "unbudded") & ~t.edge.astype(bool) & (~t.late_m.astype(bool) if skip_late_m else True)])
        rows += [dict(image=name, outcome="false pair") for m in accepted if m in unbudded]
    r = pd.DataFrame(rows)
    n = r.outcome.value_counts()
    correct, wrong, missed, false = (int(n.get(k, 0)) for k in ("correct", "wrong bud", "missed", "false pair"))
    recall = correct / max(correct + wrong + missed, 1)
    precision = correct / max(correct + wrong + false, 1)
    return dict(budded_cells=correct + wrong + missed, correct=correct, wrong_bud=wrong, missed=missed, false_pairs=false,
                recall=round(recall, 3), precision=round(precision, 3), f1=round(2 * recall * precision / max(recall + precision, 1e-9), 3))
