"""
Automatic mother/bud assignment with the trained bud model ("model D").

How it works, for one image
---------------------------
1. For every segmented cell (Omnipose object) the model looks at a crop (192 x 192 px) centred on that
   cell and answers three questions:
     * class: is this cell unbudded, budded, or is it itself a bud?
     * bud map: which pixels are the bud of THIS cell? (empty if it has none)
     * cell map: which pixels are the body of this cell? (not used by buddy, it helps the model learn)
   Input channels: DIC, DAPI and a soft dot on the cell's centre (the model gets no outline of the cell,
   so an imperfect Omnipose outline cannot mislead it) + the ASH1/CLB2 max projection for the
   'with_ASH1CLB2' model.
2. Each cell with a bud map makes a "claim" on a bud. Claim strength = P(budded) x mean bud-map value.
   Claims below the model's cut-off are dropped.
3. Claims are accepted from strongest to weakest, so if two cells claim the same bud, the stronger claim
   keeps it. A close call (the other claim's strength within 0.15) is flagged for you to check.
4. "Snap": a claimed bud that lies mostly inside one Omnipose object (not the mother) becomes that whole
   object - the model says which bud belongs to which mother, Omnipose gives the full outline. Helps for
   large buds, which Omnipose usually segments as a cell of their own. ("bud_classification" -> "snap")
5. The result is written in the usual buddy format: buds merged into their mother in the cell mask,
   and a bud mask (1 = bud, 2 = background, 3 = mother).
6. Possible problems are listed (flags) so you only need to check those in napari.

Two models (choose in the config file, "bud_classification" -> "model"):
    "DIC_DAPI_only"   uses DIC + DAPI. Works for every strain.
    "with_ASH1CLB2"   also uses the ASH1/CLB2 smFISH channel: fewer false pairs, better for small buds.
                      Not trained on she2/she3 mutants (ASH1 mRNA is not transported to the bud there).
Both were trained with late-M cells (bud with its own full-size nucleus) included, so those are paired too.
The weights (bud-models) download automatically on first use; models/<name>.json describes each model.

Typical use (the bulk-processing notebook does this for you via pipeline.classify_buds):
    model = BudModel.load("with_ASH1CLB2", use_gpu=True)
    result = model.classify(dic, dapi, cell_masks, ash1clb2_maxproj)
    result["cell_masks"], result["bud_masks"], result["pairs"], result["flags"]
"""
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from buddy.buds.bud_tools import BACKGROUND, BUD, MOTHER  # bud-mask values used everywhere in buddy
from buddy.buds.dic_orientation import dic_light_direction, rotation_to_reference

# ---------------------------------------------------------------------------------------------

log = logging.getLogger("buddy")
# settings
# ---------------------------------------------------------------------------------------------
MODELS = {   # last epoch (30) and cut-off 0.5, fixed in advance; tested on an imaging day left out of training (see models/<file>.json)
    "DIC_DAPI_only": {"file": "bud_model_DIC_DAPI_only_v2.pt", "uses_ash": False, "cut_off": 0.5, "crop_px": 192},
    "with_ASH1CLB2": {"file": "bud_model_with_ASH1CLB2_v2.pt", "uses_ash": True, "cut_off": 0.5, "crop_px": 192},
}
# where the weights are downloaded from the first time a model is used
WEIGHTS_URL = "https://github.com/Tutuccilab/buddy/releases/download/bud-models/{file}"
DEFAULT_WEIGHTS_DIR = Path(__file__).parent.parent / "models"   # buddy/models

CROP = 192                 # default crop size around each cell (px); enlarged 2x inside the network
DOT_SIGMA = 4.0            # the soft dot that marks the cell: gaussian, sigma in px
CLASSES = ["unbudded", "budded", "bud"]
SNAP_FRAC = 0.6            # snap: a claim lying >= this fraction inside one Omnipose object takes the whole object
MIN_BUD_PX = 15            # smaller bud maps are ignored
CLOSE_CALL = 0.15          # two claims closer than this -> flagged
UNCLAIMED_BUD_P = 0.8      # a cell this sure to be a bud, but not given to any mother -> flagged
BORDERLINE = 0.1           # claims within this distance of the cut-off -> flagged
LIGHT_MIN_CONSISTENCY = 0.9   # DIC lighting measured less consistently than this -> warning (set "dic_rotation" in the config)


# ---------------------------------------------------------------------------------------------
# image preparation (identical to training)
# ---------------------------------------------------------------------------------------------
def normalise(image, valid=None):
    """Scale an image so its 0.5th-99.8th percentile maps to 0-1 (clipped to -1..3). Used for DIC and DAPI."""
    image = np.asarray(image, np.float32)
    values = image[valid] if valid is not None and valid.any() else image.ravel()
    low, high = np.percentile(values, [0.5, 99.8])
    return ((image - low) / max(high - low, 1e-6)).clip(-1, 3)


def normalise_ash(maxproj):
    """ASH1/CLB2 max projection: percentile scaling, then log1p (single molecules would otherwise saturate)."""
    maxproj = np.asarray(maxproj, np.float32)
    low, high = np.percentile(maxproj, [0.5, 99.8])
    return np.log1p(np.clip((maxproj - low) / max(high - low, 1.0), 0, None)).clip(0, 4)


def pad_to(image, shape):
    """Place a 2D image in the top-left corner of an array of `shape` (zero padded / cropped)."""
    out = np.zeros(shape, np.float32)
    h, w = min(shape[0], image.shape[0]), min(shape[1], image.shape[1])
    out[:h, :w] = image[:h, :w]
    return out


def crop_around(array, cy, cx, size=CROP):
    """(..., H, W) -> (..., size, size) centred on (cy, cx), zero padded. Also returns the crop's top-left corner."""
    H, W = array.shape[-2:]
    y0, x0 = int(cy) - size // 2, int(cx) - size // 2
    ys, xs = slice(max(y0, 0), min(y0 + size, H)), slice(max(x0, 0), min(x0 + size, W))
    out = np.zeros(array.shape[:-2] + (size, size), np.float32)
    out[..., ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = array[..., ys, xs]
    return out, (y0, x0)


def centre_dot(cy, cx, size=CROP, sigma=DOT_SIGMA):
    """A soft dot (gaussian) at (cy, cx) in a size x size crop: tells the model which cell the question is about."""
    gy, gx = np.ogrid[:size, :size]
    return np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def pick_device(use_gpu=True):
    """'cuda' when a GPU is available and wanted, else 'cpu'."""
    import torch
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu:
        log.warning("no CUDA GPU found (or PyTorch was installed without GPU support) - running on CPU")
    return "cpu"


# ---------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------
def build_network(n_input_channels, pretrained_encoder=False):
    """U-Net (ResNet-34 encoder): 2 maps (bud of this cell, this cell) + a 3-class output. Input is enlarged 2x first."""
    import torch.nn as nn
    import torch.nn.functional as F

    from buddy.buds.unet import Unet

    class BudNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Unet(in_channels=n_input_channels, classes=2, aux_classes=len(CLASSES), dropout=0.2,
                            pretrained_encoder=pretrained_encoder)

        def forward(self, x):                                  # x: (batch, channels, crop, crop)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            maps, cell_class = self.net(x)
            return maps, cell_class                            # (batch, 2, 2*crop, 2*crop), (batch, 3)

    return BudNet()


def n_input_channels(model_name):
    """DIC, DAPI, centre dot (+ ASH1/CLB2)"""
    return 4 if MODELS[model_name]["uses_ash"] else 3


def weights_path(model_name, weights_dir=None):
    """Local path of a model's weights; downloads them on first use."""
    file = MODELS[model_name]["file"]
    path = Path(weights_dir or DEFAULT_WEIGHTS_DIR) / file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = WEIGHTS_URL.format(file=file)
        log.info(f"downloading bud model weights: {url}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise FileNotFoundError(f"Could not find or download {file}. Download it from {url} "
                                    f"and put it in {path.parent}") from e
    return path


# ---------------------------------------------------------------------------------------------
# the model: prediction + turning predictions into mother-bud pairs
# ---------------------------------------------------------------------------------------------
class BudModel:

    def __init__(self, network, model_name, device, cut_off=None, snap=True, crop=None, dic_rotation=None):
        self.network = network
        self.name = model_name
        self.uses_ash = MODELS[model_name]["uses_ash"]
        self.cut_off = MODELS[model_name]["cut_off"] if cut_off is None else cut_off
        self.crop = MODELS[model_name]["crop_px"] if crop is None else int(crop)
        self.snap = snap
        self.device = device
        self.dic_rotation = dic_rotation     # None = measure per image; 0-3 = fixed number of 90-degree turns

    @classmethod
    def load(cls, model_name="DIC_DAPI_only", use_gpu=True, weights_dir=None, cut_off=None, snap=True, weights_file=None,
             dic_rotation=None):
        """weights_file: your own trained weights (05_train_bud_model.ipynb) instead of the released ones."""
        import torch
        if model_name not in MODELS:
            raise ValueError(f"unknown bud model '{model_name}', choose one of {list(MODELS)}")
        device = pick_device(use_gpu)
        network = build_network(n_input_channels(model_name))
        # the weights are unpickled on the CPU: unpickling straight onto the GPU asks CUDA for the current device
        # from the loading thread, which stalled now and then in napari (step 3 after step 2); .to(device) below
        # moves the network to the GPU afterwards
        state = torch.load(weights_file or weights_path(model_name, weights_dir), map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        network.to(device).eval()
        log.info(f"bud model '{model_name}' loaded on {device}")
        return cls(network, model_name, device, cut_off, snap, dic_rotation=dic_rotation)

    # ---- step 1: predict, cell by cell ---------------------------------------------------------
    def predict_cells(self, dic, dapi, cell_masks, ash=None):
        """
        dic, dapi: 2D images in the same frame as cell_masks (dapi = max projection).
        ash: ASH1/CLB2 max projection (only for the 'with_ASH1CLB2' model).
        Returns one dict per cell: label, class probabilities, and its bud map (pixels with p > 0.1).
        """
        shape = np.asarray(cell_masks).shape
        dic = pad_to(np.asarray(dic), shape)
        valid = dic != 0                                   # area covered by the (shifted) DIC image
        dic_n, dapi_n = normalise(dic, valid), normalise(pad_to(np.asarray(dapi), shape), valid)
        dic_n[~valid], dapi_n[~valid] = 0, 0
        ash_n = None
        if self.uses_ash:
            if ash is None:
                raise ValueError("the 'with_ASH1CLB2' model needs the ASH1/CLB2 max projection")
            ash_n = pad_to(normalise_ash(ash), shape)
        return self.predict_cells_normalised(dic_n, dapi_n, cell_masks, ash_n)

    def predict_cells_normalised(self, dic_n, dapi_n, cell_masks, ash_n=None, batch_size=48):
        """predict_cells() for channels that are already normalised (used during training/evaluation)."""
        import torch
        cell_masks = np.asarray(cell_masks).astype(np.int64)
        shape = cell_masks.shape
        image = np.stack([dic_n, dapi_n] + ([ash_n] if self.uses_ash else [])).astype(np.float32)

        # turn every crop so the DIC lighting comes from the direction the model was trained on
        angle, consistency, n_cells = dic_light_direction(image[0], cell_masks)
        if self.dic_rotation is not None:
            k = int(self.dic_rotation) % 4
        else:
            k = rotation_to_reference(angle)
            if not consistency >= LIGHT_MIN_CONSISTENCY:          # also catches NaN (no cell large enough)
                log.warning(f"..DIC lighting direction unclear (angle {angle:.0f} deg, consistency {consistency:.2f}, "
                            f"{n_cells} cells): turned by {k} x 90 deg - check the bud result, or fix the turn for this "
                            'experiment with "bud_classification" -> "dic_rotation" (0-3) in the config file')
        self.last_rotation = k
        self.last_light = dict(angle=float(angle), consistency=float(consistency), n_cells=int(n_cells))

        # the model also sees each crop mirrored along the anti-diagonal (keeps the DIC lighting); both are averaged
        anti_transpose = lambda t: torch.rot90(t.transpose(-1, -2), 2, (-2, -1))
        boxes = ndi.find_objects(cell_masks)
        labels = [i for i, b in enumerate(boxes, start=1) if b is not None]
        H, W = shape
        cells = []
        for start in range(0, len(labels), batch_size):
            batch, corners = [], []
            for label in labels[start:start + batch_size]:
                yy, xx = np.nonzero(cell_masks[boxes[label - 1]] == label)
                cy, cx = yy.mean() + boxes[label - 1][0].start, xx.mean() + boxes[label - 1][1].start
                img, corner = crop_around(image, cy, cx, self.crop)
                masks, _ = crop_around(cell_masks, cy, cx, self.crop)
                yy, xx = np.nonzero(masks == label)                       # the cell's centre inside the crop
                dot_y, dot_x = (yy.mean(), xx.mean()) if len(yy) else (cy - corner[0], cx - corner[1])
                x = np.concatenate([img[:2], centre_dot(dot_y, dot_x, self.crop)[None], img[2:]])
                batch.append(np.rot90(x, k, (1, 2)) if k else x)
                corners.append(corner)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(self.device)
            with torch.no_grad():
                maps, cell_class = self.network(x)
                maps2, cell_class2 = self.network(anti_transpose(x))
                bud_map = (torch.sigmoid(maps[:, :1]) + anti_transpose(torch.sigmoid(maps2[:, :1]))) / 2
                p_class = (cell_class.softmax(1) + cell_class2.softmax(1)) / 2
                bud_map = torch.nn.functional.avg_pool2d(bud_map, 2)[:, 0].cpu().numpy()   # back to crop x crop
            p_class = p_class.cpu().numpy()
            for label, bm, pc, (y0, x0) in zip(labels[start:start + batch_size], bud_map, p_class, corners):
                if k:
                    bm = np.rot90(bm, -k)
                yy, xx = np.nonzero(bm > 0.1)
                fy, fx = yy + y0, xx + x0
                inside = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
                order = np.argsort(fy[inside] * W + fx[inside])
                cells.append(dict(label=int(label), p_unbudded=float(pc[0]), p_budded=float(pc[1]), p_bud=float(pc[2]),
                                  pixels=(fy[inside] * W + fx[inside])[order].astype(np.int64),
                                  prob=bm[yy[inside], xx[inside]][order].astype(np.float32)))
        return cells

    # ---- step 2: claims ----------------------------------------------------------------------
    @staticmethod
    def make_claim(cell, W):
        """The most confident connected part of a cell's bud map (p > 0.5, >= 15 px) is its claimed bud."""
        strong = cell["pixels"][cell["prob"] > 0.5]
        cell["claim"] = np.array([], np.int64)
        if len(strong) >= MIN_BUD_PX:
            yy, xx = strong // W, strong % W
            y0, x0 = yy.min(), xx.min()
            canvas = np.zeros((yy.max() - y0 + 1, xx.max() - x0 + 1), bool)
            canvas[yy - y0, xx - x0] = True
            parts, n = ndi.label(canvas)
            if n:
                probs = cell["prob"][cell["prob"] > 0.5]
                weight = np.bincount(parts[yy - y0, xx - x0], weights=probs, minlength=n + 1)[1:]
                best = parts[yy - y0, xx - x0] == (np.argmax(weight) + 1)
                if best.sum() >= MIN_BUD_PX:
                    cell["claim"] = np.sort(strong[best])
        cell["strength"] = cell["p_budded"] * mean_on(cell, cell["claim"]) if len(cell["claim"]) else 0.0

    # ---- step 3: who gets which bud ----------------------------------------------------------
    def assign(self, cells, cell_masks):
        """
        Accept claims from strongest to weakest; every cell gets at most one role. A claim is skipped when
          * its own cell is already an accepted bud (a bud cannot be a mother),
          * more than half of its bud is already an accepted bud (a stronger claim got there first; the winner's
            'margin' records how close it was, see find_flags),
          * its bud covers more than half of an accepted mother (a mother cannot become a bud).
        Returns {mother label: cell dict with 'claim' (and 'margin' when another cell claimed the same bud)}.
        """
        cell_masks = np.asarray(cell_masks)
        W = cell_masks.shape[1]
        pixels_of = {}                                                       # label -> flat pixel indices
        for label, b in enumerate(ndi.find_objects(cell_masks), start=1):
            if b is not None:
                yy, xx = np.nonzero(cell_masks[b] == label)
                pixels_of[label] = np.sort((yy + b[0].start) * W + (xx + b[1].start))
        for c in cells:
            self.make_claim(c, W)
        candidates = sorted((c for c in cells if c["strength"] >= self.cut_off), key=lambda c: -c["strength"])

        owner = np.zeros(cell_masks.size, np.int64)                           # pixel -> mother label (0 = free)
        accepted = {}
        for c in candidates:
            if (owner[pixels_of[c["label"]]] > 0).mean() > 0.5:
                continue                                                     # this cell is itself a bud
            taken = owner[c["claim"]]
            if (taken > 0).mean() > 0.5:                                     # its bud is already taken
                winner = accepted[int(np.bincount(taken[taken > 0]).argmax())]
                winner["margin"] = min(winner.get("margin", np.inf), winner["strength"] - c["strength"])
                continue
            if any(np.isin(pixels_of[m], c["claim"]).mean() > 0.5 for m in accepted):
                continue                                                     # its bud is an accepted mother
            accepted[c["label"]] = c
            owner[c["claim"]] = c["label"]
        self._pixels_of = pixels_of
        return accepted

    # ---- step 3b: snap -------------------------------------------------------------------------
    @staticmethod
    def snap_claims(accepted, cell_masks, pixels_of, frac=SNAP_FRAC):
        """
        A claimed bud lying mostly (>= frac) inside one Omnipose object that is not the mother (and not itself an
        accepted mother) becomes that whole object. The model found the bud; Omnipose's outline of it is usually
        better than the model's, especially for large buds.
        """
        flat = np.asarray(cell_masks).ravel()
        for mother, c in accepted.items():
            under = flat[c["claim"]]
            others = under[(under > 0) & (under != mother)]
            if not others.size:
                continue
            obj = int(np.bincount(others).argmax())
            if (under == obj).mean() >= frac and obj not in accepted:
                claim = np.union1d(c["claim"][under != mother], pixels_of[obj])
                c["snapped"] = len(claim) != len(c["claim"])          # the outline actually changed
                c["claim"] = claim
        return accepted

    # ---- step 4: write the result as buddy masks ----------------------------------------------
    @staticmethod
    def build_masks(cell_masks, accepted, pixels_of):
        """
        Merged cell mask (mother + bud share the mother's label) and the 3-class bud mask.
        A segmented cell that lies mostly (> 50 %) inside an accepted bud becomes part of that bud.
        """
        cells = np.asarray(cell_masks).astype(np.int64).copy()
        flat = cells.ravel()                       # view: edits change `cells`
        is_bud = np.zeros(flat.size, bool)
        for mother, c in accepted.items():
            claim = c["claim"]
            absorbed = [label for label in np.unique(flat[claim]) if label not in (0, mother)
                        and np.isin(pixels_of[label], claim).mean() > 0.5]
            bud_pixels = claim[np.isin(flat[claim], [0, mother] + absorbed)]
            for label in absorbed:
                bud_pixels = np.union1d(bud_pixels, pixels_of[label])
            flat[bud_pixels] = mother
            is_bud[bud_pixels] = True
        bud_mask = np.full(cells.shape, BACKGROUND, np.uint8)
        bud_mask[cells > 0] = MOTHER
        bud_mask[is_bud.reshape(cells.shape)] = BUD
        return cells, bud_mask

    # ---- step 5: what should a human check? ----------------------------------------------------
    def find_flags(self, cells, accepted, cell_masks):
        """Cells worth a look in napari, with the reason. (A bud at the image edge has its mother outside the image: not flagged.)"""
        cell_masks = np.asarray(cell_masks)
        H, W = cell_masks.shape
        boxes = ndi.find_objects(cell_masks)
        in_a_bud = set()
        for mother, c in accepted.items():
            for label in np.unique(cell_masks.ravel()[c["claim"]]):
                if label and label != mother and np.isin(self._pixels_of[label], c["claim"]).mean() > 0.5:
                    in_a_bud.add(int(label))
        flags = []
        for c in cells:
            label, box = c["label"], boxes[c["label"] - 1]
            at_edge = box[0].start == 0 or box[1].start == 0 or box[0].stop >= H or box[1].stop >= W
            if label in accepted and accepted[label].get("margin", np.inf) < CLOSE_CALL:
                flags.append((label, "close call", "two cells claim this bud almost equally strongly - check the mother"))
            elif label in accepted and c["strength"] < self.cut_off + BORDERLINE:
                flags.append((label, "weak pair", f"bud accepted, but only just (strength {c['strength']:.2f})"))
            elif label not in accepted and len(c["claim"]) and self.cut_off - BORDERLINE <= c["strength"] < self.cut_off:
                flags.append((label, "possible missed bud", f"bud just below the cut-off (strength {c['strength']:.2f})"))
            if c["p_bud"] >= UNCLAIMED_BUD_P and label not in accepted and label not in in_a_bud and not at_edge:
                flags.append((label, "bud without mother", f"looks like a bud (P = {c['p_bud']:.2f}) but no mother claimed it"))
        rows = [dict(label=label, flag=kind, reason=reason, y0=boxes[label - 1][0].start, x0=boxes[label - 1][1].start,
                     y1=boxes[label - 1][0].stop, x1=boxes[label - 1][1].stop) for label, kind, reason in flags]
        return pd.DataFrame(rows, columns=["label", "flag", "reason", "y0", "x0", "y1", "x1"])

    # ---- everything in one call ---------------------------------------------------------------
    def classify(self, dic, dapi, cell_masks, ash=None):
        """
        Returns a dict with
          cell_masks : merged cell mask (mother + bud share one label)  -> corrected/DIC_corrected_masks.tif
          bud_masks  : 1 = bud, 2 = background, 3 = mother              -> corrected/bud_corrected_masks.tif
          pairs      : table of mother-bud pairs (mother label, bud size, strength, close call)
          cells      : per-cell model output (class probabilities)
          flags      : cells to check (label, flag, reason, bounding box)
        """
        cell_masks = np.asarray(cell_masks).astype(np.int64)
        cells = self.predict_cells(dic, dapi, cell_masks, ash)
        accepted = self.assign(cells, cell_masks)
        if self.snap:
            self.snap_claims(accepted, cell_masks, self._pixels_of)
        merged, bud_masks = self.build_masks(cell_masks, accepted, self._pixels_of)
        flags = self.find_flags(cells, accepted, cell_masks)
        pairs = pd.DataFrame([dict(mother_label=m, bud_px=int((bud_masks.ravel()[c["claim"]] == BUD).sum()),
                                   strength=round(c["strength"], 3), close_call=bool(c.get("margin", np.inf) < CLOSE_CALL),
                                   snapped=bool(c.get("snapped", False)))
                              for m, c in accepted.items()], columns=["mother_label", "bud_px", "strength", "close_call", "snapped"])
        table = pd.DataFrame([{k: v for k, v in c.items() if k in ("label", "p_unbudded", "p_budded", "p_bud", "strength")}
                              for c in cells])
        return dict(cell_masks=merged, bud_masks=bud_masks, pairs=pairs, cells=table, flags=flags,
                    dic_rotation=self.last_rotation, dic_light=getattr(self, "last_light", None))


def mean_on(cell, region):
    """Mean of a cell's bud map over `region` (sorted flat pixel indices); pixels below 0.1 count as 0."""
    if not len(region) or not len(cell["pixels"]):
        return 0.0
    pos = np.clip(np.searchsorted(cell["pixels"], region), 0, len(cell["pixels"]) - 1)
    hit = cell["pixels"][pos] == region
    return float(cell["prob"][pos][hit].sum() / len(region))
