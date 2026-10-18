# Methods

## Single-molecule FISH image analysis

Single-molecule FISH (smFISH) images were analysed with a custom, fully
scripted Python pipeline (the `buddy` package, built on Big-FISH, Omnipose /
Cellpose, scikit-image and SciPy), with the interactive quality-control and
correction steps performed in napari through a companion plugin
(`napari-buddy`). The pipeline is organised as a numbered set of Jupyter
notebooks (`01`–`05`) that each read their inputs from, and write their outputs
back to, a per-image output folder, so that every stage is reproducible and every
processing parameter is recorded in a per-image `img.json` metadata file. Unless
stated otherwise, all parameters below were read from a per-experiment JSON
configuration file that also defined the mapping from microscope filter names to
transcripts (CY5 → *ASH1/CLB2*, CY3.5 → *SRL1*, CY3 → *SUN4*, DAPI → nuclei).

### Inputs

Each field of view consisted of a multi-channel Olympus `.vsi` fluorescence
z-stack containing the three mRNA probe channels and a DAPI channel, together
with a matched two-dimensional differential-interference-contrast (DIC) image of
the same field. The voxel size was 200 nm axially and 65 nm laterally
(`scale = [200, 65, 65]` nm).

### Step 1 — Reading and organising the raw data

For each field, the fluorescence stack was read channel-by-channel from the
`.vsi` file (via Bio-Formats), assigning each channel to its transcript through
the filter→mRNA mapping in the configuration file, and the DIC image was read
separately. A two-channel composite ("GRGB") was formed from the DIC image and
the maximum-intensity z-projection of the DAPI channel; this composite is the
input to cell segmentation. Each fluorescence channel and the DIC image were then
written to disk as individual TIFFs so that the later stages could reload exactly
the same data, and a metadata record was written.

### Step 2 — Cell and nucleus segmentation

Cells were segmented from the DIC + DAPI composite with Omnipose (Cellpose
`cyto2_omni` model). For *Saccharomyces cerevisiae* the segmentation parameters
were a mask threshold of −3, a flow threshold of 1, an expected diameter of
50 px and a minimum object size of 10 px, with Omnipose mask reconstruction and
DBSCAN clustering enabled. Nuclei were segmented independently from the DAPI
maximum-intensity projection with the Cellpose `nuclei` model.

Segmentation masks were then post-processed to retain only biologically
plausible objects, in three steps: (i) cells whose area contained no DAPI signal
were discarded; (ii) objects smaller than 20 px were removed; and (iii) nuclei
that did not overlap any retained cell were removed. The cleaned cell mask
(`DIC_masks_pp.tif`) and nucleus mask were saved for downstream use.

### Step 3 — Focus selection, spot detection, decomposition and registration

**Focus selection.** To restrict quantification to in-focus planes, a
high-density 200 × 200 px patch was selected deterministically from the cell mask
and a per-slice focus score was computed on that patch with Big-FISH
(`compute_focus`). The focus profile was smoothed with a Savitzky–Golay filter,
and the in-focus z-range was identified automatically from the inflection points
of the smoothed profile. Background subtraction and spot detection were run on
the whole stack and only the spots inside the in-focus range were kept, so that
no plane of the range is a boundary of the detection.

**Background suppression.** Each mRNA channel was background-subtracted with a
Gaussian filter (Big-FISH `remove_background_gaussian`, σ = [0.75, 2.3, 2.3] in
z, y, x).

**Spot detection.** Diffraction-limited mRNA spots were detected on the in-focus
sub-stack of the background-subtracted image with Big-FISH `detect_spots`, using
a voxel size of [200, 65, 65] nm, an expected spot radius of [880, 120, 120] nm
(z, y, x), and a fixed per-channel intensity threshold (50) taken from the
configuration file. For every spot, the raw and background-subtracted
intensities at the spot centre were recorded alongside its coordinates.

**Dense-region decomposition.** Regions of clustered transcripts too dense to be
resolved as individual spots were decomposed into their constituent molecules
with Big-FISH `decompose_dense` (α = 0.5, β = 2, γ = 1), yielding for each
channel a set of single spots plus a set of dense regions, each annotated with an
estimated transcript count.

**DIC–fluorescence registration.** Because the DIC image and the fluorescence
channels followed different optical paths, a residual translational offset between
the two frames was corrected using the detected spots as fiducials. The
translation that best mapped the spots onto the cell masks was estimated by
maximising spot–mask overlap: candidate integer shifts up to ±200 px were sampled
by Monte-Carlo search (4000 iterations) and then refined by exhaustive local
search within ±20 px. Each candidate shift was scored by the mean, over all
spots, of a Euclidean distance transform of the cell masks (capped at 50 px) at
the shifted spot positions; capping (rather than zeroing) the distance far from
cells prevents a degenerate optimum in which spots are pushed into empty space.
The inverse of the estimated shift was applied to the DIC image and the cell
masks so that both aligned with the fluorescence frame, and the aligned cell mask
replaced the working cell mask. The nucleus masks, being derived directly from
the DAPI channel, were already in the fluorescence frame and required no shift.

A first pass of spot-to-cell assignment (below) was then run on the aligned
masks.

### Step 4 — Interactive mask correction and bud annotation

Segmentation, registration and the automatic bud detection were reviewed and
corrected for every field in napari (napari-buddy). All layers for a field were
loaded directly from the saved pipeline outputs — DIC image, cell and nucleus
masks (preferring the aligned/corrected versions when present), each fluorescence
channel, and the detected and decomposed spots — with the z-stacks
maximum-projected for display and spots coloured by the cell they fell in.

Operators corrected the cell and nucleus masks by hand where needed, and
annotated a three-class bud mask (background, mother, bud). This mask was seeded
automatically: cells small relative to the median cell area were pre-labelled as
buds, and non-round cells (low solidity or high eccentricity), which typically
correspond to an unsplit mother–bud pair, were flagged for review. Each bud was
then linked to its mother cell — either automatically, by assigning the bud to
the cell its own outline most adjoins, or manually with a line-drawing tool that
connects a bud to a nearby cell. Linking merged the bud into its mother's label
in the cell mask (so a mother and its bud share one cell identity), while the bud
annotation was preserved so that the bud compartment remained identifiable. The
corrected cell, nucleus and bud masks were saved per field.

### Step 5 — Spot-to-cell assignment and per-cell quantification

Detected spots and decomposed dense regions were assigned to cells using the
corrected masks. Each spot or dense region was assigned to the cell whose label
occupied its (y, x) position. Single spots were counted as one transcript; dense
regions contributed their decomposed transcript count. Spots and dense regions
that additionally fell within a nucleus were scored as nascent transcripts and
transcription sites. Per cell, the pipeline reported the total number of
transcripts (single spots + decomposed transcripts − number of dense regions),
the number of nuclei, and cell morphology (bounding box, area, eccentricity,
major/minor axis lengths, orientation, perimeter, solidity). Where a bud was
annotated, all count and morphology measurements were additionally split into
mother and bud compartments (e.g. `spots_mother`, `total_RNAs_bud`), and
per-nucleus DAPI area and mean intensity were tabulated separately for the mother
and the bud. Results were written as one table per mRNA channel per field.

### Downstream data assembly

The per-field, per-channel tables were concatenated into a single master
dataframe. Experimental metadata (strain, deletion/condition, biological
replicate, field of view, mRNA) were attached from the folder structure and
configuration, and each cell was given a unique identifier
(`strain_replicate_cellID_fieldOfView`) used to trace any row back to its image
and to recall the exact cell in napari for visual verification.

### Cell-cycle assignment

Cells were assigned to a cell-cycle phase (G1, S, G2, M) from the per-cell
morphological and molecular measurements. Pixel measurements were first converted
to physical units (0.065 µm px⁻¹; 0.065² µm² px⁻²), and analysis was restricted
to cells of sensible size (major-axis length 2–20 µm). Phases were then defined
by combining bud size, nucleus position and the level of the cell-cycle-regulated
transcript *ASH1/CLB2*:

- **G1** — no bud (bud area = 0 µm²), a single nucleus in the mother, and a low
  *ASH1/CLB2* level (≤ 5 transcripts);
- **S** — a small emerging bud (0 < bud area < 2 µm²), a single mother nucleus
  and no nucleus yet in the bud;
- **G2** — a larger bud (2 ≤ bud area < 15 µm²), a single mother nucleus and no
  bud nucleus;
- **M** — a bud (area > 2 µm²) into which a nucleus has appeared (one mother
  nucleus and one bud nucleus), marking nuclear division.

Phase was assigned by evaluating these mutually exclusive criteria and encoded
as an ordered categorical variable (G1 < S < G2 < M); cells not matching any
criterion were left unassigned. To validate this automatic, feature-based
assignment, a subset of cells was also annotated manually: single-cell image
crops were generated from each cell's unique identifier and bounding box, scored
by eye for cell-cycle phase, and the resulting expert annotations were compared
against the automatic phase calls.

## Additions to buddy after the analysis

The following were added to the released version of buddy **after** the data in
this study were analysed. The results above were obtained with Big-FISH on the
CPU and with manually curated bud annotations; none of the additions below were
used for them.

### Automatic bud classification

**What the model does.** In budding yeast a daughter cell (the bud) grows out of its mother cell, and
Omnipose often segments the bud as a separate object. The bud classification step decides, for every
segmented cell, whether it is unbudded, has a bud, or is itself a bud, and if it has a bud, which pixels
belong to it. Each bud is then joined to its mother, so that mRNAs can be counted per mother and per bud.
Two versions of the model were trained: one on DIC and DAPI, and one that also uses the *ASH1/CLB2*
smFISH signal, a transcript that accumulates in the bud during mitosis.

**Input.** For every segmented cell (buddy's cleaned Omnipose layer) a square image region of 192 × 192
pixels (12.5 × 12.5 µm) centred on the cell's centre of mass is cut out, with three layers: the in-focus DIC
plane, the maximum projection of the DAPI stain, and a soft dot (Gaussian, σ = 4 pixels) marking the cell in
question. The model is deliberately not given the Omnipose outline of the cell: outlines that cut through a
bud, or merge two cells, would otherwise mislead it. The second model also receives the maximum projection of
the *ASH1/CLB2* signal. DIC and DAPI are scaled per image between the 0.5th and 99.8th percentile and clipped
to [−1, 3]; the *ASH1/CLB2* layer is scaled the same way, clipped below at 0, log-transformed (log1p) and
clipped to [0, 4] so that bright transcription sites do not dominate.

**DIC lighting direction.** A DIC image looks as if every cell is lit from one side, set by the microscope's
prism. Each image is therefore turned by a multiple of 90° so that the bright side points the same way as in
the training images (135°). The direction is measured per image: for each cell (mask grown by 3 pixels, at
least 500 pixels), the brightness relative to the cell's mean weights the direction from the cell's centre to
each pixel, and the average gives the cell's bright side; the unit vectors of up to 300 cells are averaged.
The length of that average (0–1) shows how consistently the cells agree; below 0.9 buddy warns, and the turn
can then be fixed for an imaging session in the configuration file. For the same reason, flips used during
training and prediction are restricted to the identity and the mirror along the anti-diagonal, the only ones
that keep the lighting direction.

**Network.** The model is a U-Net, a convolutional neural network that outputs one value per pixel, with a
ResNet-34 encoder initialised from weights pre-trained on everyday photographs (ImageNet) and implemented with
the segmentation-models-pytorch library (24.4 million parameters). It has three outputs: (i) a map giving, for
every pixel, the probability that it belongs to the bud of the cell in question, (ii) a map of the cell's own
body (used only during training, where it forces the model to locate the cell from the dot), and (iii) the
probability that the cell is unbudded, budded, or a bud (a classification head on the encoder, dropout 0.2).
The input is enlarged twofold inside the network so that small buds are covered by more pixels.

**From per-cell answers to mother–bud pairs.** Each cell's region is run through the network twice, as is and
mirrored along the anti-diagonal, and the two outputs are averaged. Of the bud-map pixels above 0.5, the
connected region with the highest summed probability (at least 15 pixels) is the cell's claimed bud. The
strength of a claim is the cell's probability of being budded multiplied by its mean bud probability over that
region; claims below a cut-off of 0.5 are discarded. The remaining claims are accepted from strongest to
weakest, so that every cell has at most one role: a claim is skipped when its own cell already lies mostly
(> 50%) inside an accepted bud, when more than half of its bud is already an accepted bud (the stronger claim
keeps it), or when it would cover more than half of an accepted mother. On 16 fields of imaging day 24-09-09
(*ASH1/CLB2* model) none of these conflicts occurred. A claimed region that lies mostly (≥ 60%)
inside one Omnipose object other than the mother is replaced by that whole object ("snap"): the model decides
which bud belongs to which mother, and the Omnipose outline, which is usually more complete for large buds than
the model's bud map, provides the outline. Four situations are flagged for manual checking in napari: a bud
also claimed by a second cell with a strength within 0.15 of the winner (close call), accepted claims within
0.1 above the cut-off (weak pair), rejected claims within 0.1 below it (possible missed bud), and cells that are
at least 80% likely to be a bud but were claimed by no mother.

**Training data.** The models learned from the manual mother–bud annotations made for the analyses in this
study (step 4 of the pipeline): 132 fields of view from five imaging days (22-06-01, 24-09-09, 24-10-04,
25-08-20 and 25-08-22), strains YET563 and YET572 (*she2*Δ, *she3*Δ), YET915, YET916, YET918, YET919 and
YET921, biological replicates BR1–BR5. Whole fields, not cells, were assigned to sets. To measure performance
on an imaging session the models have never seen, every field of imaging day 24-09-09 was kept out of training:
the test set consists of 20 fields of held-out replicates (24-09-09: YET915 BR5, YET916 BR4, YET921 BR4;
25-08-20: YET572 BR5), and the 19 other fields of 24-09-09 were not used at all. Of the remaining fields, 66
were used for training and 6 (about 10% per strain, chosen at random) to monitor training. The 21 fields of
imaging day 22-04-29 were not used because of poor DIC or failed segmentation. Each Omnipose cell was matched
to the annotation: an overlap (IoU) ≥ 0.5 with an annotated cell gives *budded* or *unbudded*, ≥ 50% of its
area inside an annotated bud gives *bud*; unmatched objects and cells touching the image border were not
trained on. Cells in late mitosis (a bud that already contains its own full-size nucleus) were kept. For the
*ASH1/CLB2* model the *she2* and *she3* deletion strains were left out, because in these mutants *ASH1* mRNA
is not transported to the bud. Small Omnipose objects (< 4 µm²) that the segmentation clean-up had discarded
and that were not annotated but lie within 3 pixels of a training cell were treated as "ignore regions": they
may be buds the annotation missed, so the bud map was not graded there.

**Training.** Each epoch drew 3,000 cells with replacement from four groups in fixed proportions, so that the
rare cases were seen often enough: 40% budded mothers; 25% "not mine" cells, which lie within 5 pixels of a bud
that belongs to another cell, so that the network learns not to claim a neighbour's bud, an easy mistake in
dense fields; 15% buds; and 20% other unbudded cells. A cell belongs to the first group that applies, in this
order. Each cell is drawn with probability (share of its group) / (number of cells in its group), so rare groups
are seen several times per epoch and common ones not every time. Small buds counted more heavily in the loss (weight
√(700 px / bud area), between 1 and 2). To make the model robust, the training regions were randomly mirrored
along the anti-diagonal, the brightness and contrast of DIC (× 0.8–1.25, offset −0.1 to 0.1) and DAPI
(× 0.8–1.25) were varied, Gaussian noise was added to the DIC layer (σ 0–0.05), and the centre dot was moved by
up to 2 pixels. The bud and body maps were trained with 10 × per-pixel binary cross-entropy plus the Dice
overlap loss (body map weight 0.5), and the cell class with cross-entropy (weight 0.5, label smoothing 0.05).
Training ran for 30 epochs (batch size 16, AdamW optimiser with weight decay 10⁻⁴,
one-cycle learning-rate schedule with a maximum of 3 × 10⁻⁴, mixed precision) on one GPU. The released
weights are those of the last epoch (30), used with a claim cut-off of 0.5; both were fixed before testing.

**Evaluation.** Every cell of 14 test fields of imaging day 24-09-09 (YET915 BR5, YET916 BR4, YET921 BR4)
was annotated again by hand, starting from the Omnipose outlines only and without any model output: buds were
painted and joined to their mother, debris was removed and wrong outlines corrected. One case was scored per
Omnipose object not touching the image border (2,404 cells). After a first scoring round, every cell where
either model disagreed with this annotation in a way that changes the score (254 cells) was inspected again:
in 144 the annotation was confirmed, in 67 it was corrected, and 43 that could not be decided were removed,
leaving 2,361 cells (348 budded). Because only disagreements were re-checked, the scores may be slightly
optimistic. A predicted bud counts as correct when it is given to the right mother and covers at least half of
the true bud, with the true bud covering at least 30% of the predicted one. Precision is the fraction of
predicted buds that are correct, recall the fraction of true buds that are found, and the pair F1 their
harmonic mean; 95% confidence intervals were obtained by resampling the cells 1,000 times (bootstrap), and
for recall per bud size with the Wilson method.

| | Pair F1 (95% CI) | Precision | Recall | Recall, buds < 1 µm² | 1–2 µm² | 2–4 µm² | > 4 µm² |
|---|---|---|---|---|---|---|---|
| Model, DIC + DAPI | 0.75 (0.71–0.79) | 0.80 | 0.72 | 0.78 | 0.75 | 0.80 | 0.61 |
| Model, DIC + DAPI + *ASH1/CLB2* | 0.80 (0.76–0.83) | 0.91 | 0.72 | 0.71 | 0.78 | 0.78 | 0.65 |

Both models find about seven in ten buds on an unseen imaging day; the *ASH1/CLB2* model places fewer buds on
cells without one (20 against 49 of 1,882 unbudded cells). To check that the fixed epoch and cut-off do not
drive the result, every saved epoch (2–30) was also scored at cut-offs 0.2–0.7: from epoch 22 on, pair F1
ranged 0.71–0.77 (DIC + DAPI) and 0.76–0.82 (*ASH1/CLB2*), and the best cut-off would have raised F1 at
epoch 30 by at most 0.01. Large buds remain the weakest class: there the mother–bud relation can be confused
with two neighbouring cells, and part of a large bud can fall outside the 192-pixel region. The flagged cells
are therefore meant to be checked by eye. As a biological check, the fraction of *ASH1/CLB2*, *SRL1* and *SUN4*
mRNAs in the bud was measured on the same test fields with the annotated buds and with each model's buds
(pooled fraction, annotation vs DIC + DAPI / *ASH1/CLB2* model: *ASH1/CLB2* 0.63 vs 0.57 / 0.62; *SRL1* 0.37
vs 0.32 / 0.36; *SUN4* 0.16 vs 0.14 / 0.12).

### Processing speed

Background subtraction and the Laplacian-of-Gaussian filtering of spot detection
were re-implemented in PyTorch so that they can run on a GPU, using the same
kernels, border handling and integer rounding as Big-FISH. Local-maximum
detection, thresholding (fixed or automatic) and duplicate removal use Big-FISH's
own functions. On two test images (four channel stacks at fixed thresholds of 30–80,
and two with the automatic threshold) this gave identical spots and thresholds
to Big-FISH. Background-subtracted
intensities differed by at most one grey level in about 0.01% of voxels. Background
subtraction is restricted to the in-focus z-range plus a margin of 4σ<sub>z</sub>,
which leaves the result within that range unchanged. Images can also be processed
in parallel. The decomposition of dense regions builds its reference spot from up to
2,000 randomly chosen spots; this choice now uses a fixed random seed, so that
repeated runs give identical decomposed spots.

## Software

Image processing used Big-FISH (spot detection, decomposition, focus and
background functions), Omnipose / Cellpose (`cyto2_omni` and `nuclei` models) for
segmentation, scikit-image and SciPy for morphology and image transforms, and
Bio-Formats (via bioio) for reading Olympus `.vsi` files. Interactive review and
correction used napari with the `napari-buddy` plugin. Downstream analysis and
plotting used pandas, seaborn and matplotlib. (Insert package version numbers for
your environment.)
