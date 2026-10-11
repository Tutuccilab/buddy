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
of the smoothed profile. The two lowest in-focus slices were additionally
excluded from spot detection, because spots detected there were dominated by
out-of-focus background.

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

**Input.** For every segmented cell a square image region of 192 × 192 pixels (12.5 × 12.5 µm) centred on
the cell is cut out, with three layers: the DIC image, the maximum projection of the DAPI (nucleus) stain, and
a soft dot (Gaussian, σ = 4 pixels) marking the centre of the cell in question. The model is deliberately not
given the Omnipose outline of the cell: outlines that cut through a bud, or merge two cells, would otherwise
mislead it. A second version of the model also receives the maximum projection of the *ASH1/CLB2* smFISH
signal, a transcript that accumulates in the bud during mitosis. Because a DIC image looks different depending
on the direction of its shading, all DIC regions are first rotated so that the shading falls in the same
direction. Intensities are scaled per image between the 0.5th and 99.8th percentile; the *ASH1/CLB2* layer is
also log-transformed so that bright transcription sites do not dominate.

**Network.** The model is a U-Net, a convolutional neural network that outputs one value per pixel, with a
ResNet-34 encoder initialised from weights pre-trained on everyday photographs (ImageNet) and implemented with
the segmentation-models-pytorch library. It has three outputs: (i) a map giving, for every pixel, the
probability that it belongs to the bud of the cell in the centre, (ii) a map of the cell's own body (used only
during training, where it forces the model to locate the cell from the dot), and (iii) the probability that
this cell is unbudded, budded, or a bud. The input is enlarged twofold inside the network so that small buds
are covered by more pixels.

**From per-cell answers to mother–bud pairs.** Each cell "claims" the most confident region of its bud map
(at least 15 pixels with probability > 0.5). The strength of a claim is the cell's probability of being
budded multiplied by its mean bud probability over that region. Claims below a cut-off are discarded. When
two cells claim the same bud (claimed regions overlapping by more than half), both are scored on exactly the
same pixels and the stronger claim wins. A cell cannot be both a mother and a bud. A claimed region that lies
mostly (≥ 60%) inside one Omnipose object other than the mother is replaced by that whole object ("snap"): the
model decides which bud belongs to which mother, and the Omnipose outline, which is usually more complete for
large buds than the model's bud map, provides the outline. The final prediction is the average of the
original region and its mirror image along the anti-diagonal, a transformation that keeps the DIC shading
direction intact. Three situations are flagged for manual checking in napari: two claims within 0.15 of each
other (close call), accepted claims within 0.1 above the cut-off (weak pair) or rejected claims within 0.1
below it (possible missed bud), and cells that are at least 90% likely to be a bud but were claimed by no mother.

**Training data.** The models learned from the manual mother–bud annotations made for the analyses in
this study (step 4 of the pipeline). The images (fields of view) were divided, whole fields at a time, into
a training set (81 fields), a validation set (10 fields) and a test set that was never used during development
(20 fields); 21 further fields from one imaging session (22-04-29) were not used, because their initial
Omnipose segmentations were poor and had to be corrected by hand. The annotations were corrected in two
rounds before training. First, they were checked against the *ASH1/CLB2* signal in the wild-type strains:
cells annotated as unbudded but with *ASH1/CLB2* levels at least eight times those of a typical unbudded cell
in the same image (suggesting a missed bud), and cells annotated with a large bud (> 4 µm²) but G1-like
*ASH1/CLB2* levels (suggesting two separate cells annotated as a pair); all 232 such cells were re-inspected
by eye. Second, because the manual annotation misses many small buds (see Evaluation), a model trained on the
uncorrected annotations was used to find candidate buds: every training cell annotated as unbudded in which an
early training checkpoint (before the model had learned to reproduce the omissions) saw a bud attached to the
cell. All 698 candidates were inspected by eye in random order; 294 were real buds and were added to the
annotation, 216 could not be decided and were left as annotated. The final training set contained 7,391
Omnipose cells (5,109 unbudded, 1,835 budded, 447 buds). Cells touching the image edge were left out. Small
Omnipose objects (< 4 µm²) that the segmentation clean-up had discarded and that were not annotated but touch a
training cell were treated as "ignore regions": they may be buds the annotation missed, so the bud map was not
graded there. Cells in late mitosis (a bud that already contains its own full-size nucleus) were kept; leaving
them out, as in an earlier version, made the model miss large buds. For the *ASH1/CLB2* model the *she2* and
*she3* deletion strains were left out, because in these mutants *ASH1* mRNA is not transported to the bud, so
the signal would be misleading (4,544 training cells).

**Training.** Each training step drew cells in fixed proportions so that the rare cases were seen often
enough: 40% budded mothers, 25% unbudded cells touching another cell's bud (where the correct answer is
"this bud is not mine"), 15% buds and 20% other unbudded cells. Small buds counted more heavily in the loss
(weight √(700 px / bud area), at most twofold). To make the model robust, the training regions were randomly
flipped and rotated in ways that keep the DIC shading direction, their brightness and contrast were varied
(× 0.8–1.25), noise was added to the DIC layer, and the centre dot was moved by up to 2 pixels. The bud and
cell maps were trained with a combination of per-pixel binary cross-entropy and the Dice overlap score (cell
map weight 0.5), and the cell class with cross-entropy (weight 0.5). Training ran for 30 rounds (epochs) of
3,000 cells each (batch size 16, AdamW optimiser, one-cycle learning-rate schedule with a maximum of
3 × 10⁻⁴). A copy of the model was saved every two epochs.

**Evaluation.** Because the manual annotations themselves contain mistakes, the models were evaluated on a
separate, carefully re-annotated set of cells from the test fields. Two hundred cells were drawn from the test
fields (only cells whose Omnipose outline matched the annotated cell, and no *she2*/*she3* cells): 80 budded
mothers, 20 per bud size class (< 1, 1–2, 2–4 and > 4 µm²), 40 cells where the earlier pipeline or a model
disagreed with the annotation, and 80 random other cells. These were annotated again by eye in random order,
seeing only the DIC, DAPI and *ASH1/CLB2* images and the Omnipose outlines, not the earlier annotation or any
model output; 198 cells could be scored. The agreement with this reference is given as the F1 score for
mother–bud pairs: the harmonic mean of precision (the fraction of predicted pairs that are correct) and
recall (the fraction of reference pairs that were found; a pair counts as found when the predicted bud covers
at least half of the reference bud and the reference bud at least 30% of the predicted one). Every annotated
cell counts once; because the hard cases were deliberately over-represented, the scores describe this
difficult set, not a typical image (on the randomly drawn cells alone, 95–97% of the cells were classified
correctly). 95% confidence intervals were obtained by resampling the cells 1,000 times (bootstrap).

The saved copy (epoch) and the claim cut-off of each model were chosen on this set. To obtain an unbiased
estimate of the performance, the test fields were split into two halves: the epoch and cut-off were chosen on
one half and the model scored on the other, and vice versa, so that every cell was scored with a choice made
on other cells ("2-fold"). The released weights use the epoch and cut-off chosen on all 198 cells (DIC + DAPI
model: epoch 22, cut-off 0.6; *ASH1/CLB2* model: epoch 16, cut-off 0.5).

| | F1 (95% CI), 2-fold | Precision | Recall | Recall, buds < 1 µm² | 1–2 µm² | 2–4 µm² | > 4 µm² |
|---|---|---|---|---|---|---|---|
| Original manual annotation | 0.72 (0.64–0.80) | 0.75 | 0.69 | 0.39 | 0.63 | 0.88 | 0.96 |
| Model, DIC + DAPI | 0.80 (0.73–0.86) | 0.79 | 0.81 | 0.86 | 0.78 | 0.81 | 0.81 |
| Model, DIC + DAPI + *ASH1/CLB2* | 0.82 (0.75–0.88) | 0.85 | 0.79 | 0.86 | 0.78 | 0.88 | 0.69 |

Both models recovered more mother–bud pairs than the original manual annotation of the same cells,
mainly because they find the smallest buds, which the annotation mostly missed; the differences between the
two models are within the confidence intervals. Large buds remain the weakest class: there the mother–bud
relation can be confused with two neighbouring cells, and part of a large bud can fall outside the 192-pixel
region (training with 256-pixel regions did not improve this). The flagged cells are therefore meant to be
checked by eye. Earlier versions of the models, which received the cell outline instead of the centre dot,
were trained on the annotations before the second correction round and left late-mitosis cells out of the
DIC + DAPI model, reached F1 0.68 (DIC + DAPI) and 0.74 (*ASH1/CLB2*) on the same cells with their released settings.

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
in parallel.

## Software

Image processing used Big-FISH (spot detection, decomposition, focus and
background functions), Omnipose / Cellpose (`cyto2_omni` and `nuclei` models) for
segmentation, scikit-image and SciPy for morphology and image transforms, and
Bio-Formats (via bioio) for reading Olympus `.vsi` files. Interactive review and
correction used napari with the `napari-buddy` plugin. Downstream analysis and
plotting used pandas, seaborn and matplotlib. (Insert package version numbers for
your environment.)
