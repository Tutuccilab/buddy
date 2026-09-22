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

## Software

Image processing used Big-FISH (spot detection, decomposition, focus and
background functions), Omnipose / Cellpose (`cyto2_omni` and `nuclei` models) for
segmentation, scikit-image and SciPy for morphology and image transforms, and
Bio-Formats (via bioio) for reading Olympus `.vsi` files. Interactive review and
correction used napari with the `napari-buddy` plugin. Downstream analysis and
plotting used pandas, seaborn and matplotlib. (Insert package version numbers for
your environment.)
