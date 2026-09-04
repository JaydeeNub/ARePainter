# TreeMasks
<img width="1045" height="490" alt="image" src="https://github.com/user-attachments/assets/0f113f32-6507-44d5-999f-9025d7f1ca74" />

Command-line tool that reads Arma Reforger (Enfusion World Editor) `.layer` files,
finds every placed tree, and renders one top-down 8-bit grayscale PNG mask per tree
category (by default `coniferous.png` and `deciduous.png`). Optionally it also writes
the forest floor between the crowns per forest type and a combined, inverted
"not forest" mask. 

This tool is meant to be used with new terrains as it re-paints whole map. Made by my friend and his friend(claude :) ), but if you need support with usage feel free to reach out to me on official Arma discord

```
.layer files -> parser -> classifier -> coordinate mapper -> renderer -> PNG masks
```

Typical use: masks for terrain texturing, ground clutter or any other tool that needs to
know where the forests of a map are.

## Requirements

* Python 3.10 or newer, 64-bit. The default 31501 x 31501 canvases need about 1 GB of
  RAM per tree category.

## Install

```
git clone https://github.com/JaydeeNub/ARePainter.git
cd TreeMasks
pip install .                          # installs the `treemasks` command
```

Without installing, `pip install -r requirements.txt` and run `python -m treemasks`
from the repository folder instead. For development use `pip install -e .[dev]`, which
adds pytest.

## Usage

```
treemasks --input yourLayerName.layer                       # uses ./config.yaml, writes to ./output
treemasks --input ./layers --output-dir masks               # folders are searched recursively
treemasks --input a.layer b.layer --marker-size 3=9 --report-unknown
treemasks --input ./layers --low-memory --json-report masks/report.json
treemasks --help
```

`.layer` files are the text files the World Editor saves for each layer of a world, in
the world's folder next to its `.ent` file. Copy `.layer` file into this repository folder or Point `--input` at one file, several `.layer` files or
a folder; folders are searched recursively for `parser.file_pattern` (`*.layer`).

A configuration file is required. `config.yaml` in the current directory is used by
default; pass `--config FILE` for another one (YAML or JSON).

### Output files

All masks are 8-bit grayscale PNGs of `output.width` x `output.height` pixels, background
0 and painted pixels `rendering.marker_value` (255 by default). They are written to
`output.directory`:

| File | Content | When |
| --- | --- | --- |
| `<category>.png` | filled circles at every rendered tree of that category | always |
| `<category>_gaps.png` or `<category>_area.png` | forest floor between the crowns, or the solid forest area | extent step on |
| `combined_inverse.png` | every mask together and inverted | `combined_inverse.enabled` |




## Configuration (`config.yaml`)

The shipped `config.yaml` is fully commented. In short:

```yaml
world:              # rectangle of terrain the images cover, in metres
  min_x: 0          # x = the position component picked by x_axis (default: file X, east)
  max_x: 32000
  min_y: 0          # y = the position component picked by y_axis (default: file Z, north)
  max_y: 32000
output:
  width: 31501      # pixels; one canvas = width x height bytes
  height: 31501
  directory: output
  compress_level: 4
coordinate_system:
  x_axis: 0         # index into "coords X Y Z": 0 = X (east), 1 = Y (height), 2 = Z (north)
  y_axis: 2         # x_axis runs left -> right, y_axis top -> bottom; they must differ
  flip_x: false     # mirror horizontally: world max_x becomes the left-most column
  flip_y: true      # mirror vertically: world max_y (north) becomes the top row
rendering:
  marker_value: 255
  marker_sizes: {0: 0, 1: 2, 2: 4, 3: 6}   # size category -> radius in px; 0 is never drawn
extent:                                    # optional forest-extent step, see below
  enabled: true                            # true: use the rendered markers as canopy
  mode: gaps                               # gaps = floor between crowns, area = solid forest
  mask: null                               # optional canopy texture (also enables the step)
  threshold: 128
  close_radius: 5
  max_hole: 64
  max_distance: 50
  tile: 1024
  suffix: null                             # null = "_gaps" / "_area"
  orientation: output
combined_inverse:                          # union of every final mask, hard-inverted
  enabled: true
  filename: combined_inverse.png
parser:
  file_pattern: "*.layer"
  entity_classes: [Tree]                   # only these classes are tree candidates
classifier:
  size_regex: '_(?P<size>\d)[a-z]*(?:_[a-z]+)*\.et$'
  exclude: ["*_fallen*", "*_stump_*", "*_stem_*", "*_branch_*"]
trees:                                     # category -> glob patterns on the asset file name
  coniferous: ["t_picea_abies_*", "t_piceaabies_*", "t_larix_decidua_*"]
  deciduous:  ["t_betula_pendula_*", "t_sorbus_aucuparia_*", "t_tilia_cordata_*", "t_carpinus_betulus_*"]
```

Categories are free-form: add a `bushes:` entry with its own patterns and a `bushes.png`
is produced. Assets that match no category are reported as unknown (`--report-unknown`
lists them) and skipped.

**Fitting the config to your map.** The world bounds and the flips are map-specific. The
shipped bounds (0 to 32000 m) are an example; set them to your terrain's extent so that
`(max_x - min_x) / (width - 1)` is the pixel size you want. If a mask comes out mirrored
against the map's own terrain textures, toggle `flip_x` or `flip_y`; if it is offset or
scaled, check `world`.

### Command-line overrides

Most settings have a matching flag, and the flag wins over the file:

| Flag | Overrides |
| --- | --- |
| `--world-min-x/--world-max-x/--world-min-y/--world-max-y` | `world.*` bounds (metres) |
| `--width/--height` | `output.width/height` (pixels) |
| `--flip-x` / `--no-flip-x`, `--flip-y` / `--no-flip-y` | `coordinate_system.flip_x`, `coordinate_system.flip_y` |
| `--marker-size SIZE=RADIUS` (repeatable) | one entry of `rendering.marker_sizes` |
| `--marker-value` | `rendering.marker_value` |
| `--output-dir`, `--compress-level`, `--file-pattern` | `output.directory`, `output.compress_level`, `parser.file_pattern` |
| `--extent`, `--extent-mode`, `--extent-mask`, `--extent-threshold`, `--extent-close-radius`, `--extent-max-hole`, `--extent-max-distance`, `--extent-orientation`, `--no-extent` | the `extent` section (see below) |
| `--combined-inverse` / `--no-combined-inverse`, `--combined-inverse-file` | the `combined_inverse` section (see below) |

Only editable in the file: `coordinate_system.x_axis` / `y_axis`, `trees`, the
`classifier` section, `parser.entity_classes`, `extent.tile` and `extent.suffix`.

Diagnostics flags: `--report-unknown` lists unknown and excluded asset names,
`--json-report FILE` writes the full counters plus the effective configuration as JSON,
`--low-memory` keeps only one canvas in RAM at a time, `--verbose` shows progress and
every parser warning, `--quiet` prints only errors and the summary.

## How the `.layer` format is interpreted

* The file is a brace-delimited text tree, not XML. `$grp Class : "prefab" { {..} {..} }`
  declares a group of instances of one prefab; `Class : "prefab" { ... }` is a single entity;
  an anonymous `{ ... }` block inside an entity holds its children; named blocks such as
  `Points { ... }` are properties and are ignored.
* Trees appear as `$grp Tree : "{GUID}PrefabLibrary/.../t_picea_abies_2sw.et"` groups (and
  occasionally single `Tree` entities), usually nested inside a `ForestGeneratorEntity`
  that sits inside a `PolylineShapeEntity` instance. Trees placed directly at the top
  level work as well.
* `coords X Y Z` is a **local** position relative to the parent entity. Enfusion uses
  X = east, Y = up, Z = north, so the world position of a tree is the sum of the
  polyline, generator and tree coordinates, and the mask uses X and Z. If a parent carries
  a non-zero yaw (`angles pitch yaw roll`), the child's offset is rotated around Y with the
  left-handed convention Enfusion uses. The files inspected while writing the parser
  contained no rotated parents, so that path has had less real-world exercise.
* Asset names encode the size category as `_<digit><variant letters>` right before `.et`
  or an `_aut` style suffix: `t_picea_abies_3dw.et` -> 3, `t_betula_pendula_0_aut.et` -> 0.
  Stumps, stems and branches (`t_picea_abies_stump_03.et`) carry no size and are excluded,
  as are fallen trunks (`t_picea_abies_3d_fallen.et`). Real data also contains the
  mis-spelled `t_piceaabies_3f.et`, which the default config maps to coniferous.
* Bushes (`b_corylus_avellana_*`, `b_rosa_canina_*`) use the `Tree` class too; they are
  reported as unknown unless a category matches them.

## Pixel mapping

```
pixel_x = round((world_x - min_x) / (max_x - min_x) * (width  - 1))
pixel_y = round((world_y - min_y) / (max_y - min_y) * (height - 1))     # world_y = file Z
if flip_x: pixel_x = width  - 1 - pixel_x
if flip_y: pixel_y = height - 1 - pixel_y
```

Trees outside the inclusive world bounds are counted as out-of-bounds and skipped.
Markers are filled circles (`dx² + dy² <= r²`) clipped at the canvas edge; overlapping
markers simply stay at `marker_value`. With `flip_y` on (the shipped default) north is at
the top of the image. Turning both flips on rotates the plain render by 180°.

## Forest extent (the floor between the crowns, per forest type)

The extent step inverts the canopy, but only inside the forest: the output is the ground
between the crowns, bounded by the forest's own jagged outline, with nothing painted
outside the forest. One extra mask per category is written, `<category>_gaps.png`.

```
treemasks --input ./layers --extent                      # canopy = this tool's own tree circles
treemasks --input ./layers --extent-mask canopy.png      # canopy = an existing texture
treemasks --input ./layers --extent --extent-mode area   # the solid forest area instead
```

How it works:

1. **Canopy**: either the texture given with `--extent-mask` (pixels `>= threshold` count as
   canopy) or, without a mask, the union of the rendered tree markers of all categories.
2. **Forest area**: the canopy closed with a disk of `close_radius` px (notches and channels
   narrower than about twice that count as inside), plus every enclosed hole whose bounding
   box is at most `max_hole` px on a side. The outer edge is otherwise the canopy's own
   outline, so it stays as jagged as the source. A large enclosed clearing stays outside.
3. **Ownership**: each forest pixel goes to the category whose nearest rendered tree marker
   is closest (a Voronoi split, so the per-type masks never overlap). Pixels farther than
   `max_distance` px from any marker are left alone; keep that above `max_hole / 2` plus the
   usual tree spacing, or the middle of large gaps stays unpainted (the summary warns).
4. **Output**: in `gaps` mode the canopy is subtracted from the forest area, leaving the
   floor between crowns; in `area` mode the forest area itself is written (`_area.png`).

The plain marker masks are unchanged. Trees that are not rendered (size 0, radius 0,
excluded, unknown) never claim forest. Everything runs tile by tile (`tile` px plus a
halo), and only tiles containing tree markers are processed, so the cost scales with the
forest area rather than the canvas. Because the ownership split needs every category's
markers at once, `--low-memory` is ignored while the extent step is on.

* A texture must be oriented like the finished images (after `flip_x`/`flip_y`). If it was
  made in the un-flipped render orientation, set `orientation: render` and the tool flips
  it for you.
* Memory: a texture adds `width x height` bytes (0.92 GiB at 31501²) on top of the
  canvases, plus a transient copy while decoding and small per-tile buffers.
* The summary reports the canopy source, the forest area, unassigned pixels, and per
  category the marker pixels, painted pixels, share and output path.
* Requires `scipy` (distance transforms and connected-component labelling).

## Combined inverse

With `combined_inverse.enabled` (or `--combined-inverse`) one more PNG is written,
`combined_inverse.png` by default. It is the pixel-wise OR of every output mask, hard-inverted:
the plain category masks and, when the extent step is on, the extent masks plus the forest
area the extent step worked out. A pixel painted anywhere becomes 0, so each forest is one
solid black shape with the canopy's outline; a pixel painted nowhere becomes
`marker_value`. There are no intermediate gray levels. The file has the same size and mode
as the other masks. It is computed in place on the category canvases after they are saved;
the plain masks and the forest area are carried along as packed bits (a few MB for a sparse
forest), so no extra canvas is allocated. With `--low-memory` one running union canvas is kept.

## Memory and performance

* Files are streamed line by line; only the chain of open blocks is kept, and each tree
  is plotted as soon as its closing brace is read.
* Each canvas is `width x height` bytes (31501² = 0.92 GiB). All category canvases are
  held at once by default; `--low-memory` renders one category per pass over the inputs.
* The summary reports files, lines, entities, per-category/size counts, skipped size-0
  trees, excluded/unknown assets, out-of-bounds trees, parser warnings, canvas memory,
  peak RSS and wall time.

## Development

```
pip install -e .[dev]
python -m pytest                                    # unit and end-to-end tests on synthetic layer data
```

Modules: `treemasks/parser.py`, `classifier.py`, `coordinate_mapper.py`, `renderer.py`,
`extent.py`, `config.py`, `diagnostics.py`, `pipeline.py`, `cli.py`.
