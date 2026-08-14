# AutoFigure Edit Contract — J-stage figures

## Scope and status

This contract covers local, editor-only presentation derivatives of the DART
architecture and factorial-result figures. AutoFigure-Edit's text-to-image,
SAM, RMBG, LLM-to-SVG and optimization stages were not invoked. These files
are not new scientific sources and must not be described as AI-generated
figures. The DART derivative has since been superseded by the user-supplied
manual Main Figure 1; the factorial derivative remains active as Main Figure 5.

The canonical generator, frozen evidence bundle and canonical PDF/PNG remain
authoritative. The new `*_autofigure_v1` files are separate, reversible
presentation derivatives.

Tool identity:

- Official AutoFigure-Edit release: `v1.1`, commit
  `7522638e30bda4b3d0d917630596433bf0fd2c28`.
- Local execution checkout:
  `a14889f82b9ed1376b848d8e8eaaf6bca6077033`
  (`v1.1-24-ga14889f`).
- Interface used: bundled SVG-Edit source editor on `localhost`.
- Remote API calls: none.

## Allowed operations

- Add SVG accessibility metadata.
- Adjust typography, line wrapping and panel-title hierarchy.
- Align an entire semantic group, panel, annotation or legend without
  changing its internal geometry.
- Remove an exactly duplicated legend while retaining one complete key.
- Export separate SVG/PDF/PNG derivatives.

## Forbidden operations

- Change a scientific number, unit, label, qualifier or method name.
- Add, remove, reorder or reinterpret model components.
- Change any arrow's source, target, direction or meaning.
- Move a data point, error bar, curve, axis, tick, broken axis or decision
  boundary.
- Embed a raster image or outline all text in a promoted SVG.
- Overwrite canonical source figures.
- Upload unpublished figures or run any generative AutoFigure stage without a
  new authorization and provenance record.
- Run a new experiment or scientific analysis as part of figure polishing.

## Archived DART architecture derivative

The following statements and visual relations are immutable:

- Atom representation: 128 features.
- Element table: 9 normalized attributes.
- Projection: `137 → 128`.
- `E` enrichment is added only at the impurity.
- `P × 3` uses the 5 Å local graph.
- The global Transformer has two layers and four attention heads.
- 12 Å is the radial-grid endpoint, not an attention cutoff.
- `G` is the scalar-gated fusion of atom-attention sum and channelwise max.
- Prediction head: `128 → 128 → 1`; target: formation energy.
- Every arrow connection and direction remains unchanged.
- The structure is a schematic, not a selected material.

Authorized delta:

1. Add SVG `role`, `title`, `desc` and ARIA references.
2. Change four panel headings from 7.2 px regular to 7.4 px weight 600.
3. Wrap “9 normalized attributes” over two lines without changing wording.

The 91 SVG paths and all path attributes must remain exactly identical to the
editable input.

## Active factorial-result derivative (Main Figure 5)

Authorized delta:

1. Delete the duplicate left-panel group `legend_1`.
2. Retain upper-right `legend_2` as the sole shared key.
3. Add SVG `role`, `title`, `desc` and ARIA references.

After removing `legend_1` from the editable input, the following output layers
must match exactly:

| Element | Count |
|---|---:|
| `path` | 65 |
| `use` | 121 |
| `rect` | 2 |
| `text` | 40 |
| `image` | 0 |

No point, confidence interval, axis, tick, label, selected-variant highlight
or scientific value may change.

## Policy for the remaining figures

| Figure | AutoFigure generative chain | Editor-only decision |
|---|---|---|
| Main Fig. 3 — protocol | NO-GO | Defer; contains quantitative panels and an embedded heatmap raster. |
| Main Fig. 6 — transfer | NO-GO | Keep canonical; broken-axis geometry is high risk. |
| Main Fig. 7 — uncertainty | NO-GO | Keep canonical; dense curves, hexbin and colorbar are locked. |
| Main Fig. 8 — screening | NO-GO | Keep canonical; dense scatter and ECDF geometry are locked. |
| Supplementary Fig. S1 — heterogeneity | NO-GO | Keep canonical in this pass. |

## Full-pipeline block

The full AutoFigure pipeline is blocked and was not attempted because the
local tool has no configured image/LLM provider credential, no remote SAM key
or local SAM3, and no locally available RMBG-2.0 weights. External
transmission of the unpublished figure has also not been authorized. Secrets
must never be written to this repository or pasted into the manifest.

## Promotion gate

- [x] Canonical source files remain untouched.
- [x] Editor source and persisted SVG hashes match.
- [x] The archived architecture paths and active factorial locked data layers
      pass exact comparison.
- [x] Promoted SVG candidates contain live text and zero `<image>` elements.
- [x] Candidate PDFs contain embedded fonts, Unicode mappings and no raster
      images.
- [x] 300 dpi PNG previews render without clipping or overlap.
- [x] LaTeX main-paper build, caption and reference checks pass with the
      candidates selected.
- [x] Existing numeric audit and targeted figure tests remain passing after
      integration.
- [ ] A local milestone commit records the accepted derivatives, manifest and
      catalog together.
