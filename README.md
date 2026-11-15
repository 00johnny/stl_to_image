# STL → patent line drawings

Turns an STL file into clean black-and-white line drawings for a patent filing.
No AI: the lines are computed from the geometry, so every view is exact and
repeatable.

## What it does

- **Hidden-line removal** — only edges you could actually see are drawn.
- **Standard design-patent set** — perspective, front, rear, left, right, top
  and bottom, all orthographic and all at one shared scale.
- **Vector output** — PDF sheets (US letter, USPTO margins) or one SVG per
  figure, editable in Inkscape / Illustrator. PNG for previews.
- **Optional surface shading** — line shading or stippling. Off by default.
- **Large files** — a 3.5 million-triangle STL takes about 10 s to load and
  2–3 s per view on a 2-core machine. No GPU needed.

## Use

Web app:

    ./install.sh
    python app.py            # http://localhost:5050

Command line (handy for big files):

    python patent_drawing.py part.stl                         # 7 views -> patent_drawings.pdf
    python patent_drawing.py part.stl --format svg            # zip of 7 SVGs
    python patent_drawing.py part.stl --views single --azimuth 45 --elevation 30 --format png
    python patent_drawing.py part.stl --up y --rotate 90      # fix orientation

Orientation: Z is "up" and the front view looks at the model along +Y. If your
STL is oriented differently use `--up y` and/or `--rotate 90|180|270`
(the same options are in the web app).

## Settings worth knowing

| Setting | Meaning |
|---|---|
| `crease_angle_deg` | Edges are drawn where surfaces meet at more than this angle. Lower = more lines on curved areas, higher = fewer. |
| `hlr_resolution` | Depth-buffer size used for the visibility test. Raise (up to 6000) for very fine detail. |
| `line_width_outline` / `line_width_detail` | Outline and interior line weights in points. |
| `shading` | `none`, `hatch` or `stipple`. |

## Before filing

The output is an accurate base drawing, not a finished filing. A design patent
normally still needs:

- surface shading that shows contour (the built-in shading is a starting point),
- broken lines for anything shown but not claimed — a legal decision,
- a review against 37 CFR 1.84 / 1.152 by a patent attorney or illustrator.

STL files store triangles, not true curves. If the original CAD model exists,
export the STL at a fine resolution so curved outlines stay smooth.
