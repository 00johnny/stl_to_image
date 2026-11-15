"""
patent_drawing.py — Convert STL files to patent-style line drawings.

Pipeline (all geometry, no AI, fully deterministic):
  1. Load STL -> trimesh, build edge adjacency once
  2. For each view: project vertices (orthographic by default)
  3. Classify edges: silhouette (front/back boundary), crease (sharp angle),
     open/non-manifold boundary
  4. Rasterise every triangle into a depth buffer (vectorised numpy)
  5. Hidden-line removal: sample each candidate edge against the depth buffer
     and keep only the visible runs
  6. Chain visible pieces into polylines
  7. Optional image-space surface shading (line hatching or stippling)
  8. Render with matplotlib -> PNG, SVG or PDF (vector)

A "standard" view set produces the seven views normally filed with a design
patent (perspective, front, rear, left, right, top, bottom), all orthographic
and all at one shared scale.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, field

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

matplotlib.rcParams["svg.fonttype"] = "none"   # keep labels as editable text
matplotlib.rcParams["pdf.fonttype"] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

class DrawingParams:
    """All user-adjustable parameters."""

    def __init__(
        self,
        # What to draw
        view_mode: str = "single",         # "single" or "standard" (seven views)
        output_format: str = "png",        # "png", "svg" or "pdf"
        # Camera (single view, and the perspective view of the standard set)
        view_azimuth: float = 45.0,        # 0 = front, 90 = right, 180 = rear, 270 = left
        view_elevation: float = 30.0,      # 90 = top, -90 = bottom
        projection: str = "orthographic",  # "orthographic" or "perspective"
        fov_deg: float = 35.0,             # perspective only
        # Model orientation
        up_axis: str = "z",                # which STL axis points up: "z" or "y"
        model_rotation_deg: float = 0.0,   # spin about the up axis to choose the "front"
        # Lines
        line_width_outline: float = 1.2,   # pt — silhouette edges
        line_width_detail: float = 0.7,    # pt — interior crease edges
        crease_angle_deg: float = 25.0,    # min angle between faces to draw an edge
        min_line_px: float = 3.0,          # drop visible fragments shorter than this
        hlr_resolution: int = 2400,        # depth-buffer size (long side, px)
        # Surface shading
        shading: str = "none",             # "none", "hatch" or "stipple"
        light_azimuth: float = 135.0,      # on the page: 0 = right, 90 = up, 135 = upper left
        light_elevation: float = 45.0,     # toward the viewer
        hatch_angle: float = 45.0,         # degrees
        hatch_spacing: float = 5.0,        # pt between the sparsest hatch lines / stipple cell
        # Output
        output_dpi: int = 300,
        image_width_inches: float = 6.0,
        image_height_inches: float = 6.0,
        fit: str = "sphere",               # "sphere": same scale for every angle; "tight": fill canvas
        views_per_sheet: int = 2,          # PDF sheets of the standard set: 1 or 2
        show_figure_label: bool = True,
        figure_label: str = "FIG. 1",
        # Legacy fields from the first version (accepted, ignored)
        hatch_spacing_dense: float = 3.0,
        hatch_spacing_sparse: float = 14.0,
        hatch_threshold: float = 0.15,
        background_white: bool = True,
    ):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    @classmethod
    def from_dict(cls, d: dict) -> "DrawingParams":
        p = cls()
        for key, val in d.items():
            if hasattr(p, key):
                setattr(p, key, val)
        if p.projection == "isometric":      # old name for orthographic
            p.projection = "orthographic"
        return p


# (name, azimuth, elevation). None = use the user's view angles.
STANDARD_VIEWS = [
    ("perspective", None, None),
    ("front", 0.0, 0.0),
    ("rear", 180.0, 0.0),
    ("left side", 270.0, 0.0),
    ("right side", 90.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
]

# USPTO sheet (37 CFR 1.84): US letter, margins top 1", left 1", right 5/8", bottom 3/8"
SHEET_W, SHEET_H = 8.5, 11.0
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 1.0, 0.625, 1.0, 0.375
LABEL_SPACE = 0.38  # inches reserved under each view for "FIG. n"


# ─────────────────────────────────────────────────────────────────────────────
# Mesh preparation (done once per STL, reused for every view)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedMesh:
    vertices: np.ndarray        # (N,3) centred on the bounding-box centre
    faces: np.ndarray           # (F,3)
    face_normals: np.ndarray    # (F,3)
    centroids: np.ndarray       # (F,3)
    radius: float               # bounding-sphere radius
    adj_faces: np.ndarray       # (A,2) faces sharing an edge
    adj_edges: np.ndarray       # (A,2) vertex indices of that edge
    adj_angles: np.ndarray      # (A,) angle between the two face normals
    odd_edges: np.ndarray       # (B,2) edges with 1 or 3+ faces (open / non-manifold)


def prepare_mesh(stl_bytes: bytes, params: DrawingParams) -> PreparedMesh:
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("The STL file contains no usable triangles.")

    # Inconsistent winding would create false outlines; repair when affordable.
    try:
        if len(mesh.faces) <= 1_500_000 and not mesh.is_winding_consistent:
            mesh.fix_normals()
    except Exception:
        pass

    V = np.asarray(mesh.vertices, dtype=np.float64)
    if str(params.up_axis).lower() == "y":
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)   # Y-up -> Z-up
    rot = math.radians(float(params.model_rotation_deg))
    if abs(rot) > 1e-12:
        cs, sn = math.cos(rot), math.sin(rot)
        V = np.stack([cs * V[:, 0] - sn * V[:, 1], sn * V[:, 0] + cs * V[:, 1], V[:, 2]], axis=1)
    V = V - (V.max(axis=0) + V.min(axis=0)) / 2.0

    F = np.asarray(mesh.faces)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 0
    if not keep.all():
        F, tri, n, ln = F[keep], tri[keep], n[keep], ln[keep]
    n = n / ln[:, None]

    # Edge -> face adjacency, fully vectorised
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    face_of = np.tile(np.arange(len(F)), 3)
    key = E[:, 0].astype(np.int64) * (int(V.shape[0]) + 1) + E[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[start, len(ks)])
    two = start[count == 2]
    adj_faces = np.stack([face_of[order[two]], face_of[order[two + 1]]], axis=1)
    adj_edges = E[order[two]]
    dots = np.einsum("ij,ij->i", n[adj_faces[:, 0]], n[adj_faces[:, 1]])
    adj_angles = np.arccos(np.clip(dots, -1.0, 1.0))
    odd_edges = E[order[start[count != 2]]]

    return PreparedMesh(
        vertices=V, faces=F, face_normals=n, centroids=tri.mean(axis=1),
        radius=float(np.linalg.norm(V, axis=1).max()),
        adj_faces=adj_faces, adj_edges=adj_edges, adj_angles=adj_angles,
        odd_edges=odd_edges,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def camera_basis(azimuth_deg: float, elevation_deg: float):
    """Return (right, up, toward_camera) unit vectors. Z is up, front looks along +Y."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    right = np.array([math.cos(az), math.sin(az), 0.0])
    back = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    up = np.cross(back, right)
    return right, up, back


def _project(prep: PreparedMesh, az: float, el: float, params: DrawingParams):
    """Screen x, y, a depth value that is linear in screen space, and per-face facing."""
    r, u, c = camera_basis(az, el)
    V = prep.vertices
    x, y, zc = V @ r, V @ u, V @ c
    if params.projection == "perspective":
        fov = math.radians(min(max(float(params.fov_deg), 5.0), 120.0))
        D = prep.radius / math.sin(fov / 2.0) * 1.05
        w = D - zc
        sx, sy, depth = x * D / w, y * D / w, -1.0 / w
        facing = np.einsum("ij,ij->i", prep.face_normals, c * D - prep.centroids) > 1e-9 * D
    else:
        sx, sy, depth = x, y, -zc
        facing = prep.face_normals @ c > 1e-9
    return sx, sy, depth, facing, (r, u, c)


# ─────────────────────────────────────────────────────────────────────────────
# Depth-buffer rasteriser (vectorised scanline spans — cost ~ pixels covered)
# ─────────────────────────────────────────────────────────────────────────────

def rasterize(px, py, depth, faces, W, H, want_ids=False, budget=3_000_000):
    zbuf = np.full(W * H, np.inf)
    ids = np.full(W * H, -1, dtype=np.int32) if want_ids else None
    tiny = 1e-6   # widen spans a hair so shared edges never leave pinholes

    X, Y, Z = px[faces], py[faces], depth[faces]
    ex1, ey1, ez1 = X[:, 1] - X[:, 0], Y[:, 1] - Y[:, 0], Z[:, 1] - Z[:, 0]
    ex2, ey2, ez2 = X[:, 2] - X[:, 0], Y[:, 2] - Y[:, 0], Z[:, 2] - Z[:, 0]
    area2 = ex1 * ey2 - ex2 * ey1
    r0 = np.maximum(np.ceil(Y.min(axis=1) - 0.5 - tiny), 0).astype(np.int64)
    r1 = np.minimum(np.floor(Y.max(axis=1) - 0.5 + tiny), H - 1).astype(np.int64)
    nrows = r1 - r0 + 1
    # Edge-on triangles project to (almost) nothing and have meaningless depth
    # gradients — skip them rather than let them poison the buffer.
    longest2 = np.maximum(ex1 ** 2 + ey1 ** 2, np.maximum(ex2 ** 2 + ey2 ** 2,
                          (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2))
    tris = np.flatnonzero((np.abs(area2) > 1e-7 * longest2) & (np.abs(area2) > 1e-12) & (nrows > 0))
    zlo, zhi = Z.min(axis=1), Z.max(axis=1)
    if len(tris) == 0:
        return zbuf.reshape(H, W), (ids.reshape(H, W) if want_ids else None)
    with np.errstate(divide="ignore", invalid="ignore"):
        dzdx = (ez1 * ey2 - ez2 * ey1) / area2
        dzdy = (ez2 * ex1 - ez1 * ex2) / area2

    cost = np.cumsum(nrows[tris] + np.abs(area2[tris]) / 2.0)
    i0 = 0
    while i0 < len(tris):
        base = cost[i0 - 1] if i0 else 0.0
        i1 = max(i0 + 1, int(np.searchsorted(cost, base + budget, side="right")))
        t = tris[i0:i1]
        i0 = i1

        nr = nrows[t]
        npairs = int(nr.sum())
        T = np.repeat(np.arange(len(t)), nr)              # pair -> triangle (local)
        row = r0[t][T] + (np.arange(npairs) - np.repeat(np.cumsum(nr) - nr, nr))
        yc = row + 0.5
        tt = t[T]
        xl = np.full(npairs, np.inf)
        xr = np.full(npairs, -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ya, yb, xa, xb = Y[tt, a], Y[tt, b], X[tt, a], X[tt, b]
            span = (np.minimum(ya, yb) - tiny <= yc) & (yc <= np.maximum(ya, yb) + tiny) & (ya != yb)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = xa + (yc - ya) * (xb - xa) / (yb - ya)
            xc = np.clip(xc, np.minimum(xa, xb), np.maximum(xa, xb))
            xl = np.where(span, np.minimum(xl, xc), xl)
            xr = np.where(span, np.maximum(xr, xc), xr)
        good = np.isfinite(xl) & np.isfinite(xr)
        c0 = np.zeros(npairs, dtype=np.int64)
        c1 = np.full(npairs, -1, dtype=np.int64)
        c0[good] = np.maximum(np.ceil(xl[good] - 0.5 - tiny), 0).astype(np.int64)
        c1[good] = np.minimum(np.floor(xr[good] - 0.5 + tiny), W - 1).astype(np.int64)
        cnt = np.clip(c1 - c0 + 1, 0, None)
        total = int(cnt.sum())
        if total == 0:
            continue
        zrow = Z[tt, 0] + dzdy[tt] * (yc - Y[tt, 0]) + dzdx[tt] * (c0 + 0.5 - X[tt, 0])

        keep = cnt > 0
        cntk = cnt[keep]
        P = np.repeat(np.flatnonzero(keep), cntk)          # pixel -> pair
        k = np.arange(total) - np.repeat(np.cumsum(cntk) - cntk, cntk)
        idx = row[P] * W + c0[P] + k
        tp = tt[P]
        zs = np.clip(zrow[P] + dzdx[tp] * k, zlo[tp], zhi[tp])
        np.minimum.at(zbuf, idx, zs)
        if want_ids:
            win = zs <= zbuf[idx]
            ids[idx[win]] = tp[win]

    return zbuf.reshape(H, W), (ids.reshape(H, W) if want_ids else None)


def _max_filter3(d: np.ndarray) -> np.ndarray:
    """3x3 maximum filter (background is +inf, so anything next to empty space passes)."""
    p = np.pad(d, 1, constant_values=np.inf)
    H, W = d.shape
    out = d.copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, p[dy:dy + H, dx:dx + W], out=out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-line removal
# ─────────────────────────────────────────────────────────────────────────────

def _visible_segments(ax, ay, az, bx, by, bz, zdil, eps, chunk=4_000_000):
    """Split each edge into ~1px pieces, test each against the depth buffer and
    return the visible runs as (S,2,2) pixel segments plus their source edge index."""
    H, W = zdil.shape
    n = np.maximum(1, np.ceil(np.hypot(bx - ax, by - ay))).astype(np.int64)
    out_seg, out_edge = [], []
    bounds = np.r_[0, np.cumsum(n)]
    e0 = 0
    while e0 < len(n):
        e1 = max(e0 + 1, int(np.searchsorted(bounds, bounds[e0] + chunk, side="right")) - 1)
        nn = n[e0:e1]
        total = int(nn.sum())
        eid = np.repeat(np.arange(e0, e1), nn)
        first = np.repeat(np.cumsum(nn) - nn, nn)
        k = np.arange(total) - first
        ne = n[eid]
        t = (k + 0.5) / ne
        x = ax[eid] + t * (bx - ax)[eid]
        y = ay[eid] + t * (by - ay)[eid]
        z = az[eid] + t * (bz - az)[eid]
        ix = np.clip(x.astype(np.int64), 0, W - 1)
        iy = np.clip(y.astype(np.int64), 0, H - 1)
        vis = z <= zdil[iy, ix] + eps

        change = np.ones(total, dtype=bool)
        change[1:] = (eid[1:] != eid[:-1]) | (vis[1:] != vis[:-1])
        rs = np.flatnonzero(change)
        re = np.r_[rs[1:], total] - 1
        keep = vis[rs]
        rs, re = rs[keep], re[keep]
        if len(rs):
            e = eid[rs]
            t0 = (k[rs] / n[e])[:, None]
            t1 = ((k[re] + 1) / n[e])[:, None]
            A = np.stack([ax[e], ay[e]], axis=1)
            B = np.stack([bx[e], by[e]], axis=1)
            out_seg.append(np.stack([A + t0 * (B - A), A + t1 * (B - A)], axis=1))
            out_edge.append(e)
        e0 = e1
    if not out_seg:
        return np.zeros((0, 2, 2)), np.zeros(0, dtype=np.int64)
    return np.concatenate(out_seg), np.concatenate(out_edge)


def _node_keys(pts: np.ndarray) -> np.ndarray:
    q = np.round(pts.reshape(-1, 2) * 50.0).astype(np.int64)
    return q[:, 0] * 4_000_000 + q[:, 1]


def _chain(segs: np.ndarray, min_len: float, anchors: np.ndarray | None = None,
           max_chain: int = 400_000) -> list:
    """Join segments that share endpoints into polylines. Isolated specks shorter
    than min_len are dropped unless they touch a line of the other class (anchors)."""
    if len(segs) == 0:
        return []
    seglen = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
    segs = segs[seglen > 1e-6]
    if len(segs) == 0:
        return []
    # Edges that land exactly on top of each other in this view (e.g. the top and
    # bottom rim of a vertical wall seen from above) are drawn once.
    k2 = np.sort(_node_keys(segs).reshape(-1, 2), axis=1)
    _, uniq = np.unique(k2, axis=0, return_index=True)
    segs = segs[np.sort(uniq)]
    key = _node_keys(segs)
    ukey, first, inv = np.unique(key, return_index=True, return_inverse=True)
    coords = segs.reshape(-1, 2)[first]
    nodes = inv.reshape(-1, 2)
    deg = np.bincount(inv, minlength=len(first))
    order = np.argsort(inv, kind="stable")
    if len(segs) > max_chain:   # too many to walk in Python — emit as-is
        return list(segs)
    ptr = np.r_[0, np.cumsum(deg)].tolist()
    seg_at = (order // 2).tolist()
    nodes_l, deg_l = nodes.tolist(), deg.tolist()
    S = len(segs)
    visited = [False] * S
    chains = []

    def walk(seg, node):
        path = [node]
        while True:
            visited[seg] = True
            a, b = nodes_l[seg]
            node = b if a == node else a
            path.append(node)
            if deg_l[node] != 2:
                break
            s1, s2 = seg_at[ptr[node]], seg_at[ptr[node] + 1]
            seg = s2 if s1 == seg else s1
            if visited[seg]:
                break
        return path

    for s in range(S):          # open chains: start at junctions / free ends
        if visited[s]:
            continue
        a, b = nodes_l[s]
        if deg_l[a] != 2:
            chains.append(walk(s, a))
        elif deg_l[b] != 2:
            chains.append(walk(s, b))
    for s in range(S):          # what is left are closed loops
        if not visited[s]:
            chains.append(walk(s, nodes_l[s][0]))

    anchored = np.isin(ukey, anchors) if anchors is not None and len(anchors) else np.zeros(len(ukey), bool)
    out = []
    for path in chains:
        pts = coords[path]
        a, b = path[0], path[-1]
        # only isolated specks are dropped; short links between junctions stay
        isolated = a == b or (deg_l[a] <= 1 and deg_l[b] <= 1 and not anchored[a] and not anchored[b])
        if not isolated or np.hypot(*np.diff(pts, axis=0).T).sum() >= min_len:
            out.append(pts)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Image-space surface shading (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _darkness_image(prep, ids, facing, basis, params):
    r, u, c = basis
    n = prep.face_normals
    ncam = np.stack([n @ r, n @ u, n @ c], axis=1)
    ncam[~facing] *= -1.0
    la, le = math.radians(params.light_azimuth), math.radians(params.light_elevation)
    light = np.array([math.cos(le) * math.cos(la), math.cos(le) * math.sin(la), math.sin(le)])
    dark_face = 1.0 - np.clip(ncam @ light, 0.0, 1.0)
    dark = np.zeros(ids.shape)
    m = ids >= 0
    dark[m] = dark_face[ids[m]]
    return dark, m


def _hatch_lines(dark, mask, spacing_px, angle_deg, min_run=4):
    """Straight parallel hatch lines; darker areas get in-between lines too."""
    H, W = dark.shape
    spacing_px = max(float(spacing_px), 2.0)
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    centre = np.array([W / 2.0, H / 2.0])
    half = math.hypot(W, H) / 2.0
    q = spacing_px / 4.0
    offs = np.arange(-half, half, q)
    # every 4th line is the base set, then the half-way set, then the quarter sets
    level = np.where(np.arange(len(offs)) % 4 == 0, 0, np.where(np.arange(len(offs)) % 2 == 0, 1, 2))
    thresh = np.array([0.30, 0.55, 0.80])[level]
    s = np.arange(-half, half, 1.0)
    X = centre[0] + offs[:, None] * nrm[0] + s[None, :] * d[0]
    Y = centre[1] + offs[:, None] * nrm[1] + s[None, :] * d[1]
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    ix, iy = np.clip(X.astype(np.int64), 0, W - 1), np.clip(Y.astype(np.int64), 0, H - 1)
    on = inb & mask[iy, ix] & (dark[iy, ix] > thresh[:, None])
    pad = np.zeros((on.shape[0], 1), dtype=bool)
    diff = np.diff(np.concatenate([pad, on, pad], axis=1).astype(np.int8), axis=1)
    r0, c0 = np.nonzero(diff == 1)
    r1, c1 = np.nonzero(diff == -1)
    c1 = c1 - 1
    keep = (c1 - c0) >= min_run
    r0, c0, c1 = r0[keep], c0[keep], c1[keep]
    return np.stack([np.stack([X[r0, c0], Y[r0, c0]], axis=1),
                     np.stack([X[r0, c1], Y[r0, c1]], axis=1)], axis=1)


def _stipple_dots(dark, mask, cell_px):
    """Jittered-grid stippling; dot density follows darkness. Fixed seed = repeatable."""
    H, W = dark.shape
    cell = max(float(cell_px) / 2.0, 1.5)
    rng = np.random.default_rng(12345)
    gx, gy = np.meshgrid(np.arange(0, W, cell), np.arange(0, H, cell))
    x = gx + rng.random(gx.shape) * cell
    y = gy + rng.random(gy.shape) * cell
    ix, iy = np.clip(x.astype(np.int64), 0, W - 1), np.clip(y.astype(np.int64), 0, H - 1)
    dval = np.clip((dark[iy, ix] - 0.25) / 0.75, 0.0, 1.0) ** 1.5 * 0.85
    keep = mask[iy, ix] & (rng.random(gx.shape) < dval)
    return np.stack([x[keep], y[keep]], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# One view
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ViewResult:
    name: str
    outline: list                 # polylines in view units
    detail: list
    hatch: np.ndarray             # (S,2,2)
    dots: np.ndarray              # (M,2)
    extents: tuple                # xmin, xmax, ymin, ymax
    stats: dict = field(default_factory=dict)

    @property
    def size(self):
        return self.extents[1] - self.extents[0], self.extents[3] - self.extents[2]

    @property
    def centre(self):
        return (self.extents[0] + self.extents[1]) / 2.0, (self.extents[2] + self.extents[3]) / 2.0


def view_extents(prep, az, el, params):
    sx, sy, _, _, _ = _project(prep, az, el, params)
    return float(sx.min()), float(sx.max()), float(sy.min()), float(sy.max())


def compute_view(prep: PreparedMesh, name: str, az: float, el: float,
                 params: DrawingParams, inches_per_unit: float) -> ViewResult:
    sx, sy, depth, facing, basis = _project(prep, az, el, params)
    xmin, xmax, ymin, ymax = float(sx.min()), float(sx.max()), float(sy.min()), float(sy.max())
    res = int(min(max(int(params.hlr_resolution), 400), 6000))
    pad = 3
    s = (res - 2 * pad) / max(xmax - xmin, ymax - ymin, 1e-12)
    W = int(math.ceil((xmax - xmin) * s)) + 2 * pad
    H = int(math.ceil((ymax - ymin) * s)) + 2 * pad
    px, py = (sx - xmin) * s + pad, (sy - ymin) * s + pad

    want_shade = params.shading in ("hatch", "stipple")
    zbuf, ids = rasterize(px, py, depth, prep.faces, W, H, want_ids=want_shade)
    zdil = _max_filter3(zbuf)
    eps = 2e-3 * float(depth.max() - depth.min() + 1e-30)

    # Candidate edges. Creases are tested on both sides of the object: the
    # depth buffer, not face direction, decides what is visible.
    f0, f1 = prep.adj_faces[:, 0], prep.adj_faces[:, 1]
    sil = facing[f0] != facing[f1]
    crease = ~sil & (prep.adj_angles > math.radians(float(params.crease_angle_deg)))
    edges = np.concatenate([prep.adj_edges[sil], prep.odd_edges, prep.adj_edges[crease]])
    n_outline = int(sil.sum()) + len(prep.odd_edges)

    a, b = edges[:, 0], edges[:, 1]
    segs, src = _visible_segments(px[a], py[a], depth[a], px[b], py[b], depth[b], zdil, eps)
    is_outline = src < n_outline
    min_len = float(params.min_line_px)
    to_units = lambda p: (p - pad) / s + np.array([xmin, ymin])
    so, sd = segs[is_outline], segs[~is_outline]
    outline = [to_units(c) for c in _chain(so, min_len, _node_keys(sd))]
    detail = [to_units(c) for c in _chain(sd, min_len, _node_keys(so))]

    hatch, dots = np.zeros((0, 2, 2)), np.zeros((0, 2))
    if want_shade:
        dark, mask = _darkness_image(prep, ids, facing, basis, params)
        spacing_px = float(params.hatch_spacing) / 72.0 / inches_per_unit * s
        if params.shading == "hatch":
            hatch = to_units(_hatch_lines(dark, mask, spacing_px, float(params.hatch_angle)))
        else:
            dots = to_units(_stipple_dots(dark, mask, spacing_px))

    return ViewResult(name, outline, detail, hatch, dots, (xmin, xmax, ymin, ymax),
                      stats={"buffer": (W, H), "candidate_edges": int(len(edges)),
                             "outline_paths": len(outline), "detail_paths": len(detail)})


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _new_page(w_in, h_in, dpi):
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in)
    ax.set_ylim(0, h_in)
    ax.axis("off")
    return fig, ax


def _draw_view(ax, view: ViewResult, cx, cy, scale, params, label=None):
    """Draw a view centred at (cx, cy) inches with `scale` inches per model unit."""
    vx, vy = view.centre
    off = np.array([cx - vx * scale, cy - vy * scale])
    tf = lambda p: p * scale + off
    kw = dict(colors="black", capstyle="round", joinstyle="round")
    if len(view.hatch):
        ax.add_collection(LineCollection(tf(view.hatch), linewidths=0.35, zorder=1, **kw))
    if len(view.dots):
        d = tf(view.dots)
        ax.scatter(d[:, 0], d[:, 1], s=0.5, c="black", linewidths=0, zorder=1)
    if view.detail:
        ax.add_collection(LineCollection([tf(c) for c in view.detail],
                                         linewidths=float(params.line_width_detail), zorder=2, **kw))
    if view.outline:
        ax.add_collection(LineCollection([tf(c) for c in view.outline],
                                         linewidths=float(params.line_width_outline), zorder=3, **kw))
    if label:
        ax.text(cx, cy - view.size[1] * scale / 2.0 - 0.12, label, ha="center", va="top",
                fontsize=12, fontfamily="serif", color="black")


def _save(fig, fmt, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _render_single(view, scale, params, fmt, label):
    w, h = float(params.image_width_inches), float(params.image_height_inches)
    fig, ax = _new_page(w, h, params.output_dpi)
    lab = LABEL_SPACE if label else 0.0
    _draw_view(ax, view, w / 2.0, (h + lab) / 2.0, scale, params, label)
    return _save(fig, fmt, params.output_dpi)


def _single_scale(prep, extents, params):
    w, h = float(params.image_width_inches), float(params.image_height_inches)
    m = 0.25
    lab = LABEL_SPACE if params.show_figure_label else 0.0
    aw, ah = max(w - 2 * m, 0.1), max(h - 2 * m - lab, 0.1)
    if params.fit == "tight":
        return min(aw / max(extents[1] - extents[0], 1e-12), ah / max(extents[3] - extents[2], 1e-12))
    return min(aw, ah) / (2.0 * prep.radius)


def _standard_angles(params):
    return [(n, params.view_azimuth if az is None else az, params.view_elevation if el is None else el)
            for n, az, el in STANDARD_VIEWS]


def _shared_scale(extents_list, cell_w, cell_h):
    mw = max(e[1] - e[0] for e in extents_list)
    mh = max(e[3] - e[2] for e in extents_list)
    return 0.94 * min(cell_w / max(mw, 1e-12), cell_h / max(mh, 1e-12))


def _render_sheets_pdf(views, scale, params, per_sheet):
    buf = io.BytesIO()
    uw = SHEET_W - MARGIN_L - MARGIN_R
    uh = SHEET_H - MARGIN_T - MARGIN_B
    cell_h = uh / per_sheet
    with PdfPages(buf) as pdf:
        for i in range(0, len(views), per_sheet):
            fig, ax = _new_page(SHEET_W, SHEET_H, 72)
            for j, v in enumerate(views[i:i + per_sheet]):
                top = SHEET_H - MARGIN_T - j * cell_h
                cy = top - (cell_h - LABEL_SPACE) / 2.0
                _draw_view(ax, v, MARGIN_L + uw / 2.0, cy, scale, params, f"FIG. {i + j + 1}")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    return buf.getvalue()


def _render_contact_png(views, params):
    """All views on one preview image, shared scale."""
    cols = 4
    rows = math.ceil(len(views) / cols)
    cw, ch = 3.4, 3.4
    scale = _shared_scale([v.extents for v in views], cw - 0.3, ch - LABEL_SPACE - 0.3)
    fig, ax = _new_page(cols * cw, rows * ch, 160)
    for i, v in enumerate(views):
        r_, c_ = divmod(i, cols)
        cx = (c_ + 0.5) * cw
        cy = rows * ch - (r_ + 0.5) * ch + LABEL_SPACE / 2.0
        _draw_view(ax, v, cx, cy, scale, params, f"FIG. {i + 1}")
        ax.text(cx, cy - v.size[1] * scale / 2.0 - 0.34, v.name, ha="center", va="top",
                fontsize=8, color="#666666")
    return _save(fig, "png", 160)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

MIMETYPES = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf",
             "zip": "application/zip"}


def generate(stl_bytes: bytes, params: DrawingParams, prep: PreparedMesh | None = None):
    """Returns (data_bytes, mimetype, filename)."""
    fmt = str(params.output_format).lower()
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError("output_format must be png, svg or pdf")
    prep = prep or prepare_mesh(stl_bytes, params)

    if params.view_mode != "standard":
        ext = view_extents(prep, params.view_azimuth, params.view_elevation, params)
        scale = _single_scale(prep, ext, params)
        view = compute_view(prep, "view", params.view_azimuth, params.view_elevation, params, scale)
        label = params.figure_label if params.show_figure_label else None
        return _render_single(view, scale, params, fmt, label), MIMETYPES[fmt], f"patent_drawing.{fmt}"

    # Seven standard views: always orthographic, one shared scale.
    import copy
    p = copy.copy(params)
    p.projection = "orthographic"
    angles = _standard_angles(p)
    extents = [view_extents(prep, az, el, p) for _, az, el in angles]

    if fmt == "pdf":
        per_sheet = 1 if int(p.views_per_sheet) == 1 else 2
        uw = SHEET_W - MARGIN_L - MARGIN_R
        uh = (SHEET_H - MARGIN_T - MARGIN_B) / per_sheet - LABEL_SPACE
        scale = _shared_scale(extents, uw, uh)
        views = [compute_view(prep, n, az, el, p, scale) for n, az, el in angles]
        return _render_sheets_pdf(views, scale, p, per_sheet), MIMETYPES["pdf"], "patent_drawings.pdf"

    if fmt == "svg":
        p.fit = "sphere"
        scale = _single_scale(prep, extents[0], p)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, (n, az, el) in enumerate(angles, 1):
                v = compute_view(prep, n, az, el, p, scale)
                label = f"FIG. {i}" if p.show_figure_label else None
                z.writestr(f"FIG_{i}_{n.replace(' ', '_')}.svg", _render_single(v, scale, p, "svg", label))
        return zbuf.getvalue(), MIMETYPES["zip"], "patent_drawings_svg.zip"

    scale = _shared_scale(extents, 3.1, 3.4 - LABEL_SPACE - 0.3)
    views = [compute_view(prep, n, az, el, p, scale) for n, az, el in angles]
    return _render_contact_png(views, p), MIMETYPES["png"], "patent_drawings_preview.png"


def generate_patent_drawing(stl_bytes: bytes, params: DrawingParams) -> bytes:
    """Backwards-compatible entry point: returns just the output bytes."""
    return generate(stl_bytes, params)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Command line:  python patent_drawing.py part.stl --views standard --format pdf
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description="STL -> patent-style line drawings")
    ap.add_argument("stl")
    ap.add_argument("-o", "--output")
    ap.add_argument("--views", choices=["single", "standard"], default="standard")
    ap.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf")
    ap.add_argument("--azimuth", type=float, default=45.0)
    ap.add_argument("--elevation", type=float, default=30.0)
    ap.add_argument("--up", choices=["z", "y"], default="z")
    ap.add_argument("--rotate", type=float, default=0.0, help="spin model about the up axis (deg)")
    ap.add_argument("--crease", type=float, default=25.0)
    ap.add_argument("--shading", choices=["none", "hatch", "stipple"], default="none")
    ap.add_argument("--resolution", type=int, default=2400)
    a = ap.parse_args(argv)

    params = DrawingParams(view_mode=a.views, output_format=a.format, view_azimuth=a.azimuth,
                           view_elevation=a.elevation, up_axis=a.up, model_rotation_deg=a.rotate,
                           crease_angle_deg=a.crease, shading=a.shading, hlr_resolution=a.resolution)
    t0 = time.time()
    with open(a.stl, "rb") as f:
        data, _, name = generate(f.read(), params)
    out = a.output or name
    with open(out, "wb") as f:
        f.write(data)
    print(f"Wrote {out} ({len(data) / 1024:.0f} KB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
