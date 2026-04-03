"""
patent_drawing.py — Convert STL files to USPTO-style patent drawings.

Pipeline:
  1. Load STL → trimesh
  2. Rotate mesh to camera viewpoint
  3. Project to 2D (perspective or isometric)
  4. Backface-cull + painter's algorithm for visible face ordering
  5. Detect silhouette edges and crease edges
  6. Compute cross-hatch lines per face (density = shading from light direction)
  7. Render with matplotlib → PNG bytes
"""

import io
import math
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass-style config
# ─────────────────────────────────────────────────────────────────────────────

class DrawingParams:
    """All user-adjustable parameters for the patent drawing."""

    def __init__(
        self,
        view_azimuth: float = 45.0,       # degrees — horizontal camera angle
        view_elevation: float = 30.0,      # degrees — vertical camera angle
        light_azimuth: float = 135.0,      # degrees — light source horizontal
        light_elevation: float = 45.0,     # degrees — light source vertical
        hatch_angle: float = 45.0,         # degrees — slope of hatch lines
        hatch_spacing_dense: float = 3.0,  # pt — spacing at max darkness
        hatch_spacing_sparse: float = 14.0,# pt — spacing at min darkness (threshold)
        hatch_threshold: float = 0.15,     # faces brighter than this get no hatch
        line_width_outline: float = 2.0,   # pt — silhouette edge weight
        line_width_detail: float = 0.6,    # pt — interior crease edge weight
        crease_angle_deg: float = 30.0,    # degrees — min angle to be a crease edge
        projection: str = "perspective",   # "perspective" or "isometric"
        fov_deg: float = 35.0,             # degrees — field of view (perspective only)
        output_dpi: int = 300,             # DPI of output PNG
        image_width_inches: float = 6.0,   # output image width
        image_height_inches: float = 6.0,  # output image height
        show_figure_label: bool = True,    # add "FIG. 1" label
        figure_label: str = "FIG. 1",
        background_white: bool = True,
    ):
        self.view_azimuth = view_azimuth
        self.view_elevation = view_elevation
        self.light_azimuth = light_azimuth
        self.light_elevation = light_elevation
        self.hatch_angle = hatch_angle
        self.hatch_spacing_dense = hatch_spacing_dense
        self.hatch_spacing_sparse = hatch_spacing_sparse
        self.hatch_threshold = hatch_threshold
        self.line_width_outline = line_width_outline
        self.line_width_detail = line_width_detail
        self.crease_angle_deg = crease_angle_deg
        self.projection = projection
        self.fov_deg = fov_deg
        self.output_dpi = output_dpi
        self.image_width_inches = image_width_inches
        self.image_height_inches = image_height_inches
        self.show_figure_label = show_figure_label
        self.figure_label = figure_label
        self.background_white = background_white

    @classmethod
    def from_dict(cls, d: dict) -> "DrawingParams":
        p = cls()
        for key, val in d.items():
            if hasattr(p, key):
                setattr(p, key, val)
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spherical_to_unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert azimuth/elevation in degrees to a unit vector."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return np.array([x, y, z])


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix that brings the mesh into camera space.
    Camera looks down the -Z axis after rotation.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Rotate around Z by azimuth, then around X by elevation
    Rz = np.array([
        [ math.cos(az), math.sin(az), 0],
        [-math.sin(az), math.cos(az), 0],
        [0,             0,            1],
    ])
    Rx = np.array([
        [1, 0,            0           ],
        [0, math.cos(el), math.sin(el)],
        [0,-math.sin(el), math.cos(el)],
    ])
    return Rx @ Rz


def _project_vertices(verts_cam: np.ndarray, params: DrawingParams) -> np.ndarray:
    """
    Project 3D camera-space vertices to 2D screen coordinates.
    Returns (N, 2) array in normalised coordinates (centred, scale ~1).
    """
    if params.projection == "isometric":
        return verts_cam[:, :2]

    # Perspective divide
    fov = math.radians(params.fov_deg)
    f = 1.0 / math.tan(fov / 2.0)
    # Use z depth (negative = in front of camera after rotation)
    z = verts_cam[:, 2]
    # Shift so nothing is at z=0
    z_shifted = z - z.min() + 1.0
    x2d = f * verts_cam[:, 0] / z_shifted
    y2d = f * verts_cam[:, 1] / z_shifted
    return np.stack([x2d, y2d], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

class PatentDrawingGenerator:

    def generate(self, stl_bytes: bytes, params: DrawingParams) -> bytes:
        """
        Full pipeline: STL bytes → PNG bytes.
        """
        mesh = self._load_stl(stl_bytes)
        verts_cam, faces, normals_cam, face_depths = self._transform_mesh(mesh, params)
        verts_2d = self._project_and_normalise(verts_cam, params)
        visible_mask = self._backface_cull(normals_cam)
        sorted_indices = self._painters_sort(face_depths, visible_mask)
        silhouette_edges, crease_edges = self._detect_edges(
            mesh, verts_2d, normals_cam, visible_mask, params
        )
        light_dir = _spherical_to_unit(params.light_azimuth, params.light_elevation)
        hatch_segments = self._build_hatch(
            faces, verts_2d, normals_cam, visible_mask, sorted_indices, light_dir, params
        )
        png_bytes = self._render(
            verts_2d, silhouette_edges, crease_edges, hatch_segments, params
        )
        return png_bytes

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_stl(self, stl_bytes: bytes) -> trimesh.Trimesh:
        mesh = trimesh.load(
            io.BytesIO(stl_bytes),
            file_type="stl",
            force="mesh",
        )
        if not isinstance(mesh, trimesh.Trimesh):
            # Scene — merge all geometry
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        # Centre and normalise scale to unit sphere
        mesh.apply_translation(-mesh.centroid)
        scale = np.linalg.norm(mesh.vertices, axis=1).max()
        if scale > 0:
            mesh.apply_scale(1.0 / scale)
        return mesh

    # ── Transform ─────────────────────────────────────────────────────────────

    def _transform_mesh(self, mesh: trimesh.Trimesh, params: DrawingParams):
        R = _rotation_matrix(params.view_azimuth, params.view_elevation)
        verts_cam = (R @ mesh.vertices.T).T  # (N, 3)
        faces = mesh.faces                    # (F, 3) vertex indices
        # Face normals in camera space
        normals_world = mesh.face_normals     # (F, 3)
        normals_cam = (R @ normals_world.T).T # (F, 3)
        # Depth = mean Z of face vertices in camera space
        face_depths = verts_cam[faces].mean(axis=1)[:, 2]
        return verts_cam, faces, normals_cam, face_depths

    # ── Project ───────────────────────────────────────────────────────────────

    def _project_and_normalise(self, verts_cam: np.ndarray, params: DrawingParams) -> np.ndarray:
        verts_2d = _project_vertices(verts_cam, params)
        # Normalise to [-1, 1] with equal aspect
        centre = (verts_2d.max(axis=0) + verts_2d.min(axis=0)) / 2
        verts_2d -= centre
        span = (verts_2d.max(axis=0) - verts_2d.min(axis=0)).max()
        if span > 0:
            verts_2d /= (span / 1.8)  # 1.8 gives 10% margin each side
        return verts_2d

    # ── Backface cull ─────────────────────────────────────────────────────────

    def _backface_cull(self, normals_cam: np.ndarray) -> np.ndarray:
        """Return boolean mask — True for faces facing toward camera (z < 0 points away)."""
        # Camera looks down -Z in our convention; face visible if normal_z > 0
        return normals_cam[:, 2] > 0

    # ── Painter's sort ────────────────────────────────────────────────────────

    def _painters_sort(self, face_depths: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        """Return face indices sorted back-to-front (for painter's algorithm)."""
        visible_indices = np.where(visible_mask)[0]
        order = np.argsort(-face_depths[visible_indices])  # descending depth (far first)
        return visible_indices[order]

    # ── Edge detection ────────────────────────────────────────────────────────

    def _detect_edges(
        self,
        mesh: trimesh.Trimesh,
        verts_2d: np.ndarray,
        normals_cam: np.ndarray,
        visible_mask: np.ndarray,
        params: DrawingParams,
    ):
        """
        Returns:
          silhouette_edges: list of (p1, p2) tuples of 2D points
          crease_edges: list of (p1, p2) tuples of 2D points
        """
        faces = mesh.faces
        edges_face = mesh.edges_sorted           # (E, 2) — each edge as sorted vertex pair
        edges_face_idx = mesh.edges_sorted_tree   # not used; use face_adjacency instead

        # Build edge → face adjacency manually
        edge_to_faces: dict[tuple, list] = {}
        for fi, face in enumerate(faces):
            for i in range(3):
                e = tuple(sorted([face[i], face[(i+1) % 3]]))
                edge_to_faces.setdefault(e, []).append(fi)

        silhouette = []
        crease = []
        cos_crease = math.cos(math.radians(params.crease_angle_deg))

        for edge, flist in edge_to_faces.items():
            v0, v1 = verts_2d[edge[0]], verts_2d[edge[1]]
            seg = (v0, v1)

            if len(flist) == 1:
                # Boundary edge — always a silhouette if visible
                fi = flist[0]
                if visible_mask[fi]:
                    silhouette.append(seg)
                continue

            fi0, fi1 = flist[0], flist[1]
            vis0 = visible_mask[fi0]
            vis1 = visible_mask[fi1]

            if vis0 != vis1:
                # One front-facing, one back-facing → silhouette
                silhouette.append(seg)
            elif vis0 and vis1:
                # Both visible — check if it's a crease
                dot = np.dot(normals_cam[fi0], normals_cam[fi1])
                dot = np.clip(dot, -1.0, 1.0)
                if dot < cos_crease:
                    crease.append(seg)

        return silhouette, crease

    # ── Hatching ──────────────────────────────────────────────────────────────

    def _build_hatch(
        self,
        faces: np.ndarray,
        verts_2d: np.ndarray,
        normals_cam: np.ndarray,
        visible_mask: np.ndarray,
        sorted_indices: np.ndarray,
        light_dir: np.ndarray,
        params: DrawingParams,
    ) -> list:
        """
        For each visible face (back-to-front), compute cross-hatch line segments
        clipped to the face polygon. Returns flat list of (p1, p2) segments.
        """
        angle_rad = math.radians(params.hatch_angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        # Perpendicular direction (spacing direction)
        px = -dy
        py = dx

        all_segments = []

        for fi in sorted_indices:
            tri = verts_2d[faces[fi]]  # (3, 2)

            # Light intensity: dot product of face normal with light direction
            # We use the world-space normal but it's already in cam space here.
            # For shading we want the angle to the light in world space.
            # normals_cam is already in camera space; light_dir is also camera space here
            # (we rotate light the same way as the mesh)
            intensity = float(np.dot(normals_cam[fi], light_dir))
            intensity = max(0.0, intensity)  # clamp to [0, 1]

            if intensity > (1.0 - params.hatch_threshold):
                # Bright face — skip hatching
                continue

            # Map darkness (1-intensity) to line spacing
            darkness = 1.0 - intensity
            spacing = params.hatch_spacing_sparse - darkness * (
                params.hatch_spacing_sparse - params.hatch_spacing_dense
            )
            spacing = max(params.hatch_spacing_dense, spacing)

            # Build shapely polygon for this face
            try:
                poly = Polygon(tri)
                if not poly.is_valid or poly.area < 1e-8:
                    continue
                poly = poly.buffer(0)  # fix any self-intersections
            except Exception:
                continue

            # Bounding box in the perpendicular direction
            dots = tri @ np.array([px, py])
            t_min, t_max = dots.min(), dots.max()

            # Extend slightly past bbox
            t = t_min - spacing
            segments = []
            while t <= t_max + spacing:
                # Line through origin along (dx, dy) offset by t*(px, py)
                # Parameterise: point = t*(px,py) + s*(dx,dy)
                # We sweep s from -3 to +3 (well outside unit space)
                s_range = 3.0
                p_start = np.array([t * px + (-s_range) * dx, t * py + (-s_range) * dy])
                p_end   = np.array([t * px +   s_range  * dx, t * py +   s_range  * dy])
                try:
                    line = LineString([p_start, p_end])
                    clipped = poly.intersection(line)
                    if clipped.is_empty:
                        t += spacing
                        continue
                    # Extract segments from result
                    if isinstance(clipped, LineString):
                        coords = list(clipped.coords)
                        for i in range(len(coords) - 1):
                            segments.append((np.array(coords[i]), np.array(coords[i+1])))
                    elif isinstance(clipped, MultiLineString):
                        for part in clipped.geoms:
                            coords = list(part.coords)
                            for i in range(len(coords) - 1):
                                segments.append((np.array(coords[i]), np.array(coords[i+1])))
                except Exception:
                    pass
                t += spacing

            all_segments.extend(segments)

        return all_segments

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(
        self,
        verts_2d: np.ndarray,
        silhouette_edges: list,
        crease_edges: list,
        hatch_segments: list,
        params: DrawingParams,
    ) -> bytes:
        fig, ax = plt.subplots(
            figsize=(params.image_width_inches, params.image_height_inches),
            dpi=params.output_dpi,
        )
        bg = "white" if params.background_white else "black"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_aspect("equal")
        ax.axis("off")

        # Compute axis limits with margin
        if len(verts_2d) > 0:
            margin = 0.15
            xmin, xmax = verts_2d[:, 0].min(), verts_2d[:, 0].max()
            ymin, ymax = verts_2d[:, 1].min(), verts_2d[:, 1].max()
            span = max(xmax - xmin, ymax - ymin)
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            ax.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
            ax.set_ylim(cy - span/2 - margin, cy + span/2 + margin)

        ink = "black" if params.background_white else "white"

        # Draw hatch lines
        if hatch_segments:
            hatch_lc = LineCollection(
                hatch_segments,
                linewidths=0.4,
                colors=ink,
                zorder=1,
            )
            ax.add_collection(hatch_lc)

        # Draw crease (interior detail) edges
        if crease_edges:
            crease_lc = LineCollection(
                crease_edges,
                linewidths=params.line_width_detail,
                colors=ink,
                zorder=2,
            )
            ax.add_collection(crease_lc)

        # Draw silhouette (outline) edges — drawn last/thickest
        if silhouette_edges:
            sil_lc = LineCollection(
                silhouette_edges,
                linewidths=params.line_width_outline,
                colors=ink,
                zorder=3,
            )
            ax.add_collection(sil_lc)

        # Figure label (USPTO style)
        if params.show_figure_label and params.figure_label:
            ax.text(
                0.5, 0.02,
                params.figure_label,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=10,
                fontfamily="serif",
                color=ink,
            )

        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=params.output_dpi, bbox_inches="tight",
                    facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def generate_patent_drawing(stl_bytes: bytes, params: DrawingParams) -> bytes:
    """Top-level entry point. Returns PNG bytes."""
    return PatentDrawingGenerator().generate(stl_bytes, params)
