"""
app.py — Flask web server for the Patent Drawing Converter.
"""

import hashlib
import io
import json
import os
from collections import OrderedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

load_dotenv()

from patent_drawing import DrawingParams, generate, prepare_mesh

app = Flask(__name__)
CORS(app, expose_headers=["X-Filename"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", 300)) * 1024 * 1024

FLOAT_FIELDS = {
    "view_azimuth", "view_elevation", "light_azimuth", "light_elevation",
    "hatch_angle", "hatch_spacing", "line_width_outline", "line_width_detail",
    "crease_angle_deg", "fov_deg", "image_width_inches", "image_height_inches",
    "model_rotation_deg", "min_line_px",
}
INT_FIELDS = {"output_dpi", "hlr_resolution", "views_per_sheet"}
BOOL_FIELDS = {"show_figure_label"}
CHOICE_FIELDS = {
    "view_mode": {"single", "standard"},
    "output_format": {"png", "svg", "pdf"},
    "projection": {"orthographic", "perspective", "isometric"},
    "up_axis": {"z", "y"},
    "shading": {"none", "hatch", "stipple"},
    "fit": {"sphere", "tight"},
}
LIMITS = {
    "output_dpi": (50, 600), "hlr_resolution": (400, 6000),
    "image_width_inches": (1, 24), "image_height_inches": (1, 24),
    "crease_angle_deg": (1, 179), "hatch_spacing": (1, 60),
    "line_width_outline": (0.05, 10), "line_width_detail": (0.05, 10),
}

# Loading a big STL is the slow part, so keep the last few prepared meshes.
# The preview and each download then reuse the same mesh.
_MESH_CACHE: "OrderedDict[str, object]" = OrderedDict()
_MESH_CACHE_SIZE = int(os.getenv("MESH_CACHE_SIZE", 2))


def _prepared(stl_bytes: bytes, params: DrawingParams):
    key = hashlib.sha1(stl_bytes).hexdigest() + f"|{params.up_axis}|{params.model_rotation_deg}"
    prep = _MESH_CACHE.get(key)
    if prep is None:
        prep = prepare_mesh(stl_bytes, params)
        _MESH_CACHE[key] = prep
        while len(_MESH_CACHE) > _MESH_CACHE_SIZE:
            _MESH_CACHE.popitem(last=False)
    else:
        _MESH_CACHE.move_to_end(key)
    return prep


def _clean_params(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        try:
            if k in FLOAT_FIELDS:
                v = float(v)
            elif k in INT_FIELDS:
                v = int(float(v))
            elif k in BOOL_FIELDS:
                v = v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
            elif k in CHOICE_FIELDS:
                v = str(v).lower()
                if v not in CHOICE_FIELDS[k]:
                    continue
            elif k == "figure_label":
                v = str(v)[:40]
            else:
                continue
        except (ValueError, TypeError):
            continue
        if k in LIMITS:
            lo, hi = LIMITS[k]
            v = min(max(v, lo), hi)
        out[k] = v
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate_route():
    """
    multipart/form-data:
      - stl_file: the STL file
      - params:   JSON string of DrawingParams fields (all optional), including
                  view_mode ("single" | "standard") and output_format ("png" | "svg" | "pdf")

    Returns the drawing. Standard view set: png = preview sheet, pdf = USPTO-size
    sheets, svg = zip with one SVG per figure.
    """
    if "stl_file" not in request.files:
        return jsonify({"error": "No STL file provided. Send file as 'stl_file'."}), 400

    stl_file = request.files["stl_file"]
    if not (stl_file.filename or "").lower().endswith(".stl"):
        return jsonify({"error": "File must be an STL file (.stl extension required)."}), 400

    stl_bytes = stl_file.read()
    if len(stl_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    params_dict = {}
    if "params" in request.form:
        try:
            params_dict = json.loads(request.form["params"])
            if not isinstance(params_dict, dict):
                raise ValueError("params must be a JSON object")
        except ValueError as e:
            return jsonify({"error": f"Invalid params JSON: {e}"}), 400

    params = DrawingParams.from_dict(_clean_params(params_dict))

    try:
        data, mimetype, filename = generate(stl_bytes, params, prep=_prepared(stl_bytes, params))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Error generating patent drawing")
        return jsonify({"error": f"Failed to generate drawing: {e}"}), 500

    resp = send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=False, download_name=filename)
    resp.headers["X-Filename"] = filename
    return resp


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    print(f"Starting Patent Drawing Converter on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
