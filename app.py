"""
app.py — Flask web server for the Patent Drawing Converter.
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import io

load_dotenv()

from patent_drawing import generate_patent_drawing, DrawingParams

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", 50)) * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Accepts multipart/form-data:
      - stl_file: the STL file
      - params: JSON string of DrawingParams fields (all optional)

    Returns the generated PNG as image/png.
    """
    if "stl_file" not in request.files:
        return jsonify({"error": "No STL file provided. Send file as 'stl_file'."}), 400

    stl_file = request.files["stl_file"]
    filename = stl_file.filename or ""
    if not filename.lower().endswith(".stl"):
        return jsonify({"error": "File must be an STL file (.stl extension required)."}), 400

    stl_bytes = stl_file.read()
    if len(stl_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    # Parse optional params JSON
    params_dict = {}
    if "params" in request.form:
        try:
            params_dict = json.loads(request.form["params"])
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid params JSON: {e}"}), 400

    # Coerce numeric types (HTML form values arrive as strings)
    float_fields = {
        "view_azimuth", "view_elevation", "light_azimuth", "light_elevation",
        "hatch_angle", "hatch_spacing_dense", "hatch_spacing_sparse",
        "hatch_threshold", "line_width_outline", "line_width_detail",
        "crease_angle_deg", "fov_deg", "image_width_inches", "image_height_inches",
    }
    int_fields = {"output_dpi"}
    bool_fields = {"show_figure_label", "background_white"}

    for k, v in params_dict.items():
        if k in float_fields:
            try:
                params_dict[k] = float(v)
            except (ValueError, TypeError):
                pass
        elif k in int_fields:
            try:
                params_dict[k] = int(v)
            except (ValueError, TypeError):
                pass
        elif k in bool_fields:
            if isinstance(v, str):
                params_dict[k] = v.lower() in ("true", "1", "yes")

    params = DrawingParams.from_dict(params_dict)

    try:
        png_bytes = generate_patent_drawing(stl_bytes, params)
    except Exception as e:
        app.logger.exception("Error generating patent drawing")
        return jsonify({"error": f"Failed to generate drawing: {str(e)}"}), 500

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="patent_drawing.png",
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    print(f"Starting Patent Drawing Converter on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
