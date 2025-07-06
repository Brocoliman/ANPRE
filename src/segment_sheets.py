# ◇━━━━━━━━━━━━  BORDER-DETECTION with ADAPTIVE DARK-PIXEL MASK  ━━━━━━━━━━━◇
# Loads one PDF page, detects the page frame via inward probes, and shows:
#   • probe hits  (blue)     • fitted border lines (green)
#   • RANSAC corner pts (red)• warped, rectified page
#   • binary mask of “dark / black” pixels (white)
#
# If any edge gets < 10 probe hits, thresholds are relaxed once (+20 on V & S).

import cv2
import fitz
import numpy as np
from pathlib import Path
import os
import time
import json
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="case_outputs/sheets",
        help="Path to folder with patient sheets",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/seg",
        help="Path to folder with sheet segments",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/seg",
        help="Path to folder for debugging segmentation method (not used, currently)",
    )
    parser.add_argument(
        "-template",
        "-t",
        type=str,
        default="templates/spring2025.json",
        help="Path to sheet template file",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        default=150,
        help="Loading and processing DPI of PDF for segmentation",
    )
    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


# ═════════════════════  helper: adaptive probe homography  ════════════════
def find_border_homography(
    page,
    dpi=150,
    max_search_frac=0.15,
    probes=100,
    min_probes=10,  # otherwise, SV adjust
    min_inliers=10,  # otherwise, also include outliers
    v_thresh_init=200,
    s_thresh_init=128,
    tolerance=15,
    ncheckwin=((-1, 2), (0, 1)),  # inclusive-exclusive
    dilationwin=(3, 3),
    y_band_px=10,
    min_cluster_pts=10,
):
    """
    Returns:
      Hmat (3×3 homography),
      (w_out, h_out) target canvas,
      border_pts dict, lines dict, corners tuple, border_mask (bool array)
    """
    (rmin, rmax), (cmin, cmax) = ncheckwin

    # ── render RGB and convert to HSV ───────────────────────────────
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    Hpx, Wpx = hsv.shape[:2]

    # ── collect probe points along dark pixel edges and return endpoints ─────────────────
    def collect_edges(v_thr, s_thr):
        mask = (hsv[..., 2] < v_thr) & (hsv[..., 1] < s_thr)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones(dilationwin, np.uint8)) > 0

        # Store march endpoint coordinates
        bp = {"L": [], "R": [], "T": [], "B": []}

        # march to limit, with given directional step vectors
        def march(start, step, limit):
            r, c = start
            for _ in range(limit):
                if np.any(mask[r + rmin : r + rmax, c + cmin : c + cmax]):
                    return c if step[1] else r
                r += step[0]
                c += step[1]
            return None

        # march start points
        rows = np.linspace(int(0.05 * Hpx), int(0.95 * Hpx), probes, dtype=int)
        cols = np.linspace(int(0.05 * Wpx), int(0.95 * Wpx), probes, dtype=int)

        # pixel borders for 15%
        max_dx = int(max_search_frac * Wpx)
        max_dy = int(max_search_frac * Hpx)

        # utilize direction step vectors
        for r in rows:
            xL = march((r, 0), (0, 1), max_dx)
            if xL is not None:
                bp["L"].append([xL, r])
            xR = march((r, Wpx - 1), (0, -1), max_dx)
            if xR is not None:
                bp["R"].append([xR, r])
        for c in cols:
            yT = march((0, c), (1, 0), max_dy)
            if yT is not None:
                bp["T"].append([c, yT])
            yB = march((Hpx - 1, c), (-1, 0), max_dy)
            if yB is not None:
                bp["B"].append([c, yB])
        return bp, mask

    border_pts, border_mask = collect_edges(v_thresh_init, s_thresh_init)

    # Try again if less than certain amount of probe points (min_probes) reliable for lsrl
    if any(len(pts) < min_probes for pts in border_pts.values()):
        border_pts, border_mask = collect_edges(v_thresh_init + 20, s_thresh_init + 20)
        if any(len(pts) < min_probes for pts in border_pts.values()):
            raise RuntimeError("Border detection failed: insufficient probe hits.")

    # ── Outlier bucket to avoid dict-resize error ───────────────────
    border_pts["OUT"] = []

    # ── Top-cluster extraction for "B" (bottom) only ────────────────
    def extract_top_cluster(points):
        buckets = defaultdict(list)
        for x, y in points:
            band = int(y // y_band_px)
            buckets[band].append((x, y))
        for band in sorted(buckets.keys()):
            if (
                len(buckets[band]) >= min_cluster_pts
            ):  # min amount of points to count as significant cluster for lsrl
                return np.array(buckets[band], np.float32)
        return np.array(points, np.float32)

    # ── Bottom-cluster extraction for "T" (bottom) only ────────────────
    def extract_bottom_cluster(points):
        buckets = defaultdict(list)
        for x, y in points:
            band = int(y // y_band_px)
            buckets[band].append((x, y))
        for band in sorted(buckets.keys(), reverse=True):
            if (
                len(buckets[band]) >= min_cluster_pts
            ):  # min amount of points to count as significant cluster for lsrl
                return np.array(buckets[band], np.float32)
        return np.array(points, np.float32)

    # ── Line fitting and filtering per edge ─────────────────────────
    lines = {}
    for side in ["T", "B", "L", "R"]:

        ## Load points
        pts = border_pts[side]  # list of coords

        # Bottom side requires special processing due to existence of footers
        if side == "B":
            # Add potential outliers to allow fallback
            pts += border_pts["OUT"]
            pts = extract_top_cluster(pts)

        # Top side requires special processing due to existence of header notes
        if side == "T":
            # Add potential outliers to allow fallback
            pts += border_pts["OUT"]
            pts = extract_bottom_cluster(pts)

        pts = np.asarray(pts, np.float32)

        if pts.shape[0] < 2:
            raise RuntimeError(f"Too few points on edge {side}")

        ## Capture line equation
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # Distance calculation for outliers
        A, B = vy, -vx  # norms
        C = -(A * x0 + B * y0)  # intercept Ax+By+C=0
        dist = np.abs(A * pts[:, 0] + B * pts[:, 1] + C) / np.hypot(A, B)
        inliers = pts[dist < tolerance]
        outliers = pts[dist >= tolerance]
        border_pts["OUT"].extend(outliers.tolist())  # add outliers

        # Fit only inliers if sufficient quantity of inliers
        if len(inliers) >= min_inliers:
            vx, vy, x0, y0 = cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01)
            A, B = vy, -vx
            C = -(A * x0 + B * y0)
            border_pts[side] = inliers.tolist()

        lines[side] = (A, B, C)

    # ── Intersect the lines ─────────────────────────
    def intersect(L1, L2):
        A1, B1, C1 = L1
        A2, B2, C2 = L2
        D = A1 * B2 - A2 * B1
        return [(B1 * C2 - B2 * C1) / D, (C1 * A2 - C2 * A1) / D]

    tl = intersect(lines["T"], lines["L"])
    tr = intersect(lines["T"], lines["R"])
    br = intersect(lines["B"], lines["R"])
    bl = intersect(lines["B"], lines["L"])

    # Intersections
    dst = np.array([tl, tr, br, bl], np.float32)  # location of intersections
    w_out = int(np.linalg.norm(np.subtract(tr, tl)))
    h_out = int(np.linalg.norm(np.subtract(bl, tl)))
    src = np.array(
        [[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]], np.float32
    )  # normalized

    Hmat, _ = cv2.findHomography(dst, src, 0)
    return Hmat, (w_out, h_out), border_pts, lines, (tl, tr, br, bl), border_mask


# ═════════════════════  segment all sheets  ════════════════


def segment(
    source_dir: str,
    output_dir: str,
    debug_dir: str,
    template: str,
    dpi: int,
):
    # ---------- load template ----------------------------------------
    with open(template) as f:
        format = json.load(f)

    segments = format["sections"]

    # ═══════════ processing ════════════════════════════════════
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    filenames = source_dir.iterdir()

    for filename in filenames:
        if not filename.is_file():
            print(f"[WARN] Missing file: {filename}")
            continue
        if not filename.suffix == ".pdf":
            continue

        # ═══════════ align all sheets ════════════════════════════════════
        aligned_pages = []
        doc = fitz.open(filename)

        # ---------- load page ----------------------------------------
        for page_no, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            img_rgb = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # ---------- compute homography, warp, and draw diagnostics  ---------

            t0 = time.perf_counter()
            Hmat, (W_out, H_out), border_pts, lines, corners, border_mask = (
                find_border_homography(page, dpi=dpi)
            )
            homography_time = time.perf_counter() - t0
            print(f"[REPORT] Homography time: {homography_time}")

            pix_aligned = cv2.warpPerspective(img_bgr, Hmat, (W_out, H_out))
            aligned_pages.append(pix_aligned)

        # ═══════════ segment all sheets ════════════════════════════════════
        for seg_title, seg_coords in segments.items():
            page_no, p0, q0, p1, q1 = seg_coords
            pix_aligned = aligned_pages[page_no]
            h, w, _ = pix_aligned.shape
            x0, y0, x1, y1 = int(w * p0), int(h * q0), int(w * p1), int(h * q1)
            cv2.imwrite(
                os.path.join(output_dir, f"{seg_title}_{filename.stem}.png"),
                pix_aligned[y0 : y1 + 1, x0 : x1 + 1],
            )


# ═════════════════════  Main  ════════════════

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    template_pth = args.template
    dpi = args.dpi

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    segment(source_dir, output_dir, debug_dir, template_pth, dpi)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
