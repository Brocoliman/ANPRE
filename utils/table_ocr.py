from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import pandas as pd

pipeline = None


def initialize_pipeline(gpu):
    global pipeline
    pipeline = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu" if gpu else "cpu",
    )


def ocr(source_paths, debug_path):
    """
    Function inputs multiple
    Run OCR one image at a time to avoid internal multiprocessing.
    """

    results = []
    print(f"[EVENT] Scanning {source_paths}...")
    results = pipeline.predict(source_paths)
    for res in results:
        res.save_to_img(debug_path)
        res.save_to_json(debug_path)
    return results


def get_table(source_path, outputs):
    # ── unchanged pre-processing ───────────────────────────────────────────────
    img = Image.open(source_path)
    page_w, _ = img.size
    t1, t2 = 0.675 * page_w, 0.746 * page_w

    res = outputs[0]
    j = res.json["res"]
    texts = j["rec_texts"]
    scores = j["rec_scores"]
    boxes = np.array(j["rec_boxes"])
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    med_h = np.median(boxes[:, 3] - boxes[:, 1])

    # ── unchanged row clustering ──────────────────────────────────────────────
    idx_sorted = np.argsort(y_centers)
    rows, cur = [], [idx_sorted[0]]
    for idx in idx_sorted[1:]:
        if abs(y_centers[idx] - y_centers[cur[-1]]) < med_h * 0.5:
            cur.append(idx)
        else:
            rows.append(cur)
            cur = [idx]
    rows.append(cur)

    # ── build table with confidence columns ───────────────────────────────────
    table = []
    for row in rows:
        cells = {
            "head": None,
            "test_info": [],
            "raw_score": [],
            "comments": [],
            "test_info_conf": [],
            "raw_score_conf": [],
            "comments_conf": [],
            "test_info_boxes": [],
            "raw_score_boxes": [],
            "comments_boxes": [],
        }

        for k in sorted(row, key=lambda k: x_centers[k]):
            txt, conf, box, xc = texts[k], round(scores[k], 2), boxes[k], x_centers[k]

            if xc < t1:
                tgt_text, tgt_conf, tgt_boxes = (
                    "test_info",
                    "test_info_conf",
                    "test_info_boxes",
                )
            elif xc < t2:
                tgt_text, tgt_conf, tgt_boxes = (
                    "raw_score",
                    "raw_score_conf",
                    "raw_score_boxes",
                )
            else:
                tgt_text, tgt_conf, tgt_boxes = (
                    "comments",
                    "comments_conf",
                    "comments_boxes",
                )

            cells[tgt_text].append(txt)
            cells[tgt_conf].append(conf)
            cells[tgt_boxes].append(box)  # box is [x1,y1,x2,y2]

        # now join strings / aggregate confs
        for col in ("test_info", "raw_score", "comments"):
            cells[col] = "|".join(cells[col]) if cells[col] else ""
        for col in ("test_info_conf", "raw_score_conf", "comments_conf"):
            vals = cells[col]
            cells[col] = float(np.mean(vals)) if vals else np.nan

        # compute the enclosing box for each column
        for col in ("test_info", "raw_score", "comments"):
            box_list = cells[f"{col}_boxes"]
            if box_list:
                arr = np.array(box_list)
                x1, y1 = arr[:, 0].min(), arr[:, 1].min()
                x2, y2 = arr[:, 2].max(), arr[:, 3].max()
                # depending on your output format preference:
                # OPTION A: four separate scalars
                cells[f"{col}_x1"] = x1
                cells[f"{col}_y1"] = y1
                cells[f"{col}_x2"] = x2
                cells[f"{col}_y2"] = y2

                # OPTION B: packed as one tuple (gets stringified in CSV)
                # cells[f"{col}_bbox"] = (x1, y1, x2, y2)
            else:
                # no tokens → NaN or empty
                for suffix in ("_x1", "_y1", "_x2", "_y2"):
                    cells[f"{col}{suffix}"] = np.nan
                # or cells[f"{col}_bbox"] = ""

        table.append(cells)

    df = pd.DataFrame(table)
    return df
