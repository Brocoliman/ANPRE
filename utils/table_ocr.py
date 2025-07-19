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
            "test_info_conf": [],
            "raw_score": [],
            "raw_score_conf": [],
            "comments": [],
            "comments_conf": [],
        }

        for k in sorted(row, key=lambda k: x_centers[k]):
            txt, conf, xc = texts[k], scores[k], x_centers[k]
            conf = round(conf, 2)

            if xc < t1:  # test description column
                cells["test_info"].append(txt)
                cells["test_info_conf"].append(conf)
            elif xc < t2:  # numeric score column
                cells["raw_score"].append(txt)
                cells["raw_score_conf"].append(conf)
            else:  # qualitative comments column
                cells["comments"].append(txt)
                cells["comments_conf"].append(conf)

        # join the pieces collected in each column
        for col in ("test_info", "raw_score", "comments"):
            cells[col] = "|".join(cells[col]) if cells[col] else ""
        for col in ("test_info_conf", "raw_score_conf", "comments_conf"):
            if cells[col]:
                # choose your favorite aggregation: "|".join, np.mean, np.min …
                cells[col] = np.mean(cells[col])  # float – easier to sort/filter later
            else:
                cells[col] = np.nan

        table.append(cells)

    return pd.DataFrame(
        table,
        columns=[
            "head",
            "test_info",
            "raw_score",
            "comments",
            "test_info_conf",
            "raw_score_conf",
            "comments_conf",
        ],
    )
