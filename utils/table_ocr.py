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
    # exactly the same clustering logic you had in get_table,
    # but operating on the `outputs` argument rather than
    # calling pipeline.predict() again.

    img = Image.open(source_path)
    page_w, _ = img.size
    t1, t2 = 0.675 * page_w, 0.746 * page_w

    res = outputs[0]
    j = res.json["res"]
    texts = j["rec_texts"]
    boxes = np.array(j["rec_boxes"])
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    med_h = np.median(boxes[:, 3] - boxes[:, 1])

    # cluster into rows
    idx_sorted = np.argsort(y_centers)
    rows, cur = [], [idx_sorted[0]]
    for idx in idx_sorted[1:]:
        if abs(y_centers[idx] - y_centers[cur[-1]]) < med_h * 0.5:
            cur.append(idx)
        else:
            rows.append(cur)
            cur = [idx]
    rows.append(cur)

    # build table
    table = []
    for row in rows:
        cells = {"head": None, "test_info": [], "raw_score": [], "comments": []}
        for k in sorted(row, key=lambda k: x_centers[k]):
            txt = texts[k]
            xc = x_centers[k]
            if xc < t1:
                cells["test_info"].append(txt)
            elif xc < t2:
                cells["raw_score"].append(txt)
            else:
                cells["comments"].append(txt)
        for col in ("test_info", "raw_score", "comments"):
            cells[col] = "|".join(cells[col]) if cells[col] else ""
        table.append(cells)

    return pd.DataFrame(table, columns=["head", "test_info", "raw_score", "comments"])
