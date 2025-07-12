from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import pandas as pd

pipeline = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)


def get_table(source_path):
    # load image once to get page width
    img = Image.open(source_path)
    page_w, page_h = img.size

    t1 = 0.675 * page_w  # 67.5%
    t2 = 0.746 * page_w  # 74.6%

    outputs = pipeline.predict(source_path)
    assert len(outputs) == 1

    for i, res in enumerate(outputs):
        res.print()

        # pull out the raw OCR JSON
        j = res.json["res"]
        texts = j["rec_texts"]
        boxes = np.array(j["rec_boxes"])  # shape (n,4): [x_min,y_min,x_max,y_max]

        # compute centers & median height for row-clustering
        x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
        y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
        heights = boxes[:, 3] - boxes[:, 1]
        med_h = np.median(heights)

        # 1) sort all tokens by y-center
        idx_sorted = np.argsort(y_centers)

        # 2) cluster into rows whenever gap > half median height
        rows = []
        current = [idx_sorted[0]]
        for idx in idx_sorted[1:]:
            if abs(y_centers[idx] - y_centers[current[-1]]) < (med_h * 0.5):
                current.append(idx)
            else:
                rows.append(current)
                current = [idx]
        rows.append(current)

        # 3) for each row, bucket into your 4 columns
        table = []
        for row in rows:
            # init your four “strips”:
            #   Col1 is always None (“Left Head”), the other three start empty
            cells = {"Left Head": None, "Left": [], "Middle": [], "Right": []}
            # sort tokens in this row by x-center so pipes preserve L→R
            for k in sorted(row, key=lambda k: x_centers[k]):
                xc = x_centers[k]
                txt = texts[k]
                if xc < t1:
                    cells["Left"].append(txt)
                elif xc < t2:
                    cells["Middle"].append(txt)
                else:
                    cells["Right"].append(txt)

            # join multi-token lists with "|" and leave empty lists as ""
            for col in ("Left", "Middle", "Right"):
                cells[col] = "|".join(cells[col]) if cells[col] else ""
            table.append(cells)

        # 4) make a DataFrame with exactly those four columns
        df = pd.DataFrame(table, columns=["Left Head", "Left", "Middle", "Right"])
        return df
