import os
import argparse
import cv2
import numpy as np
import time
import fitz
import pytesseract
from pathlib import Path
from collections import defaultdict

from pdf_set import PDFSet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )
    parser.add_argument(
        "-dataset",
        "-i",
        type=str,
        default="data/one",
        help="Path to folder with list of full patient case files in PDF",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/sheets",
        help="Path to folder with list of patient summary sheets in PDF",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        default=150,
        help="Loading and processing DPI of PDF for extraction",
    )
    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


##############################
# Extract
##############################
def crop_read(
    page: fitz.Page, crop_px: tuple[int, int, int, int], dpi: int, kw: list[str]
):
    """
    Crop *page* to *crop_px*, up-sample if needed, run Tesseract,
    draw green boxes around *all* recognised words.

    Returns
    -------
    matched : set[str]          # words from kw found in the crop
    annotated_bgr : np.ndarray  # BGR image with rectangles & labels
    """
    x0, y0, x1, y1 = crop_px
    clip = fitz.Rect(x0 * 72 / dpi, y0 * 72 / dpi, x1 * 72 / dpi, y1 * 72 / dpi)

    # ---- render at *at least* 300 dpi for small fonts --------------------
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csRGB, alpha=False)

    img = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, pix.n
    )  # H×W×3 (RGB)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---- light pre-processing -------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # local contrast boost (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # simple Otsu (non-inverted) threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # ---- OCR -------------------------------------------------------------
    config = "--oem 3 --psm 6"  # LSTM, assume block of text
    tdata = pytesseract.image_to_data(
        thresh, output_type=pytesseract.Output.DICT, config=config
    )

    kw_lower = {w.lower() for w in kw}
    matched = set()
    BOX_COLOR = (0, 255, 0)

    for i, txt in enumerate(tdata["text"]):
        if int(tdata["conf"][i]) < 0 or not txt.strip():
            continue
        word = txt.strip()
        # record matches for caller
        if word.lower() in kw_lower:
            matched.add(word)
        # draw EVERY word’s box
        x, y, w, h = (tdata[k][i] for k in ("left", "top", "width", "height"))
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), BOX_COLOR, 1)
        cv2.putText(
            img_bgr, word, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1
        )

    return matched, img_bgr


def extract(data: PDFSet, dpi: int, output_dir: str = None, debug=True):
    # ----- configuration -------------------------------------------------
    CROP_PX = (20, 0, 200, 300)  # header strip (px)
    MIN_LENGTH = 4
    PROGRESS_STEP = 50
    KEYWORDS = [
        "Patient",
        "Hospital",
        "Referral",
        "Hearing",
        "Rosenbaum",
        "Masks",
    ]  # BOTH must appear
    # --------------------------------------------------------------------

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x0, y0, x1, y1 = CROP_PX
    w_pt = (x1 - x0) * 72 / dpi
    h_pt = (y1 - y0) * 72 / dpi

    totals, hits = 0, defaultdict(list)
    t0 = time.perf_counter()

    for pdf_name in data.filenames:
        src = Path(data.src) / pdf_name
        if not src.is_file():
            print(f"[WARN] Missing file: {src}")
            continue

        keep_doc = None  # destination for kept pages
        crop_doc = fitz.open()  # debug crops
        kept_list = []

        with fitz.open(src) as doc:
            for page in doc:
                totals += 1
                if totals % PROGRESS_STEP == 0:
                    print(f"[EVENT] Processed {totals} pages")

                matched, annotated = crop_read(page, CROP_PX, dpi, KEYWORDS)
                is_sheet = len(matched) >= MIN_LENGTH

                # Require BOTH Benton & Sheet
                if is_sheet:
                    if keep_doc is None:
                        keep_doc = fitz.open()
                    keep_doc.insert_pdf(
                        doc, from_page=page.number, to_page=page.number + 1
                    )
                    kept_list.append(page.number + 1)

                    print(f"[EVENT] {pdf_name} page {page.number} has word {matched}")

                # Save crop if debug OR not kept
                if debug or not is_sheet:
                    dst = crop_doc.new_page(width=w_pt, height=h_pt)
                    dst.insert_image(
                        dst.rect, stream=cv2.imencode(".png", annotated)[1].tobytes()
                    )

        # -------- write outputs ------------------------------------------
        if keep_doc:
            keep_doc.save(out_dir / f"{src.stem}.pdf", garbage=4, deflate=True)
            keep_doc.close()
            hits[src] = kept_list
            if debug:
                crop_doc.save(out_dir / f"crop_{src.stem}.pdf", garbage=4, deflate=True)
        else:
            crop_doc.save(out_dir / f"crop_{src.stem}.pdf", garbage=4, deflate=True)
        crop_doc.close()

    # -------- summary ----------------------------------------------------
    elapsed = time.perf_counter() - t0
    print("\n=== Extraction summary ===")
    if not hits:
        print("No pages containing both Benton and Sheet were found.")
    else:
        for pdf, pages in hits.items():
            print(
                f"{pdf.name}: page(s) {', '.join(map(str, pages))}" f" → {pdf.stem}.pdf"
            )
    print(f"\nTotal pages scanned: {totals}")
    print(f"Elapsed time: {elapsed:.1f} s")


##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.dataset
    output_dir = args.output
    dpi = args.dpi

    clear_pdfdir(output_dir)

    data = PDFSet(source_dir, load_format=f"numpy{dpi}")
    extract(data, dpi, output_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
