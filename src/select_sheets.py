import os
import argparse
import cv2
import numpy as np
from pandas import DataFrame
import time
import fitz
import pytesseract
from pathlib import Path
from collections import defaultdict

BASE_DPI = 150  # dpi which CROP_PX derived from
CROP_PX = (20, 0, 200, 300)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data collection tool for patient neuropsychological assessment scores"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="data/mini",
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
        "-debug",
        "-d",
        type=str,
        default="debug/sheets",
        help="Path to folder for debugging sheet scanning",
    )
    parser.add_argument(
        "-spreadsheet",
        "-xl",
        type=str,
        default="case_outputs/patient_data.csv",
        help="Path to CSV file with final patient data",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        default=150,
        help="Loading and processing DPI of PDF for selection",
    )
    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


# ═════════════════════  helper: crop and then use OCR to detect tokens  ════════════════


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
    scale = 72 / BASE_DPI
    x0, y0, x1, y1 = crop_px
    clip = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)

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
        for kw in kw_lower:
            if kw in word.lower():
                matched.add(word)
        # draw EVERY word’s box
        x, y, w, h = (tdata[k][i] for k in ("left", "top", "width", "height"))
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), BOX_COLOR, 1)
        cv2.putText(
            img_bgr, word, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1
        )

    return matched, img_bgr


# ═════════════════════  select sheets from a folder of cases  ════════════════


def select(
    source_dir: str,
    output_dir: str,
    debug_dir: str,
    xl_pth: str,
    dpi: int,
):
    # ----- configuration -------------------------------------------------
    MIN_LENGTH = 4
    KEYWORDS = [
        "Benton",
        "Date",
        "Patient",
        "Hospital",
        "Referral",
        "HX",
        "Hearing",
        "Rosenbaum",
        "Masks",
    ]
    # --------------------------------------------------------------------

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    debug_dir = Path(debug_dir)
    xl_pth = Path(xl_pth)

    filenames = source_dir.iterdir()

    # Contains record_id, new filename, other fields not added yet
    prelim_datalist = []

    w_pt = (CROP_PX[2] - CROP_PX[0]) * 72 / BASE_DPI
    h_pt = (CROP_PX[3] - CROP_PX[1]) * 72 / BASE_DPI

    totals, hits = 0, defaultdict(list)
    t0 = time.perf_counter()

    occurrence_dicts = {}

    for idx, filename in enumerate(filenames):
        if not filename.is_file():
            print(f"[WARN] Missing file: {filename}")
            continue
        if not filename.suffix == ".pdf":
            continue

        keep_doc = None  # destination for kept pages
        debug_doc = fitz.open()  # debug crops
        kept_list = []

        with fitz.open(filename) as doc:
            for page in doc:
                totals += 1

                matched, annotated = crop_read(page, CROP_PX, dpi, KEYWORDS)
                is_sheet = len(matched) >= MIN_LENGTH

                # Sheet inclusion
                if is_sheet:
                    if keep_doc is None:
                        keep_doc = fitz.open()
                    keep_doc.insert_pdf(
                        doc, from_page=page.number, to_page=page.number + 1
                    )
                    kept_list.append(page.number + 1)

                    print(f"[EVENT] {filename} page {page.number} has word {matched}")

                # Save crop if debug OR not kept
                if debug_dir:
                    dst = debug_doc.new_page(width=w_pt, height=h_pt)
                    dst.insert_image(
                        dst.rect, stream=cv2.imencode(".png", annotated)[1].tobytes()
                    )

        # -------- write outputs ------------------------------------------
        if keep_doc:
            new_filename = f"{idx:05d}.pdf"
            keep_doc.save(output_dir / new_filename, garbage=4, deflate=True)
            keep_doc.close()

            # -------- extract all available info from filename ------------------------------------------
            data_tokens = filename.stem.split(" ")
            # if name is more than 2 space-separated words
            if len(data_tokens) > 6:
                data_tokens = [data_tokens[0] + data_tokens[1]] + data_tokens[2:]
            lastname, firstname, mrn, exam_y, exam_m, exam_d = data_tokens

            if mrn not in occurrence_dicts.keys():
                occurrence_dicts.update({mrn: 1})
            else:
                occurrence_dicts[mrn] += 1

            prelim_datalist.append(
                [
                    idx,  # record_id
                    new_filename,  # filename
                    None,  # format (will have to be determined)
                    lastname,  # lastname
                    firstname,  # firstname
                    occurrence_dicts[mrn],  # occurrence number
                    int(exam_m),
                    int(exam_d),
                    int(exam_y),
                    None,
                    None,
                    None,
                ]
            )
            hits[filename] = kept_list

            if debug_dir:
                debug_doc.save(debug_dir / new_filename, garbage=4, deflate=True)
        else:
            debug_doc.save(debug_dir / f"{filename.stem}.pdf", garbage=4, deflate=True)
        debug_doc.close()

    # Save to excel
    col_names = [
        "record_id",  # not mrn
        "filename",
        "format",
        "lastname",
        "firstname",
        "occurrence",
        "exam_m",
        "exam_d",
        "exam_y",
        "dob_m",
        "dob_d",
        "dob_y",
    ]
    prelim_df = DataFrame(prelim_datalist, columns=col_names)
    prelim_df.to_csv(xl_pth, index=False)

    # -------- summary ----------------------------------------------------
    elapsed = time.perf_counter() - t0
    print("\n=== Selection summary ===")
    if not hits:
        print("No pages containing both Benton and Sheet were found.")
    else:
        for pdf, pages in hits.items():
            print(
                f"{pdf.name}: page(s) {', '.join(map(str, pages))}" f" → {pdf.stem}.pdf"
            )
    print(f"\nTotal pages scanned: {totals}")
    print(f"Elapsed time: {elapsed:.1f} s")


# ═════════════════════  Main: Runs all functions from source data to sheets  ════════════════

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    dpi = args.dpi
    xl_pth = args.spreadsheet

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    select(source_dir, output_dir, debug_dir, xl_pth, dpi)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
