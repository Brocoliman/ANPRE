import os
import argparse
import cv2
import numpy as np
from pypdf import PdfReader, PdfWriter
import time
import fitz
import pytesseract
from pathlib import Path
import re
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


def extract(data: PDFSet, dpi: int, output_dir: str = None, debug = False):
    """Extract case set into two folders, for yellow and red sheets

    Args:
        data (PDFSet): Source Data [numpy mode]
        colormaps (np.ndarray[..., 200, 3]): Colormap description of the data
        yellow (str): Directory to save yellow sheets (if provided)
        red (str): Directory to save red sheets (if provided)
    """
    
    # ----- configuration -----------------------------------------------------
    WORD          = "Benton"
    CROP_PX       = (40, 20, 540, 70)   # (x0, y0, x1, y1) in *pixels*
    PROGRESS_STEP = 50                  # pages between progress prints
    # ------------------------------------------------------------------------

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regex = re.compile(rf"\b{re.escape(WORD)}\b", re.IGNORECASE)
    x0, y0, x1, y1 = CROP_PX
    clip_rect_pt   = fitz.Rect(x0 * 72 / dpi, y0 * 72 / dpi,
                               x1 * 72 / dpi, y1 * 72 / dpi)
    w_pt = (x1 - x0) * 72 / dpi
    h_pt = (y1 - y0) * 72 / dpi

    # ---- output holders -----------------------------------------------------
    hits          = defaultdict(list)   # src Path → [page numbers]
    crop_doc      = fitz.open()         # debug PDF for all header crops
    total_pages   = 0
    t0            = time.perf_counter()

    for pdf_name in data.filenames:
        src_path = Path(data.src) / pdf_name
        if not src_path.is_file():
            print(f"[WARN] Source file missing: {src_path}")
            continue

        out_doc       = None               # lazy-create on first hit
        hit_page_nums = []

        with fitz.open(src_path) as doc:
            for page in doc:
                total_pages += 1
                if total_pages % PROGRESS_STEP == 0:
                    print(f"[EVENT] Processed {total_pages} pages")

                # ---- OCR the header strip ----------------------------------
                pix  = page.get_pixmap(dpi=dpi, clip=clip_rect_pt,
                                       colorspace=fitz.csRGB, alpha=False)
                img  = np.frombuffer(pix.samples, np.uint8).reshape(
                           pix.height, pix.width, pix.n)        # H×W×3 (RGB)

                if debug:
                    # • append crop to debug PDF (but never read it back)
                    dst_pg = crop_doc.new_page(width=w_pt, height=h_pt)
                    dst_pg.insert_image(dst_pg.rect, stream=pix.tobytes("png"))
                
                img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thr  = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                if regex.search(pytesseract.image_to_string(thr)):
                    if out_doc is None:
                        out_doc = fitz.open()
                    out_doc.insert_pdf(doc, from_page=page.number,
                                            to_page  =page.number+1)
                    print(f"[EVENT] {pdf_name} page {page.number} has word {WORD}")
                    hit_page_nums.append(page.number + 1)       # 1-based

        # ---- save per-patient extract --------------------------------------
        if out_doc:
            out_path = out_dir / f"{src_path.stem}.pdf"
            out_doc.save(out_path, garbage=4, deflate=True)
            out_doc.close()
            hits[src_path] = hit_page_nums

    # ---- save the debug crop PDF -------------------------------------------
    if debug:
        crop_pdf_path = out_dir / "cropped_headers.pdf"
        crop_doc.save(crop_pdf_path, garbage=4, deflate=True)
        crop_doc.close()
        print(f"[INFO] Saved all header crops → {crop_pdf_path}")
    
    # ----- summary -----------------------------------------------------------
    elapsed = time.perf_counter() - t0
    print("\n=== Benton extraction summary ===")
    if not hits:
        print(f"No pages containing '{WORD}' were found.")
    else:
        for pdf, pages in hits.items():
            plist = ", ".join(map(str, pages))
            print(f"{pdf.name}: page(s) {plist}  →  {pdf.stem}_Benton.pdf")
    print(f"\nTotal pages scanned: {total_pages}")
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

    data = PDFSet(source_dir, load_format=f"numpy{dpi}")
    extract(data, dpi, output_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
