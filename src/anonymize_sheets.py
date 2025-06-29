import os
import argparse
import cv2
import numpy as np
from pypdf import PdfReader, PdfWriter
import time
import json
import fitz

from pdf_set import PDFSet

def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )

    parser.add_argument(
        "-src",
        "-i",
        type=str,
        default="data/mini",
        help="Path to folder with list of full patient case files in PDF",
    )
    parser.add_argument(
        "-anon_dir",
        "-a",
        type=str,
        default="case_outputs/yelanonlow",
        help="Path to folder to output patient anonymized sheets in PDF",
    )

    return parser.parse_args()

##############################
# Helper functions
##############################

def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))

def rect_bound(approx: np.ndarray) -> tuple[np.ndarray, int]:
    """Create rectangular (sides parallel to axes) boundary from approximation contour

    Args:
        approx (np.ndarray): approximation contour extracted from image

    Returns:
        tuple[np.ndarray, int]: position of rectangle and given area
    """
    x, y, w, h = cv2.boundingRect(approx)
    box = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    return box, w*h

def para_bound(approx: np.ndarray) -> tuple[np.ndarray, int]:
    """Create parallelogram boundary from approximation contour

    Args:
        approx (np.ndarray): approximation contour extracted from image

    Returns:
        tuple[np.ndarray, int]: position of parallelogram and given area
    """
    rect = cv2.minAreaRect(approx) 
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    (_, _), (w, h), _ = rect
    return box, w*h

def pos (box: np.ndarray, rel):
    """Bilinear interpolation in quadrilateral interior

    Args:
        box (np.ndarray): coordinates of box (top left -> top right -> bottom right -> bottom left)
        rel (tuple(int, int)): interpolation parameter along horizontal and vertical coordinates 

    Returns:
        tuple (tuple(int, int)): absolute position (cv2) of interpolate position
    """
    p, q = tuple(rel)
    # Assumed order, may be incorrect
    topleft_x, topleft_y = box[0]
    topright_x, topright_y = box[1]
    bottomright_x, bottomright_y = box[2]
    bottomleft_x, bottomleft_y = box[3]
    return (
        (1-p)*(1-q)*topleft_x+(1-q)*p*topright_x+p*q*bottomright_x+(1-p)*q*bottomleft_x,
        (1-p)*(1-q)*topleft_y+(1-q)*p*topright_y+p*q*bottomright_y+(1-p)*q*bottomleft_y
    )

##############################
# Anonymize summary sheets
##############################

def anonymize(data: PDFSet, anon_dir: str = None):
    anonSet = data.copy()
    
    template = '../configs/template2025.json'
    with open(template) as f:
        config = json.load(f)
    anchors = config.get('anchors')
    blackout_query = config.get('censor')

    assert len(anchors) == len(anchor_boxes)

    for doc_idx in range(len(anonSet)):
        page, img, fn = anonSet[doc_idx,0]
        img = img.copy()
        
        # Reduce image channels
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        # Reduce image noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0) 

        # Define adaptive threshold to convert image to BW, adjust for light
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

        # Find external contours, and compress information
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anchor_boxes = []
        valids = 0
        for contour in contours:
            # Simplify contour shape to retrieve bounding polygon
            approx = cv2.approxPolyDP(contour, 0.0395 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                valids += 1
                box, area = para_bound(approx)
                if area > 10000:
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
                    anchor_boxes.append((box, area))

        anchor_boxes.sort(key=lambda pair: pair[1])
        
        for field in blackout_query:
            parent_name = field['parent']
            parent_idx = anchors.index(parent_name)
            assert parent_idx != -1
            box, _ = anchor_boxes[parent_idx]
            abs_x1, abs_y1 = pos(box, field['topleft'])
            abs_x2, abs_y2 = pos(box, field['bottomright'])
            # turn to fitz units
            a = 72/300
            print(abs_x1*a, abs_y1*a, abs_x2*a, abs_y2*a)
            redact_area = fitz.Rect(abs_x1*a, abs_y1*a, abs_x2*a, abs_y2*a)
            page.add_redact_annot(redact_area, fill=(0, 0, 0))
        page.apply_redactions()
    
    if anon_dir:
        anonSet.save(anon_dir)
    
    return anonSet
    

##############################
# Main: Runs all functions from summary sheet to anonymization
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.src
    anon_dir = args.anon_dir

    clear_pdfdir(anon_dir)

    data = PDFSet.fromFolder(source_dir, "numpy300 fitz")
    anonymize(data, anon_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
