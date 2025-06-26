import os, argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
import time

from pdf_set import PDFSet

def parse_args():
    parser = argparse.ArgumentParser(description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording")
    
    parser.add_argument("-dataset", "-i", type=str, default='data_mini', help='Path to folder with list of full patient case files in PDF')
    parser.add_argument("-yellow_dir", "-y", type=str, default='case_outputs/yellow', help='Path to folder with list of patient yellow sheets in PDF')
    parser.add_argument("-red_dir", "-r", type=str, default='case_outputs/red', help='Path to folder with list of patient red sheets in PDF')
    
    return parser.parse_args()

def clear_pdfdir(path):
    for file in os.listdir(path): 
        if file != ".gitignore": os.remove(os.path.join(path, file))

##############################
# Extract yellow and red sheets
##############################

def extract(data: PDFSet, yellow_dir: str = None, red_dir: str = None):
    """Extract case set into two folders, for yellow and red sheets

    Args:
        data (PDFSet): Source Data [numpy mode]
        colormaps (np.ndarray[..., 200, 3]): Colormap description of the data
        yellow (str): Directory to save yellow sheets (if provided)
        red (str): Directory to save red sheets (if provided)
    """
    colormaps = data.get_colormaps()
    
    # Categorize by color
    for file_idx in range(len(data)):
        yellow_pages_idx = []
        yellow_prev_start = 0
        red_pages_idx = []
        num_pages = len(colormaps[file_idx])
        
        ### Parse in file
        for page_idx in range(num_pages):
            
            pixel_rgb = np.uint8([[colormaps[file_idx, page_idx]]])
            pixel_hsv = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2HSV)[0,0]
            hue, sat, val = pixel_hsv
            
            # Red
            if (sat > 20 and val > 50) and (hue > 160 or hue <= 0):
                red_pages_idx.append(page_idx)
                #print(pixel_hsv, file_idx, "red")
            
            # Yellow
            elif (sat > 20 and val > 50) and (16 <= hue < 40):
                yellow_pages_idx.append(page_idx)
                #print(pixel_hsv, file_idx, "yellow")
                
                # if a new chain of yellow, mark it so
                if page_idx-1 not in yellow_pages_idx:
                    yellow_prev_start = page_idx
                
            # Neither; check if previous was yellow or red
            else: 
                # If previous was yellow, check if > 4 in a row at the back end -> remove those
                if yellow_pages_idx and len(yellow_pages_idx) > 4 and page_idx > num_pages/2:
                    del yellow_pages_idx[yellow_pages_idx.index(yellow_prev_start):]
                
                # If previous was red, end loop because we don't need to check more; 
                if red_pages_idx:
                    break
        
        if yellow_dir and red_dir:
            ### Save pages
            reader = PdfReader(os.path.join(data.src, data.filenames[file_idx]))
            writer = PdfWriter()
            
            # If had red pages, add the red pages to a certain document
            if red_pages_idx:
                for page_idx in red_pages_idx:
                    writer.add_page(reader.pages[page_idx])
                with open(os.path.join(red_dir, f"{file_idx} pg{red_pages_idx[0]+1}{('-'+str(red_pages_idx[-1]+1))if len(red_pages_idx)>1 else ''}.pdf"), "wb") as f:
                    writer.write(f)

            # No red pages, then add all valid yellow pages to a certain document
            else:
                if yellow_pages_idx: # some might not have valid yellow pages
                    for page_idx in yellow_pages_idx:
                        writer.add_page(reader.pages[page_idx])
                    with open(os.path.join(yellow_dir, f"{file_idx} pg{yellow_pages_idx[0]+1}{('-'+str(yellow_pages_idx[-1]+1))if len(yellow_pages_idx)>1 else ''}.pdf"), "wb") as f:
                        writer.write(f)

##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.dataset
    yellow_dir = args.yellow_dir
    red_dir = args.red_dir
    
    clear_pdfdir(yellow_dir)
    clear_pdfdir(red_dir)
    
    data = PDFSet(source_dir, load_format='numpy20')
    extract(data, yellow_dir, red_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
