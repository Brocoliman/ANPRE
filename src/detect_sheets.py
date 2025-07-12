import os
import argparse
import time
import pandas as pd
from table_ocr import get_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="case_outputs/seg",
        help="Path to folder with list of segmented sheets in PNG",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/tables",
        help="Path to folder with list of csv tables for patient information",
    )
    parser.add_argument(
        "-spreadsheet",
        "-xl",
        type=str,
        default="case_outputs/patient_data.csv",
        help="Path to CSV file with final patient data",
    )
    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


##############################
# Detect text
##############################


def segment(
    source_dir: str,
    output_dir: str,
    xl_pth: str,
):
    """
    Creates case_outputs/tables/000x.csv of patient records
    """

    # ---------- Config ----------
    seg_query = ["body1_", "body2_"]

    # --------------------------- Set up data ---------------------------
    df = pd.read_csv(xl_pth)
    fn_roots = df["filename"]  # 0000x.pdf

    for fn in fn_roots:
        result = pd.DataFrame(columns=["Left Head", "Left", "Middle", "Right"])
        for seg in seg_query:
            result = pd.concat(
                [result, get_table(os.path.join(source_dir, seg + fn[:-3] + "png"))]
            )
        result.to_csv(os.path.join(output_dir, fn))


##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    xl_pth = args.spreadsheet

    clear_pdfdir(output_dir)

    segment(source_dir, output_dir, xl_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
