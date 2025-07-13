import os
import argparse
import time
from table_ocr import scan, get_table
import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


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


def detect(
    source_dir: str,
    output_dir: str,
    xl_pth: str,
):
    """
    Creates case_outputs/tables/000x.csv of patient records
    """

    # ---------- Config ----------
    # seg_query = ["body1_", "body2_"]

    # --------------------------- Set up data ---------------------------
    if os.path.isfile(source_dir):
        source_files = [source_dir]
    else:
        source_files = glob.glob(os.path.join(source_dir, "*.png"))

    try:
        source_files = glob.glob(os.path.join(source_dir, "*.png"))
        if not source_files:
            print(f"No PNG files found in {source_dir}")
            return
        outputs = scan(source_files)
        tables = {
            path: get_table(path, [outs]) for path, outs in zip(source_files, outputs)
        }

        for path, table in tables.items():
            output_file = os.path.join(
                output_dir, os.path.splitext(os.path.basename(path))[0] + ".csv"
            )
            table.to_csv(output_file, index=False)

        print(f"[EVENT] Tables saved to {output_dir}")
    finally:
        pass

    # result = pd.DataFrame(columns=["Left Head", "Left", "Middle", "Right"])


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

    detect(source_dir, output_dir, xl_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
