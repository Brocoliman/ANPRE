import os
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
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
        default="case_outputs/sheets_debug",
        help="Path to folder for debugging sheet detection",
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

    pass


##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.dataset
    output_dir = args.output
    debug_dir = args.debug
    xl_pth = args.spreadsheet

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
