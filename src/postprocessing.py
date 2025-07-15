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
        default="case_outputs/tables",
        help="Path to folder with list of csv tables for patient information",
    )
    parser.add_argument(
        "-aux",
        type=str,
        default="case_outputs/tables_concat",
        help="Path to folder with list of csv tables for patient record that are concatenated ",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/tables_clean",
        help="Path to folder with list of cleaned csv tables ",
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
# Postprocess
##############################


def postprocess(
    source_dir: str,
    output_dir: str,
):
    pass


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    concat_dir = args.aux
    output_dir = args.output
    xl_pth = args.spreadsheet

    clear_pdfdir(output_dir)

    postprocess(concat_dir, output_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
