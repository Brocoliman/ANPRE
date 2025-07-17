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
        default="case_outputs/tables_clean",
        help="Path to folder with cleaned patient CSV data ready for packaging into database",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/tables",
        help="Path to database excel sheet",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/tables",
        help="Path to folder with debug information for scanning",
    )
    parser.add_argument(
        "-spreadsheet",
        "-xl",
        type=str,
        default="case_outputs/patient_data.csv",
        help="Path to CSV file with final patient data",
    )
    parser.add_argument(
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with final patient data",
    )

    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


##############################
# Package patient info
##############################


def package(
    source_dir: str,
    output_dir: str,
    debug_dir: str,
    xl_pth: str,
    fl_pth: str,
):
    """
    Compiles case_outputs/tables_clean to fl_pth final excel
    """
    ...


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    xl_pth = args.spreadsheet
    fl_pth = args.fields

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    package(source_dir, output_dir, debug_dir, xl_pth, fl_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
