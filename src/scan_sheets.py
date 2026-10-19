import os
import argparse
import time
import pandas as pd
import json
from pathlib import Path
from utils.table_ocr import ocr, get_table, initialize_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patient Sheet Segments, Patient Info, Segmentation Template \u2192 Raw Patient Tables, (Groundtruth)"
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
        help="Path to folder with list of raw patient csv tables",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/tables",
        help="Path to folder with debug information for scanning",
    )
    parser.add_argument(
        "-template",
        "-t",
        type=str,
        default="instructions/seg_template.json",
        help="Path to sheet template file",
    )
    parser.add_argument(
        "-patient_info",
        "-pi",
        type=str,
        default="case_outputs/patient_info.csv",
        help="Path to CSV file with final patient data",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        help="Whether to process on GPU or CPU",
    )

    return parser.parse_args()


def clear_pdfdir(path):
    for file in os.listdir(path):
        if file != ".gitignore":
            os.remove(os.path.join(path, file))


##############################
# Scan text
##############################


def scan(
    source_dir: str,
    output_dir: str,
    debug_dir: str,
    prefixes: list[str],
    gpu=False,
):
    """
    Creates case_outputs/tables/000x.csv of patient records
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    for fn in sorted(source_dir.iterdir()):
        outputs = ocr(fn, debug_dir)
        tables = {fn: get_table(fn, outputs, prefixes)}

        for path, table in tables.items():
            output_file = output_dir / (path.stem + ".csv")
            table.to_csv(output_file, index=False)

        print(f"[EVENT] Tables saved to {output_dir}")


##############################
# Concatenate patient records
##############################


def concat(
    source_dir: str,
    output_dir: str,
    pi_pth: str,
    prefixes: list[str],
):
    # ------------------------ config ----------------------

    patients = pd.read_csv(pi_pth)
    found_patients = []
    missing_patients = []

    # Iterate through each patient
    for _, row in patients.iterrows():
        # Get filename
        patient_id = int(row["patient_id"])
        csv_stem = f"{patient_id:05d}.csv"

        # See if all of patient's pages (file) exists (each prefix corresp to a page)
        filenames = [os.path.join(source_dir, prefix + csv_stem) for prefix in prefixes]

        if all(map(os.path.exists, filenames)):
            found_patients.append(patient_id)
            # Now concatenate
            patient_csv = pd.concat(
                [pd.read_csv(os.path.join(file)) for file in filenames]
            )
            patient_csv.to_csv(os.path.join(output_dir, csv_stem), index=False)
        else:
            missing_patients.append(patient_id)
        for fn in filenames:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    print(f"[REPORT] Data collected for {len(found_patients)} cases")
    print(f"[REPORT] Missing cases for {len(missing_patients)}: {missing_patients} ")


##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    pi_pth = args.patient_info
    template_pth = args.template
    use_gpu = args.gpu

    initialize_pipeline(use_gpu)

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    with open(template_pth) as f:
        prefixes = [segname + "_" for segname in json.load(f)["sections"].keys()]

    scan(source_dir, output_dir, debug_dir, prefixes, use_gpu)
    concat(output_dir, output_dir, pi_pth, prefixes)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
