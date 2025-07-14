import os
import argparse
import time
import pandas as pd


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
# Concatenate patient records first
##############################


def concat(
    source_dir: str,
    output_dir: str,
    xl_pth: str,
):
    # ------------------------ config ----------------------
    prefixes = ["body1_", "body2_"]  # all need to exist for valid file

    patients = pd.read_csv(xl_pth)
    found_patients = []
    missing_patients = []

    # Iterate through each patient
    for _, row in patients.iterrows():
        # Get filename
        record_id = int(row["record_id"])
        csv_stem = f"{record_id:05d}.csv"

        # See if all of patient's pages (file) exists (each prefix corresp to a page)
        filenames = [os.path.join(source_dir, prefix + csv_stem) for prefix in prefixes]
        if all(map(os.path.exists, filenames)):
            found_patients.append(record_id)
            # Now concatenate
            patient_csv = pd.concat(
                [pd.read_csv(os.path.join(file)) for file in filenames]
            )
            patient_csv.to_csv(os.path.join(output_dir, csv_stem), index=False)
        else:
            missing_patients.append(record_id)

    print(f"[REPORT] Data collected for {len(found_patients)} cases")
    print(f"[REPORT] Missing cases for {len(missing_patients)}: {missing_patients} ")


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

    concat(source_dir, concat_dir, xl_pth)
    postprocess(concat_dir, output_dir)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
