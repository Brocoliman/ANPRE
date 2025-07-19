import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json

from utils.flatten_labels import flatten_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )
    parser.add_argument(
        "-spreadsheet",
        "-xl",
        type=str,
        default="case_outputs/patient_info.csv",
        help="Path to CSV file with patient info",
    )
    parser.add_argument(
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with all test names in tree structure",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/final_data.xlsx",
        help="Path to database excel sheet",
    )

    return parser.parse_args()


##############################
# Setup database patient info
##############################


def init_db(
    xl_pth: str,
    fl_pth: str,
    output_pth: str,
):
    """
    Creates final excel with field labels and all patient ids (no other identifiable patient info)
    """

    xl_pth = Path(xl_pth)
    fl_pth = Path(fl_pth)

    # ---------------------- setup: load patient ids and field labels -------------

    # load patient ids
    patient_ids = pd.read_csv(xl_pth).loc[:, "patient_id"]

    # get field labels flattened into tuples
    fl = flatten_labels(json.loads(fl_pth.read_text()))

    # ---------------------- build: create xlsx file -------------

    multi_cols = pd.MultiIndex.from_tuples(fl, names=["Category", "Test", "Subtest"])

    n_patients = len(patient_ids)
    data = np.empty((n_patients * 6, len(multi_cols)), dtype=object)

    # ---------------------- wrap final excel  -------------

    # add row MultiIndex
    row_tuples = [
        (pid, label)
        for pid in patient_ids
        for label in (
            "raw_score_pred",
            "raw_score_gt",
            "comments_pred",
            "comments_gt",
            "test_info_pred",
            "test_info_gt",
        )
    ]

    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["Patient", "Field"])

    df = pd.DataFrame(data, index=row_index, columns=multi_cols)

    # ----------------------------------  EXPORT  ----------------------------------

    with pd.ExcelWriter(output_pth, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Results")  # keep the index; it's meaningful

        ws = writer.sheets["Results"]
        ws.set_column(
            first_col=0, last_col=len(df.columns) - 1, width=12
        )  # uniform width (or auto-fit loop)


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    output_pth = args.output
    xl_pth = args.spreadsheet
    fl_pth = args.fields

    init_db(xl_pth, fl_pth, output_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
