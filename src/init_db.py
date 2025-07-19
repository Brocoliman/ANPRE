import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json

from utils.flatten_labels import flatten_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patient Info, Field Labels \u2192 Empty Database"
    )
    parser.add_argument(
        "-patient_info",
        "-pi",
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
        "-db_path",
        "-db",
        type=str,
        default="case_outputs/final_data.xlsx",
        help="Path to database excel sheet",
    )
    parser.add_argument(
        "--gt",
        action=argparse.BooleanOptionalAction,
        help="Whether to include groundtruth subrows",
    )

    return parser.parse_args()


##############################
# Setup database patient info
##############################


def init_db(pi_pth: str, fl_pth: str, db_pth: str, gt: bool):
    """
    Creates final excel with field labels and all patient ids (no other identifiable patient info)
    """

    pi_pth = Path(pi_pth)
    fl_pth = Path(fl_pth)

    # ---------------------- setup: load patient ids and field labels -------------

    # load patient ids
    patient_ids = pd.read_csv(pi_pth).loc[:, "patient_id"]

    # get field labels flattened into tuples
    fl = flatten_labels(json.loads(fl_pth.read_text()))

    # ---------------------- build: create xlsx file -------------

    multi_cols = pd.MultiIndex.from_tuples(fl, names=["Category", "Test", "Subtest"])

    n_patients = len(patient_ids)
    data = np.empty((n_patients * 3 * (2 if gt else 1), len(multi_cols)), dtype=object)

    # ---------------------- wrap final excel  -------------

    # add row MultiIndex
    subrows = (
        (
            "raw_score_pred",
            "raw_score_gt",
            "comments_pred",
            "comments_gt",
            "test_info_pred",
            "test_info_gt",
        )
        if gt
        else (
            "raw_score_pred",
            "comments_pred",
            "test_info_pred",
        )
    )
    row_tuples = [(pid, label) for pid in patient_ids for label in subrows]

    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["Patient", "Field"])

    df = pd.DataFrame(data, index=row_index, columns=multi_cols)

    # ----------------------------------  EXPORT  ----------------------------------

    with pd.ExcelWriter(db_pth, engine="xlsxwriter") as writer:
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
    db_pth = args.db_path
    pi_pth = args.patient_info
    fl_pth = args.fields
    gt = args.gt

    init_db(pi_pth, fl_pth, db_pth, gt)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
