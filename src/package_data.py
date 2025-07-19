import argparse
import time
from pathlib import Path
import pandas as pd
import json
from openpyxl.utils import get_column_letter

from utils.flatten_labels import flatten_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleaned Patient Tables, Field Labels, Groundtruth-only Database \u2192 Paired Database"
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
        "-db_pth",
        "-db",
        type=str,
        default="case_outputs/final_data.xlsx",
        help="Path to database excel sheet",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/final_data.txt",
        help="Path to folder with debug information for scanning",
    )
    parser.add_argument(
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with all test names in tree structure",
    )

    return parser.parse_args()


# ═════════════════════  helper: print match logs for debugging  ════════════════


def showdebug(rows, debug_pth, aligns=None, spacing=1, header=""):
    """
    Print rows (iterables) with comma separators and aligned columns.

    rows   : iterable of row iterables (e.g., list of tuples/lists)
    aligns : iterable of alignment chars per column: '<' (left), '>' (right), '^' (center).
             If None, all columns left-aligned.
    spacing: number of spaces after each comma.
    """
    # Convert everything to strings first
    str_rows = [[str(cell) for cell in row] for row in rows]

    # Determine max width per column
    widths = [max(len(cell) for cell in col) for col in zip(*str_rows)]

    # Default alignments
    if aligns is None:
        aligns = ["<"] * len(widths)
    else:
        # pad/truncate aligns to match number of columns
        aligns = (list(aligns) + ["<"] * len(widths))[: len(widths)]

    # Build a format string segment for each column
    col_specs = [
        f"{{:{a}{w}}}" for a, w in zip(aligns, widths)  # e.g., '{:<10}' or '{:>5}'
    ]

    # Join with comma + spacing spaces
    sep = ", " * spacing if spacing else ","
    fmt = sep.join(col_specs)

    # Print each row
    with open(debug_pth, "a") as debug:
        debug.write(header)
        for row in str_rows:
            debug.write(fmt.format(*row) + "\n")


##############################
# Package patient info
##############################


def package(
    source_dir: str,
    db_pth: str,
    debug_pth: str,
    fl_pth: str,
):
    """
    Compiles case_outputs/tables_clean to fl_pth final excel
    """

    source_dir = Path(source_dir)
    fl_pth = Path(fl_pth)
    patient_csvs = []
    patient_ids = []
    patient_maps = []

    # ---------------------- standardize: map patient csv rows with the field labels -------------

    # Get field_labels with 'used' label
    fieldlabels1_base = [
        (field, False)
        for field in flatten_labels(json.loads(fl_pth.read_text()), blanks=False)
    ]

    for patient_fn in sorted(source_dir.iterdir()):
        fieldlabels1 = fieldlabels1_base.copy()

        # Get patient
        patient_csv = pd.read_csv(patient_fn).fillna("").astype(str)
        patient_csvs.append(patient_csv)
        patient_ids.append(patient_fn.stem)

        # Match patient rows with fieldlabels
        idx1, idx2 = 0, 0
        patient_map = []  # list of (idx2, idx1): field label to patient row
        debugstream = []
        while idx1 < patient_csv.shape[0] - 1:
            # See if all fields have been checked
            if idx2 < len(fieldlabels1):
                Y, used = fieldlabels1[idx2]

                # If the field is already filled out
                if used:
                    idx2 += 1
                    continue

                # See if field is a category or test
                if len(Y) == 1:  # if category, needs to also match casewise (uppercase)
                    X = patient_csv.iloc[idx1]["test_info"]
                    Y0 = Y[0].upper()
                    flag = "CATEG"
                elif len(Y) == 2:  # if test
                    X = patient_csv.iloc[idx1]["test_info"].lower()
                    Y0 = Y[1].lower()
                    flag = "TNAME"
                else:  # if subtest, just check with lower
                    X = patient_csv.iloc[idx1]["test_info"].lower()
                    Y0 = Y[2].lower()
                    flag = "STNAME"

                # Check if field is in the patient row
                if Y0 in X:
                    patient_map.append((idx2, idx1))
                    debugstream.append(
                        (
                            flag,
                            f"{idx1:02d}",
                            f"{idx2:02d}",
                            Y,
                            patient_csv.iloc[idx1].values,
                            f"'{Y0}' is in '{X}'",
                        )
                    )
                    fieldlabels1[idx2] = Y, True
                    idx1 += 1
                    idx2 = 0
                else:  # if field not in, then next field
                    idx2 += 1
            else:  # if so, the patient row does not correspond
                patient_map.append((idx2, None))
                debugstream.append(
                    (
                        "NMTCH",
                        f"{idx1:02d}",
                        f"{idx2:02d}",
                        "",
                        patient_csv.iloc[idx1].values,
                        "",
                    )
                )
                idx1 += 1
                idx2 = 0

        patient_maps.append(patient_map)

        # Print debug information
        showdebug(
            debugstream,
            debug_pth,
            header=f"=========================================={patient_fn.stem}\n",
        )

    # ---------------------- setup: load existing template -------------
    # Read the pre‐initialized final_data.xlsx (with MultiIndex rows & cols)
    df = pd.read_excel(db_pth, sheet_name="Results", header=[0, 1, 2], index_col=[0, 1])
    multi_cols = df.columns

    # ---------------------- build: fill patients into loaded DataFrame -------------

    # Now for each patient, write their 3 rows directly into df
    for patient_csv, patient_id, patient_map in zip(
        patient_csvs, patient_ids, patient_maps
    ):
        for fl_idx, patient_row_idx in patient_map:
            if patient_row_idx is None:
                continue
            row = patient_csv.iloc[patient_row_idx]
            col = multi_cols[fl_idx]
            # Write raw_score, comments, test_info
            df.at[(int(patient_id), "raw_score_pred"), col] = row["raw_score"]
            df.at[(int(patient_id), "comments_pred"), col] = row["comments"]
            df.at[(int(patient_id), "test_info_pred"), col] = row["test_info"]

    # ----------------------------------  EXPORT  ----------------------------------
    # Overwrite the "Results" sheet in the same workbook
    # (xlsxwriter can't append; switch to openpyxl for replace-on-write)
    with pd.ExcelWriter(
        db_pth, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name="Results")  # preserve MultiIndex
        ws = writer.sheets["Results"]

        # set every column width to 12
        for idx in range(1, ws.max_column + 1):
            letter = get_column_letter(idx)
            ws.column_dimensions[letter].width = 12


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    db_pth = args.db_pth
    debug_pth = args.debug
    fl_pth = args.fields

    package(source_dir, db_pth, debug_pth, fl_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
