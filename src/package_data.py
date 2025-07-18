import os
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json


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
        "-spreadsheet",
        "-xl",
        type=str,
        default="case_outputs/patient_data.csv",
        help="Path to CSV file with patient info",
    )
    parser.add_argument(
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with all test names in tree structure",
    )

    return parser.parse_args()


# ═════════════════════  helper: flatten field_labels.json tree  ════════════════


def flatten_labels(label_tree, blanks=True):
    flat = []
    for category, tests in label_tree.items():
        if blanks:
            flat.append((category, "", ""))
        else:
            flat.append((category,))
        if isinstance(tests, list):
            for t in tests:
                if blanks:
                    flat.append((category, t, ""))
                else:
                    flat.append((category, t))
        else:  # dict
            for test_name, subs in tests.items():
                if subs:
                    for sub in subs:
                        flat.append((category, test_name, sub))
                else:
                    if blanks:
                        flat.append((category, test_name, ""))
                    else:
                        flat.append((category, test_name))
    return flat


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
    output_pth: str,
    debug_pth: str,
    xl_pth: str,
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
                else:  # if tname, just check with lower
                    X = patient_csv.iloc[idx1]["test_info"].lower()
                    Y0 = Y[-1].lower()
                    flag = "TNAME"

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

    # ---------------------- setup: create xlsx file -------------

    fieldlabels2 = flatten_labels(json.loads(fl_pth.read_text()))
    multi_cols = pd.MultiIndex.from_tuples(
        fieldlabels2, names=["Category", "Test", "Subtest"]
    )
    # <- replace with real list/iterator

    n_patients = len(patient_ids)
    data = np.empty((n_patients * 3, len(multi_cols)), dtype=object)

    thres_low, thres_mid = 0.7, 0.9
    low_conf_cells = []
    mid_conf_cells = []

    # ---------------------- build: fill patients -------------

    for p_idx, patient_csv in enumerate(patient_csvs):
        patient_map = patient_maps[p_idx]
        base = p_idx * 3  # first row (raw_score) for this patient
        for fl_idx, patient_row_idx in patient_map:
            if patient_row_idx is None:
                continue

            # Get the full row from the source CSV to access conf scores
            source_row = patient_csv.iloc[patient_row_idx]

            # Helper to check confidence and store cell coordinates
            def check_confidence(conf_str, df_row_idx, df_col_idx):
                if not conf_str:
                    return
                try:
                    conf_val = float(conf_str)
                    # Excel row = DataFrame row + 3 header rows (for column MultiIndex)
                    # Excel col = DataFrame col + 2 index cols (for row MultiIndex)
                    excel_r = df_row_idx + 4
                    excel_c = df_col_idx + 2

                    if conf_val < thres_low:
                        low_conf_cells.append((excel_r, excel_c))
                    elif conf_val < thres_mid:
                        mid_conf_cells.append((excel_r, excel_c))
                except (ValueError, TypeError):
                    pass  # Ignore non-numeric confidence values

            # ------------------ RAW SCORE ------------------
            df_row_idx_raw = base
            data[df_row_idx_raw, fl_idx] = source_row["raw_score"]
            check_confidence(source_row.get("raw_score_conf"), df_row_idx_raw, fl_idx)

            # ------------------ COMMENTS -------------------
            df_row_idx_comm = base + 1
            data[df_row_idx_comm, fl_idx] = source_row["comments"]
            check_confidence(source_row.get("comments_conf"), df_row_idx_comm, fl_idx)

            # ------------------ TEST-INFO ------------------
            df_row_idx_info = base + 2
            data[df_row_idx_info, fl_idx] = source_row["test_info"]
            check_confidence(source_row.get("test_info_conf"), df_row_idx_info, fl_idx)

    # ---------------------- wrap final excel  -------------

    # add row MultiIndex
    row_tuples = [
        (pid, label)
        for pid in patient_ids
        for label in ("raw_score", "comments", "test_info")
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

        wb = writer.book
        low_conf_format = wb.add_format(
            {
                "bg_color": "#EF5567",
            }
        )
        mid_conf_format = wb.add_format(
            {
                "bg_color": "#FFEE6D",
            }
        )

        for r, c in low_conf_cells:
            # Map Excel coordinates back to the numpy array's index
            df_row = r - 4
            df_col = c - 2
            # Get the original value
            value = data[df_row, df_col]
            # Write the value AND the format
            ws.write(r, c, value, low_conf_format)

        for r, c in mid_conf_cells:
            # Map Excel coordinates back to the numpy array's index
            df_row = r - 4
            df_col = c - 2
            # Get the original value
            value = data[df_row, df_col]
            # Write the value AND the format
            ws.write(r, c, value, mid_conf_format)


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_pth = args.output
    debug_pth = args.debug
    xl_pth = args.spreadsheet
    fl_pth = args.fields

    if os.path.exists(output_pth):
        os.remove(output_pth)
    if os.path.exists(debug_pth):
        os.remove(debug_pth)

    package(source_dir, output_pth, debug_pth, xl_pth, fl_pth)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
