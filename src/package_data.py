import os
import argparse
import time
from pathlib import Path
import pandas as pd
import json
from PIL import Image as PILImage
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill
from openpyxl.drawing.image import Image as OpenpyxlImage

from utils.flatten_labels import flatten_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleaned Patient Tables, Field Labels, Segmentation Template, Empty Database \u2192 Fill (possibly paired --gt) Database"
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
        "-seg_pth",
        "-seg",
        type=str,
        default="case_outputs/seg",
        help="Path to segmented patient images",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/final_data.txt",
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
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with all test names in tree structure",
    )
    parser.add_argument(
        "--gt",
        action=argparse.BooleanOptionalAction,
        help="Whether to load images into groundtruth subrows",
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


# ═════════════════════  helper: resize image to fit cell  ════════════════


def resize_image_to_cell(image, cell_width_pixels, cell_height_pixels):
    """
    Resize a PIL Image to fit within cell dimensions while preserving aspect ratio.

    Args:
        image: PIL.Image object
        cell_width_pixels: Target cell width in pixels
        cell_height_pixels: Target cell height in pixels

    Returns:
        PIL.Image object resized to fit within cell
    """
    # img_width, img_height = image.size
    # aspect_ratio = img_width / img_height

    # # Calculate scaling factor to fit within cell dimensions
    # width_ratio = cell_width_pixels / img_width
    # height_ratio = cell_height_pixels / img_height
    # scaling_factor = max(width_ratio, height_ratio)  # Preserve aspect ratio

    # # Compute new dimensions
    # new_width = int(img_width * scaling_factor)
    # new_height = int(img_height * scaling_factor)

    # # Resize image
    # return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

    return image.resize(
        (int(cell_width_pixels), int(cell_height_pixels)), PILImage.Resampling.LANCZOS
    )


##############################
# Package patient info
##############################


def package(
    source_dir: str,
    db_pth: str,
    seg_pth: str,
    debug_pth: str,
    fl_pth: str,
    prefixes: list[str],
    load_gt: bool,
    load_gt_range: list,
):
    """
    Compiles case_outputs/tables_clean to fl_pth final excel
    """

    source_dir = Path(source_dir)
    fl_pth = Path(fl_pth)
    patient_csvs = []
    patient_ids = []
    patient_maps = []

    seg_pth = Path(seg_pth)

    # ====================== standardize: map patient csv rows with the field labels ==============

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

    # ====================== load existing template ==============
    # Read the pre‐initialized final_data.xlsx (with MultiIndex rows & cols)
    df = pd.read_excel(db_pth, sheet_name="Results", header=[0, 1, 2], index_col=[0, 1])
    multi_cols = df.columns

    # ====================== build: fill patients into loaded DataFrame ==============

    # Define confidence fillings
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    yellow_fill = PatternFill(
        start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"
    )
    white_fill = PatternFill(
        start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"
    )

    HEADER_ROWS = 3
    INDEX_COLS = len(df.index.names)

    # ---------------------- build ----------------------
    # iterate over patients
    for patient_csv, patient_id, patient_map in zip(
        patient_csvs, patient_ids, patient_maps
    ):
        patient_id_5dig = patient_id
        patient_id_int = int(patient_id)
        load_this_gt = load_gt_range[0] <= int(patient_id_int) <= load_gt_range[1]

        # load the segmented images for this patient
        if load_this_gt:
            segs = dict()
            for prefix in prefixes:
                segs.update(
                    {
                        prefix: PILImage.open(
                            seg_pth / (prefix + "_" + patient_id_5dig + ".png")
                        )
                    }
                )

        # iterate over field labels
        for fl_idx, patient_row_idx in patient_map:
            if patient_row_idx is None:
                continue

            # ---------------------- data ----------------------
            # get the row and column
            row = patient_csv.iloc[
                patient_row_idx
            ]  # the patient row information (object)
            col = multi_cols[fl_idx]  # exact field label (index)

            # Write raw_score, comments, test_info
            df.at[(int(patient_id), "raw_score_pred"), col] = row["raw_score"]
            df.at[(int(patient_id), "comments_pred"), col] = row["comments"]
            df.at[(int(patient_id), "test_info_pred"), col] = row["test_info"]

            # which Excel column is this field_label?
            excel_col = df.columns.get_loc(col) + INDEX_COLS + 1

            # for each of the three metrics:
            for metric in ("raw_score", "test_info", "comments"):
                # make sure this has a value
                if not row[metric]:
                    continue

                pred_label = metric + "_pred"
                conf_label = metric + "_conf"

                # ---------------------confidence----------------
                # find the Excel row
                excel_data_row = (
                    df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 2
                )

                # pull the confidence straight from the CSV
                conf_val = patient_csv.iloc[patient_row_idx][conf_label]
                conf_val = float(conf_val) if conf_val != "" else 1.0

                # choose fill
                if conf_val < 0.7:
                    fill = red_fill
                elif conf_val < 0.9:
                    fill = yellow_fill
                else:
                    fill = white_fill

                # Note: We’ll apply the fill after writing the DataFrame

                if not load_this_gt:
                    continue

                # ---------------------groundtruth----------------
                excel_gt_row = (
                    df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 3
                )

                cn_x1, cn_y1, cn_x2, cn_y2 = (
                    metric + "_x1",
                    metric + "_y1",
                    metric + "_x2",
                    metric + "_y2",
                )  # column names
                segname = patient_csv.iloc[patient_row_idx]["head"][
                    :-1
                ]  # e.g. body1; key for 'segs'
                cropbox = (
                    float(row[cn_x1]),
                    float(row[cn_y1]),
                    float(row[cn_x2]),
                    float(row[cn_y2]),
                )
                cropimg = segs[segname].crop(cropbox)
                # Resize to fit cell (12 char widths ≈ 96 pixels, 50 points ≈ 66.65 pixels)
                cropimg = resize_image_to_cell(
                    cropimg, cell_width_pixels=96, cell_height_pixels=66.65 / 3
                )
                temp_image_path = (
                    f"cropped_img_{patient_id}_{metric}.jpg"  # Unique temp file
                )
                cropimg.save(temp_image_path)
                cropimg_excel = OpenpyxlImage(temp_image_path)
                excel_col_letter = get_column_letter(excel_col)
                cropimg_excel.anchor = f"{excel_col_letter}{excel_gt_row}"
                # Note: We’ll add the image after writing the DataFrame

    temp_files = []  # List to store temporary image file paths

    # Overwrite the "Results" sheet in the same workbook
    with pd.ExcelWriter(
        db_pth, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        # Write the updated DataFrame
        df.to_excel(writer, sheet_name="Results")  # Write after all df.at modifications
        ws = writer.sheets["Results"]

        # Set every column width to 12
        for idx in range(1, ws.max_column + 1):
            letter = get_column_letter(idx)
            ws.column_dimensions[letter].width = 12

        # Re-iterate to apply fills and images (since ws is fresh after df.to_excel)
        for patient_csv, patient_id, patient_map in zip(
            patient_csvs, patient_ids, patient_maps
        ):
            patient_id_int = int(patient_id)
            load_this_gt = load_gt_range[0] <= int(patient_id_int) <= load_gt_range[1]

            if load_this_gt:
                segs = dict()
                for prefix in prefixes:
                    segs.update(
                        {
                            prefix: PILImage.open(
                                seg_pth / (prefix + "_" + patient_id + ".png")
                            )
                        }
                    )

            for fl_idx, patient_row_idx in patient_map:
                if patient_row_idx is None:
                    continue

                row = patient_csv.iloc[patient_row_idx]
                col = multi_cols[fl_idx]
                excel_col = df.columns.get_loc(col) + INDEX_COLS + 1

                for metric in ("raw_score", "test_info", "comments"):
                    if not row[metric]:
                        continue

                    pred_label = metric + "_pred"
                    conf_label = metric + "_conf"

                    # Apply confidence fill
                    excel_data_row = (
                        df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 2
                    )
                    conf_val = patient_csv.iloc[patient_row_idx][conf_label]
                    conf_val = float(conf_val) if conf_val != "" else 1.0
                    if conf_val < 0.7:
                        fill = red_fill
                    elif conf_val < 0.9:
                        fill = yellow_fill
                    else:
                        fill = white_fill
                    ws.cell(row=excel_data_row, column=excel_col).fill = fill

                    if not load_this_gt:
                        continue

                    # Apply groundtruth image
                    excel_gt_row = (
                        df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 3
                    )
                    cn_x1, cn_y1, cn_x2, cn_y2 = (
                        metric + "_x1",
                        metric + "_y1",
                        metric + "_x2",
                        metric + "_y2",
                    )
                    segname = patient_csv.iloc[patient_row_idx]["head"][:-1]
                    cropbox = (
                        float(row[cn_x1]),
                        float(row[cn_y1]),
                        float(row[cn_x2]),
                        float(row[cn_y2]),
                    )
                    cropimg = segs[segname].crop(cropbox)
                    # Resize to fit cell (12 char widths ≈ 96 pixels, 50 points ≈ 66.65 pixels)
                    cropimg = resize_image_to_cell(
                        cropimg, cell_width_pixels=96, cell_height_pixels=66.65 / 3
                    )
                    temp_image_path = f"cropped_img_{patient_id}_{metric}.jpg"
                    cropimg.save(temp_image_path)
                    cropimg_excel = OpenpyxlImage(temp_image_path)
                    excel_col_letter = get_column_letter(excel_col)
                    cropimg_excel.anchor = f"{excel_col_letter}{excel_gt_row}"
                    ws.add_image(cropimg_excel)
                    temp_files.append(temp_image_path)  # Store for later cleanup

    # Clean up all temporary image files after saving
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except FileNotFoundError:
            pass  # Ignore if file was already deleted


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    db_pth = args.db_pth
    seg_pth = args.seg_pth
    debug_pth = args.debug
    template_pth = args.template
    fl_pth = args.fields
    load_gt = args.gt

    # inclusive on both sides
    load_gt_range = [0, 10]

    with open(template_pth) as f:
        prefixes = json.load(f)["sections"].keys()

    package(
        source_dir, db_pth, seg_pth, debug_pth, fl_pth, prefixes, load_gt, load_gt_range
    )

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
