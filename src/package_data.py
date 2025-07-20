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


def showdebug(rows, debug_pth, aligns=None, spacing=1, header=""):
    """
    Print rows (iterables) with comma separators and aligned columns.

    rows   : iterable of row iterables (e.g., list of tuples/lists)
    aligns : iterable of alignment chars per column: '<' (left), '>' (right), '^' (center).
             If None, all columns left-aligned.
    spacing: number of spaces after each comma.
    """
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(cell) for cell in col) for col in zip(*str_rows)]
    if aligns is None:
        aligns = ["<"] * len(widths)
    else:
        aligns = (list(aligns) + ["<"] * len(widths))[: len(widths)]
    col_specs = [f"{{:{a}{w}}}" for a, w in zip(aligns, widths)]
    sep = ", " * spacing if spacing else ","
    fmt = sep.join(col_specs)
    with open(debug_pth, "a") as debug:
        debug.write(header)
        for row in str_rows:
            debug.write(fmt.format(*row) + "\n")


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
    img_width, img_height = image.size
    width_ratio = cell_width_pixels / img_width
    height_ratio = cell_height_pixels / img_height
    scaling_factor = min(width_ratio, height_ratio)
    new_width = int(img_width * scaling_factor)
    new_height = int(img_height * scaling_factor)
    return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)


def package(
    source_dir: str,
    db_pth: str,
    seg_pth: str,
    debug_pth: str,
    fl_pth: str,
    prefixes: list[str],
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
    fieldlabels1_base = [
        (field, False)
        for field in flatten_labels(json.loads(fl_pth.read_text()), blanks=False)
    ]

    for patient_fn in sorted(source_dir.iterdir()):
        fieldlabels1 = fieldlabels1_base.copy()
        patient_csv = pd.read_csv(patient_fn).fillna("").astype(str)
        patient_csvs.append(patient_csv)
        patient_ids.append(patient_fn.stem)

        idx1, idx2 = 0, 0
        patient_map = []
        mapping_debugstream = []
        while idx1 < patient_csv.shape[0]:
            if idx2 < len(fieldlabels1):
                Y, used = fieldlabels1[idx2]
                if used:
                    idx2 += 1
                    continue
                if len(Y) == 1:
                    X = patient_csv.iloc[idx1]["test_info"]
                    Y0 = Y[0].upper()
                    flag = "CATEG"
                elif len(Y) == 2:
                    X = patient_csv.iloc[idx1]["test_info"].lower()
                    Y0 = Y[1].lower()
                    flag = "TNAME"
                else:
                    X = patient_csv.iloc[idx1]["test_info"].lower()
                    Y0 = Y[2].lower()
                    flag = "STNAME"
                if Y0 in X:
                    patient_map.append((idx2, idx1))
                    mapping_debugstream.append(
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
                    idx2 += 1
                else:
                    idx2 += 1
            else:
                patient_map.append((idx2, None))
                mapping_debugstream.append(
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
        showdebug(
            mapping_debugstream,
            debug_pth,
            header=f"=========================================={patient_fn.stem}\n",
        )

    # ====================== load existing template ==============
    df = pd.read_excel(db_pth, sheet_name="Results", header=[0, 1, 2], index_col=[0, 1])
    multi_cols = df.columns

    # ====================== build: fill patients into loaded DataFrame ==============
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    yellow_fill = PatternFill(
        start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"
    )
    white_fill = PatternFill(
        start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"
    )
    HEADER_ROWS = 3
    INDEX_COLS = len(df.index.names)
    temp_files = []

    loading_debugstream = ""

    # First loop: Populate DataFrame only
    for patient_csv, patient_id, patient_map in zip(
        patient_csvs, patient_ids, patient_maps
    ):
        patient_id_int = int(patient_id)
        for fl_idx, patient_row_idx in patient_map:
            if patient_row_idx is None:
                continue
            row = patient_csv.iloc[patient_row_idx]
            col = multi_cols[fl_idx]
            df.at[(patient_id_int, "raw_score_pred"), col] = row["raw_score"]
            df.at[(patient_id_int, "comments_pred"), col] = row["comments"]
            df.at[(patient_id_int, "test_info_pred"), col] = row["test_info"]

    # Second loop: Apply fills and images
    with pd.ExcelWriter(
        db_pth, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name="Results")
        ws = writer.sheets["Results"]
        for idx in range(1, ws.max_column + 1):
            letter = get_column_letter(idx)
            ws.column_dimensions[letter].width = 12

        for patient_csv, patient_id, patient_map in zip(
            patient_csvs, patient_ids, patient_maps
        ):
            patient_id_int = int(patient_id)
            load_this_gt = load_gt_range[0] <= patient_id_int <= load_gt_range[1]
            if load_this_gt:
                segs = dict()
                for prefix in prefixes:
                    segs[prefix] = PILImage.open(
                        seg_pth / (prefix + "_" + patient_id + ".png")
                    )

            for fl_idx, patient_row_idx in patient_map:
                if patient_row_idx is None:
                    continue
                row = patient_csv.iloc[patient_row_idx]
                col = multi_cols[fl_idx]
                excel_col = df.columns.get_loc(col) + INDEX_COLS + 1
                field_label = "_".join(col).replace(" ", "_")

                for metric in ("raw_score", "test_info", "comments"):
                    if not row[metric]:
                        continue
                    pred_label = metric + "_pred"
                    conf_label = metric + "_conf"
                    excel_data_row = (
                        df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 2
                    )
                    conf_val = float(row[conf_label]) if row[conf_label] != "" else 1.0
                    fill = (
                        red_fill
                        if conf_val < 0.7
                        else yellow_fill if conf_val < 0.9 else white_fill
                    )
                    ws.cell(row=excel_data_row, column=excel_col).fill = fill

                    if not load_this_gt:
                        continue
                    excel_gt_row = (
                        df.index.get_loc((patient_id_int, pred_label)) + HEADER_ROWS + 3
                    )
                    cn_x1, cn_y1, cn_x2, cn_y2 = (
                        metric + "_x1",
                        metric + "_y1",
                        metric + "_x2",
                        metric + "_y2",
                    )
                    segname = row["head"][:-1]
                    cropbox = (
                        float(row[cn_x1]) if cn_x1 in row and row[cn_x1] else 0,
                        float(row[cn_y1]) if cn_y1 in row and row[cn_y1] else 0,
                        float(row[cn_x2]) if cn_x2 in row and row[cn_x2] else 0,
                        float(row[cn_y2]) if cn_y2 in row and row[cn_y2] else 0,
                    )
                    loading_debugstream += f"Patient {patient_id}, Metric {metric}, Field {col}, Cropbox {cropbox}, Segname {segname}\n"
                    if cropbox != (0, 0, 0, 0):
                        cropimg = segs[segname].crop(cropbox)
                        cropimg = resize_image_to_cell(
                            cropimg, cell_width_pixels=96, cell_height_pixels=66.65 / 3
                        )
                        temp_image_path = (
                            f"cropped_img_{patient_id}_{metric}_{field_label}.jpg"
                        )
                        cropimg.save(temp_image_path)
                        cropimg_excel = OpenpyxlImage(temp_image_path)
                        excel_col_letter = get_column_letter(excel_col)
                        cropimg_excel.anchor = f"{excel_col_letter}{excel_gt_row}"
                        ws.add_image(cropimg_excel)
                        temp_files.append(temp_image_path)

    # send into debug
    with open(debug_pth, "a") as debug_file:
        debug_file.write(loading_debugstream)

    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except FileNotFoundError:
            pass


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

    if not load_gt:
        load_gt_range = [0, -1]
    else:
        load_gt_range = [0, 9]

    with open(template_pth) as f:
        prefixes = json.load(f)["sections"].keys()

    package(source_dir, db_pth, seg_pth, debug_pth, fl_pth, prefixes, load_gt_range)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
