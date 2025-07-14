import os
import argparse
import time
from table_ocr import scan, get_table, initialize_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Processing tool to extract, anonymize, and then segment patient case sheets for data processing and recording"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="case_outputs/one",
        help="Path to folder with list of segmented sheets in PNG",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/tables",
        help="Path to folder with list of csv tables for patient information",
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
        "--parallel",
        "--p",
        action=argparse.BooleanOptionalAction,
        help="Whether to parallelize OCR processing (on CPU, it seems not to matter)",
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
# Detect text
##############################


def detect(
    source_dir: str,
    output_dir: str,
    debug_dir: str,
    xl_pth: str,
    parallel=False,
):
    """
    Creates case_outputs/tables/000x.csv of patient records
    """
    if parallel:
        # ---------- batch all files in one go ----------
        # build list of all image paths
        files = [
            os.path.join(source_dir, fn)
            for fn in sorted(os.listdir(source_dir))
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # run OCR on the entire batch
        outputs_list = scan(files, debug_dir)

        # one table per file
        for file_path, page_result in zip(files, outputs_list):
            # wrap single-page result in a list to match get_table signature
            table = get_table(file_path, [page_result])

            output_file = os.path.join(
                output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".csv"
            )
            table.to_csv(output_file, index=False)
            print(f"[EVENT] Saved table for {os.path.basename(file_path)}")
    else:
        for fn in os.listdir(source_dir):
            file = os.path.join(source_dir, fn)
            outputs = scan(file, debug_dir)
            tables = {file: get_table(file, outputs)}

            for path, table in tables.items():
                output_file = os.path.join(
                    output_dir, os.path.splitext(os.path.basename(path))[0] + ".csv"
                )
                table.to_csv(output_file, index=False)

            print(f"[EVENT] Tables saved to {output_dir}")


##############################
# Main: Runs all functions from source data to generating red and yellow sheets
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    xl_pth = args.spreadsheet
    use_parallel = args.parallel
    use_gpu = args.gpu

    initialize_pipeline(use_gpu)

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    detect(source_dir, output_dir, debug_dir, xl_pth, use_parallel)

    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
