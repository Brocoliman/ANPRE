import os
import argparse
import time
from utils.rulesets import (
    Cleaner,
    AddSpaces,
    BiasedSpellCheck,
    RemoveSelectSymbols,
    RemoveLogographics,
    SlashScores,
    Substitutions,
    RemoveParentheses,
)
from utils.diff import diff_folder
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data collection tool for patient neuropsychological assessment scores"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="case_outputs/tables",
        help="Path to folder with list of CSV tables for patient information",
    )
    parser.add_argument(
        "-output",
        "-o",
        type=str,
        default="case_outputs/tables_clean",
        help="Path to folder with list of cleaned CSV tables ",
    )
    parser.add_argument(
        "-debug",
        "-d",
        type=str,
        default="debug/tables_clean",
        help="Path to folder with debug for showing diff in table cleaning",
    )
    parser.add_argument(
        "-rules",
        "-r",
        type=str,
        default="instructions/rules.json",
        help="Path to JSON with rules ",
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
# Postprocess
##############################


def postprocess(source_dir: str, output_dir: str, debug_dir: str, rules_pth: str):
    # Open ruleset
    with open(rules_pth) as f:
        rules = json.load(f)

    cleaner = Cleaner(
        [
            RemoveSelectSymbols(..., 1, rules["RM"]["BLUE"]),
            RemoveSelectSymbols(..., 3, rules["RM"]["GREEN"]),
            RemoveLogographics(..., 1),
            RemoveLogographics(..., 2),
            RemoveLogographics(..., 3),
            RemoveParentheses(..., 1),
            Substitutions(..., 1, rules["SUBS"]["BLUE"]),
            AddSpaces(..., 1),
            BiasedSpellCheck(  # blue
                ...,
                1,
                rules["SPELLCHECK"]["BLUE"] + rules["SPELLCHECK"]["GREEN"],
                rules["SPELLCHECK"]["WORD_DELIMITERS"],
                rules["SPELLCHECK"]["INCLUDE"],
            ),
            BiasedSpellCheck(  # green
                ...,
                3,
                rules["SPELLCHECK"]["BLUE"] + rules["SPELLCHECK"]["GREEN"],
                rules["SPELLCHECK"]["WORD_DELIMITERS"],
                rules["SPELLCHECK"]["INCLUDE"],
            ),
            SlashScores(
                rules["EXPECTED_SLASH_FORMAT"]["YELLOW"],
                2,
                rules["EXPECTED_SLASH_FORMAT"]["SLASH_REGEX"],
                True,
                True,
            ),
        ]
    )

    cleaner.clean(source_dir, output_dir)

    diff_folder(source_dir, output_dir, debug_dir)


##############################
# Main
##############################

if __name__ == "__main__":
    start = time.perf_counter()
    args = parse_args()
    source_dir = args.input
    output_dir = args.output
    debug_dir = args.debug
    rules_pth = args.rules
    xl_pth = args.spreadsheet

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    postprocess(source_dir, output_dir, debug_dir, rules_pth)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
