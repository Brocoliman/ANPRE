import os
import argparse
import time
import json
from pathlib import Path

from utils.rulesets import (
    Cleaner,
    AddSpaces,
    BiasedSpellCheck,
    Remove,
    RemoveLogographicUnicodes,
    SlashScores,
    Substitutions,
    ManageWrappingParentheses,
    CleanAlphaNumericalUnicodes,
)
from utils.diff import diff_folder
from utils.flatten_labels import flatten_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raw Patient Tables, Ruleset, Field Labels \u2192 Cleaned Patient Tables"
    )
    parser.add_argument(
        "-input",
        "-src",
        "-i",
        type=str,
        default="case_outputs/tables",
        help="Path to folder with list of raw patient csv tables",
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
        "-fields",
        "-f",
        type=str,
        default="instructions/field_labels.json",
        help="Path to CSV file with all test names in tree structure",
    )
    parser.add_argument(
        "-patient_info",
        "-pi",
        type=str,
        default="case_outputs/patient_info.csv",
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


def postprocess(
    source_dir: str, output_dir: str, debug_dir: str, rules_pth: str, fl_pth: str
):
    # Open ruleset
    with open(rules_pth) as f:
        rules = json.load(f)

    # Add all known field labels to biased spell check
    fl_pth = Path(fl_pth)
    fl_flat = flatten_labels(json.loads(fl_pth.read_text()), blanks=False)
    fls = [item[-1] for item in fl_flat]  # take the most specific field label
    fls = [fl_token for fl in fls for fl_token in fl.split()]
    fls = (
        [fl.upper() for fl in fls]
        + [fl.lower() for fl in fls]
        + [fl.capitalize() for fl in fls]
    )

    cleaner = Cleaner(
        [
            Remove(..., 1, rules["RM"]["BLUE"]),
            Remove(..., 3, rules["RM"]["GREEN"]),
            RemoveLogographicUnicodes(..., 1),
            RemoveLogographicUnicodes(..., 2),
            RemoveLogographicUnicodes(..., 3),
            ManageWrappingParentheses(..., 1),
            CleanAlphaNumericalUnicodes(..., 1),
            CleanAlphaNumericalUnicodes(..., 2),
            CleanAlphaNumericalUnicodes(..., 3),
            Substitutions(..., 1, rules["SUBS"]["BLUE"]),
            AddSpaces(..., 1),
            BiasedSpellCheck(  # blue
                ...,
                1,
                rules["SPELLCHECK"]["BLUE"] + rules["SPELLCHECK"]["GREEN"] + fls,
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
    fl_pth = args.fields
    pi_pth = args.patient_info

    clear_pdfdir(output_dir)
    clear_pdfdir(debug_dir)

    postprocess(source_dir, output_dir, debug_dir, rules_pth, fl_pth)
    print(f"[REPORT] Total time taken: {time.perf_counter()-start}")
