#!/usr/bin/env python3
import csv
import difflib
from rich.console import Console
from rich.syntax import Syntax


def load_and_normalize_csv(path):
    """
    Read a CSV into memory, sort its columns, and
    return a list of text lines (with newline chars).
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # sort the headers for consistent ordering
        headers = sorted(reader.fieldnames or [])
        lines = []
        # header row
        lines.append(",".join(headers) + "\n")
        for row in reader:
            # extract values in header order
            vals = [row.get(h, "") for h in headers]

            # quote any fields containing commas or quotes
            def quote(v):
                if any(c in v for c in [",", '"', "\n"]):
                    v = v.replace('"', '""')
                    return f'"{v}"'
                return v

            lines.append(",".join(quote(v) for v in vals) + "\n")
        return lines


def diff_csvs(file_old, file_new):
    a = load_and_normalize_csv(file_old)
    b = load_and_normalize_csv(file_new)
    return difflib.unified_diff(
        a, b, fromfile=file_old, tofile=file_new, lineterm=""  # avoid extra blank lines
    )


def main():
    n = 0
    f1 = f"case_outputs/tables/{n:05d}.csv"
    f2 = f"case_outputs/tables_clean/{n:05d}.csv"
    old_csv, new_csv = f1, f2
    diff_lines = list(diff_csvs(old_csv, new_csv))
    if not diff_lines:
        print("No differences found.")
        return

    # join and syntax-highlight as a "diff"
    diff_text = "\n".join(diff_lines)
    console = Console()
    syntax = Syntax(diff_text, "diff", line_numbers=False)
    console.print(syntax)


if __name__ == "__main__":
    main()
