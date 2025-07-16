import os
import csv
import difflib


def load_and_normalize_csv(path):
    """
    Read a CSV into memory, sort its columns, and
    return a list of text lines (with newline chars).
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = sorted(reader.fieldnames or [])
        lines = [",".join(headers) + "\n"]
        for row in reader:
            vals = [row.get(h, "") for h in headers]

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
        a,
        b,
        fromfile=os.path.basename(file_old),
        tofile=os.path.basename(file_new),
        lineterm="",
    )


def diff_folder(source_dir, output_dir, debug_dir):
    hd = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    for fn in os.listdir(source_dir):
        if not fn.lower().endswith(".csv"):
            continue

        old = load_and_normalize_csv(os.path.join(source_dir, fn))
        new = load_and_normalize_csv(os.path.join(output_dir, fn))
        if old == new:
            continue

        # HtmlDiff wants lists of *strings* without trailing newlines
        old = [line.rstrip("\n") for line in old]
        new = [line.rstrip("\n") for line in new]

        html = hd.make_file(
            old,
            new,
            fromdesc=f"OLD — {fn}",
            todesc=f"NEW — {fn}",
            context=True,  # show only changed blocks
            numlines=3,  # lines of context
        )

        out_path = os.path.join(debug_dir, fn.replace(".csv", ".html"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
