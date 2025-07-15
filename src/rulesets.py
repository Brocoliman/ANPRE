import pandas as pd
from typing import List


class Rule:
    """
    Base class for all cleaning rules. Each rule implements the `apply` method,
    which takes a DataFrame and returns a cleaned DataFrame.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("CleaningRule subclasses must implement apply()")


class TrimWhitespaceRule(Rule):
    """
    Strip leading/trailing whitespace from all string columns.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).str.strip()
        return df


class FixCommonOCRErrorsRule(Rule):
    """
    Correct common OCR mis-recognitions in specified columns.

    `corrections` is a dict mapping column names to lists of
    (pattern, replacement) tuples.
    """

    def __init__(self, corrections: dict):
        self.corrections = corrections

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, rules in self.corrections.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                for pattern, repl in rules:
                    df[col] = df[col].str.replace(pattern, repl, regex=True)
        return df


class DropEmptyRowsRule(Rule):
    """
    Drop rows that are fully empty or contain only NaNs/empty strings.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace empty strings with NaN then drop rows where all are NaN
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        return df.dropna(how="all")


class ConvertNumericRule(Rule):
    """
    Convert specified columns to numeric dtype, coercing errors to NaN.
    """

    def __init__(self, numeric_columns: List[str]):
        self.numeric_columns = numeric_columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class UnifyDateFormatRule(Rule):
    """
    Parse dates in specified columns to a uniform YYYY-MM-DD format.
    """

    def __init__(self, date_columns: List[str], dayfirst: bool = True):
        self.date_columns = date_columns
        self.dayfirst = dayfirst

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col], errors="coerce", dayfirst=self.dayfirst
                )
                df[col] = df[col].dt.strftime("%Y-%m-%d")
        return df


class MergeSplitRowsRule(Rule):
    """
    Merge rows where a record has been split across multiple lines.
    This example assumes `id_column` should be non-null for the first part,
    and blank for continuation lines.
    """

    def __init__(self, id_column: str):
        self.id_column = id_column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = []
        buffer = None
        for idx, row in df.iterrows():
            if pd.notna(row.get(self.id_column)) and str(row[self.id_column]).strip():
                if buffer is not None:
                    merged.append(buffer)
                buffer = row.copy()
            else:
                # continuation row: append non-null fields to buffer
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        buffer[col] = f"{buffer[col]} {row[col]}".strip()
        if buffer is not None:
            merged.append(buffer)
        return pd.DataFrame(merged)


class OCRCSVCleaner:
    """
    Pipeline to apply a sequence of cleaning rules to an OCR-extracted CSV.
    """

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def clean(self, input_csv: str, output_csv: str) -> None:
        df = pd.read_csv(input_csv)
        for rule in self.rules:
            df = rule.apply(df)
        df.to_csv(output_csv, index=False)


# Example usage:
if __name__ == "__main__":
    corrections = {
        "Score": [(r"O", "0"), (r"I", "1"), (r"S", "5")],
        "ID": [(r"[^0-9]", "")],
    }
    rules = [
        TrimWhitespaceRule(),
        FixCommonOCRErrorsRule(corrections),
        DropEmptyRowsRule(),
        ConvertNumericRule(["Score", "ID"]),
        UnifyDateFormatRule(["Date"]),
        MergeSplitRowsRule("ID"),
    ]
    cleaner = OCRCSVCleaner(rules)
    cleaner.clean("raw_ocr.csv", "cleaned.csv")
    print("Cleaning complete. Output saved to cleaned.csv.")
