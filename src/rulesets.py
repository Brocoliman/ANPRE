from pathlib import Path
import re
import pandas as pd
import enchant


class RowWiseRule:
    """
    Base class for all cleaning rules. Applies a rule to each row matching the domain.
    """

    def __init__(self, domain: list[str]):
        self.domain = domain  # should be lowercase patterns or Ellipsis

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Iterate with index and row Series
        for idx, row in df.iterrows():
            # Check domain: match on 'Left' column by default
            left_val = str(row.get("Left", "")).lower()
            if self.domain is ... or any(
                re.search(item, left_val) for item in self.domain
            ):
                new_row = self._row_apply(row.copy())
                df.loc[idx] = new_row
        return df

    def _row_apply(self, row: pd.Series) -> pd.Series:
        raise NotImplementedError("Subclasses must implement _row_apply()")


class BiasedSpellCheck(RowWiseRule):
    """
    Spellcheck on a specific column, biased towards favorites.
    Splits tokens by '|' or spaces, corrects only pure words.
    """

    def __init__(
        self, domain, col: int, favs: set[str], delimiters: str, lang: str = "en_US"
    ):
        super().__init__(domain)
        self.col = col  # integer index of the column to process
        self.favs = set(favs)
        self.delimiters = delimiters
        self.d = enchant.Dict(lang)
        for fav in self.favs:
            self.d.add_to_session(fav)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        # Extract original text from the target column
        text = str(row.iloc[self.col])
        # Keep delimiters by capturing them
        parts = re.split(r"([|\s]+)", text)
        # Only even indices are actual tokens
        for i in range(0, len(parts), 2):
            token = parts[i]
            # needs to be alpha
            if not token or not token.isalpha():
                continue
            # If not recognized
            if not self.d.check(token):
                suggestions = self.d.suggest(token)
                # Use first favorite if available
                for w in suggestions:
                    if w in self.favs:
                        token = w
                        break
                else:
                    token = suggestions[0] if suggestions else token
                parts[i] = token
        # Reassemble and write back
        row.iloc[self.col] = "".join(parts)
        return row


class Cleaner:
    """
    Pipeline to apply multiple RowWiseRule instances over CSVs in a folder.
    """

    def __init__(self, rules: list[RowWiseRule]):
        self.rules = rules

    def clean(self, source_dir: str, output_dir: str) -> None:
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in source_dir.iterdir():
            if csv_path.suffix.lower() != ".csv":
                continue
            df = pd.read_csv(csv_path)
            for rule in self.rules:
                df = rule.apply(df)
            df.to_csv(output_dir / csv_path.name, index=False)
