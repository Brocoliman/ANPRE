from pathlib import Path
import re
import pandas as pd
import enchant


class RowWiseRule:
    """
    Base class for all cleaning rules. Applies a rule to each row matching the domain.
    """

    def __init__(self, domain: list[str], col: int):
        self.domain = domain  # keyw for rows that should be included in a rule
        self.col = col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Iterate with index and row Series
        for idx, row in df.iterrows():
            # 1) domain check (on "Left")
            left_val = str(row.get("Left", "")).lower()
            if not (
                self.domain is ...
                or any(re.search(item, left_val) for item in self.domain)
            ):
                continue

            # 2) skip if the target cell is actually NaN
            if pd.isna(row.iloc[self.col]):
                continue

            # 3) now we know it's non-null and matches domain
            new_row = self._row_apply(row.copy())
            df.loc[idx] = new_row
        return df

    def _row_apply(self, row: pd.Series) -> pd.Series:
        raise NotImplementedError("Subclasses must implement _row_apply()")


class AddSpaces(RowWiseRule):
    """
    Add spaces between parentheses and outer words
    """

    def __init__(self, domain: list[str], col: int):
        super().__init__(domain, col)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        # Extract original text from target column
        text = str(row.iloc[self.col])

        # only if preceding char is [A-Za-z0-9]
        text = re.sub(r"(?<=[A-Za-z0-9])\(", r" (", text)
        # only if following char is [A-Za-z0-9]
        text = re.sub(r"\)(?=[A-Za-z0-9])", r") ", text)

        row.iloc[self.col] = text
        return row


class RemoveSelectSymbols(RowWiseRule):
    """
    Remove certain symbols
    """

    def __init__(self, domain: list[str], col: int, rm: list[str]):
        super().__init__(domain, col)
        self.rm = {ord(c): None for c in rm}

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])
        text = text.translate(self.rm)
        row.iloc[self.col] = text
        return row


class RemoveLogographics(RowWiseRule):
    """
    Remove chinese lettering
    """

    def __init__(self, domain: list[str], col: int):
        super().__init__(domain, col)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])
        logographic = r"[\u4E00-\u9FFF]+"
        text = re.sub(logographic, "", text)
        row.iloc[self.col] = text
        return row


class BiasedSpellCheck(RowWiseRule):
    """
    Spellcheck on a specific column, biased towards favorites.
    Prioritizes a certain list of words (favorites)
    Splits tokens by spaces or (|)
    Spell checks words with only letters or certain include characters (and disregards those characters)
    """

    def __init__(
        self,
        domain: list[str],
        col: int,
        favs: set[str],
        delimiters: str = "([|\\s]+)",
        include: set[str] = [],
        lang: str = "en_US",
    ):
        super().__init__(domain, col)
        self.favs = set(
            favs
        )  # words to prioritize in spell correct (standard test names etc.)
        self.delimiters = delimiters  # regex of delimiters to split token process
        self.rm_include = {
            ord(c): None for c in include
        }  # normally, only purely alpha words are spell-checked, but words with these only these additional characters are considered
        self.d = enchant.Dict(lang)

        for fav in self.favs:
            self.d.add_to_session(fav)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        # 1)
        value = row.iloc[self.col]

        # 2) work with the string
        text = str(value)
        parts = re.split(self.delimiters, text)  # Keep delimiters by capturing them

        for i in range(0, len(parts), 2):  # Only even indices are actual tokens
            token = parts[i]

            # 3) skip empty, purely non-alpha, or the literal "nan"
            if not token or not token.translate(self.rm_include).isalpha():
                continue

            # 4) spell check
            if not self.d.check(token.translate(self.rm_include)):
                suggestions = self.d.suggest(token.translate(self.rm_include))
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
