from pathlib import Path
import re
import pandas as pd
import enchant


class RowWiseRule:
    """
    Base class for all cleaning rules. Applies a rule to each row matching the domain.

    Args:
        domain (list[str], dict[str, str], or ...): the set of rows that a rule is applied on, ... if all, otherwise row left name will be passed to _row_apply
        col (int): the column number that the rule is applied on
    """

    def __init__(self, domain, col: int):
        self.domain = domain  # keyw for rows that should be included in a rule
        self.col = col

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Iterate with index and row Series
        for idx, row in df.iterrows():
            # 1) skip if the target cell is actually NaN
            if pd.isna(row.iloc[self.col]):
                continue

            # 2) domain check (on "Left")
            left_val = str(row.get("test_info", "")).lower()
            # print(left_val)
            matches = None
            if self.domain is not ...:
                matches = [re.search(item.lower(), left_val) for item in self.domain]
                if not any(matches):
                    continue

            if matches:
                match = next((m for m in matches if m), None).group(
                    0
                )  # should not be more than one match
                new_row = self._row_apply(row.copy(), match)
            else:
                new_row = self._row_apply(row.copy())
            df.loc[idx] = new_row
        return df

    def _row_apply(self, *args) -> pd.Series:
        raise NotImplementedError("Subclasses must implement _row_apply()")


class Substitutions(RowWiseRule):
    """
    Direct substitutions (no conditions)
    For example, ll -> II as the roman numeral

    Args:
        subs (dict[str, str]): dictionary of substitutions, is case sensitive, and must be order-agnostic (e.g., no chain substitutions should be expected to work correctly)
    """

    def __init__(self, domain: ..., col: int, subs: dict[str, str]):
        super().__init__(domain, col)
        self.subs = subs

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])

        keys = sorted(self.subs, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in keys))
        text = pattern.sub(lambda m: self.subs[m.group(0)], text)

        row.iloc[self.col] = text
        return row


class SlashScores(RowWiseRule):
    """
    Logic check to retrieve slashes (that are likely marked as 1)
    Meant for YELLOW column

    Uses domain as the mapping dictionary due to efficient routing

    Args:
        domain (dict[str, str]): based on given row field name, finds the expected slash tokens present, keys MUST be lowercase
        slash_regex (tuple[str, str]): contains 1) regex for chars that would treated as slash, 2) regex for prerequisite of making the slash replacement
        do_logic_check (bool): whether to check if number falls in slash range, if not, will mark
    """

    def __init__(
        self,
        domain: dict[str, str],
        col: int,
        slash_regex: tuple[str, str],
        do_logic_check=True,
        do_logic_correct=True,
    ):
        super().__init__(domain, col)
        self.slash_subs, self.slash_condition = slash_regex
        self.do_logic_check = do_logic_check
        self.do_logic_correct = do_logic_correct

    def _row_apply(self, row: pd.Series, field: str) -> pd.Series:
        text = str(row.iloc[self.col])

        replacements = self.domain[field].split()  # get the slash replacements
        for replacement in replacements:
            slash, max_score = replacement[0], replacement[1:]
            if slash != "/":
                raise ValueError("Slash Domain Value Tokens should being with '/'")
            text = re.sub(
                self.slash_subs + max_score + self.slash_condition,
                "/" + max_score,
                text,
            )

        for m in re.finditer(r"/(\d+)", text):
            slash_pos = m.start()  # index of '/'
            max_score = int(m.group(1))  # number after slash

            # walk left until you hit a space or the start of the string
            start = slash_pos
            while start > 0 and text[start - 1] != " ":
                start -= 1

            score_str = text[start:slash_pos]  # the token before '/'
            if score_str.isdigit():
                score = int(score_str)
                # now you have `score` and `max_score` to do your logic check:
                if self.do_logic_check and score > max_score:
                    # e.g. mark invalid ones with an asterisk
                    text = (
                        text[1 + slash_pos :]
                        if self.do_logic_correct
                        else text[:slash_pos]
                    ) + f"{text[slash_pos:]}"

        row.iloc[self.col] = text
        return row


class AddSpaces(RowWiseRule):
    """
    Add spaces between parentheses and outer words
    """

    def __init__(self, domain: ..., col: int):
        super().__init__(domain, col)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        # Extract original text from target column
        text = str(row.iloc[self.col])

        # only if preceding char is [A-Za-z0-9]
        text = re.sub(r"(?<=[A-Za-z0-9])\(", r" (", text)
        # only if following char is [A-Za-z0-9]
        text = re.sub(r"\)(?=[A-Za-z0-9])", r") ", text)

        text = re.sub(r"(?<!^)\#", r" #", text)
        text = re.sub(r"\#(?!$)", r"# ", text)

        row.iloc[self.col] = text
        return row


class ManageWrappingParentheses(RowWiseRule):
    """
    Remove parentheses that wrap around entire test name
    Meant for blue column only
    """

    def __init__(self, domain: ..., col: int):
        super().__init__(domain, col)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])

        tokens = text.split("|")
        tokens = [
            token[1:].replace(")", "") if token.startswith("(") else token
            for token in tokens
        ]

        text = "|".join(tokens)

        row.iloc[self.col] = text
        return row


class RemoveSelectSymbols(RowWiseRule):
    """
    Remove certain symbols

    Args:
        rm (list[str]): list of characters or substrings that will be removed, is case sensitive
    """

    def __init__(self, domain: ..., col: int, rm: list[str]):
        super().__init__(domain, col)
        self.rm = {ord(c): None for c in rm}

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])
        text = text.translate(self.rm)
        row.iloc[self.col] = text
        return row


class RemoveLogographicUnicodes(RowWiseRule):
    """
    Remove chinese lettering
    """

    def __init__(self, domain: ..., col: int):
        super().__init__(domain, col)

    def _row_apply(self, row: pd.Series) -> pd.Series:
        text = str(row.iloc[self.col])
        logographic = r"[\u4E00-\u9FFF]+"
        text = re.sub(logographic, "", text)
        row.iloc[self.col] = text
        return row


class BiasedSpellCheck(RowWiseRule):
    """
    Spellcheck on a specific column, biased towards favorites. It prioritizes a certain list of words (favorites)
    Tokens are first split by spaces or (|), and spell checks words with only letters or certain include characters (and disregards those characters when creating spellcheck suggestions)

    Args:
        favs (set[str]): set of words that would be prioritized during spellcheck substitution, is case sensitive
        delimiters (str): regex for spaces or (|), that the row would be split into tokens by
        include (set[str]): these characters will be excluded when checking that a token is only alpha, and also excluded during spellcheck suggestions
        lang (str): language for spellcheck
    """

    def __init__(
        self,
        domain: ...,
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
        for csv_path in sorted(source_dir.iterdir()):
            print(f"[Event] Cleaning {csv_path}")
            if csv_path.suffix.lower() != ".csv":
                continue
            df = pd.read_csv(csv_path)
            for rule in self.rules:
                df = rule.apply(df)
            df.to_csv(output_dir / csv_path.name, index=False)
