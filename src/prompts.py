# this file contains prompts to be used by Ollama

common_context = """Table Text Recognition

You are helping to transcribe images of patient evaluation tables into text within python dictionaries.
This will allow us to automatically transcribe the images into excel files for research.
All queries will include images and additional context specifying the fields to find and transcribe.
Some of the fields might be empty - for those return the empty string.

Rules: DO NOT INCLUDE ANY SYSTEM MESSAGES. RETURN ONLY VALID PYTHON DICTIONARIES AS TEXT ONLY.
MUST NOT INCLUDE THINGS LIKE:
"Here is a Python dictionary that matches the specified fields:


```python"

"""

patient_info_prompt = f"""Context: {common_context}

Image includes general patient and study information. Create a python dictionary that contains the following fields:
- Date
- Sex
- Education
- Age
- Handedness

Do not include any additional information
"""

dict_format = str(
    {
        "[Field Name]": {
            "Checked": bool,
            "Response": float,
            "Comments": list[str],
        }
    }
)

patient_full_prompt = f"""Context: {common_context}

Image includes general patient and study information. Create a python dictionary that contains the following fields from the following sections, listed below in bullet points.
For sections that are not part of the header, please provide query results in the format of {dict_format}.

Checked refers to whether the first column in a field row is marked with X.
Response refers to the written numerical value in the third column, while excluding any printed text.
Comments refers to the fourth and final column. Please parse the content in this column as a list of strings, delimited by each subfield in the column.
Additionally, leave items as None if it is not filled out.

Header:
- Date
- Sex

Orientation
- Time
- Personal Information
- Place

Do not include any additional information. Create a python dictionary.
"""
