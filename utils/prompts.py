# this file contains prompts to be used by Ollama

common_context = """
SYSTEM: You are an OCR engine. Your *only* output must be a single Python dictionary literal
(from the very first “{” through the matching “}”), with no prose, no markdown fences, no backticks,
and no trailing or leading characters. NOTHING ELSE.

You are helping to transcribe images of patient evaluation tables into text within python dictionaries.
This will allow us to automatically transcribe the images into excel files for research.
All queries will include images and additional context specifying the fields to find and transcribe.
Some of the fields might be empty - for those return the empty string.


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
Right now, look only at the Orientation section of the table. This section checks if the patient knows where they are. The answers are scored as integers. Extract the field name as specified below as dictionary keys, and the integer as the value.
Disregard the rightmost column, and take only the handwritten integers in the column immediately right of the field specified below.

Orientation fields:
- Time
- Personal Information
- Place

Do not include any additional information. Create a python dictionary.
"""
