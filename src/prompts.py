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
