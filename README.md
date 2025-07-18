# ANPRE
Project to Automate NeuroPsych Record Entry. We aim to utilize AI to engineer a system for automated processing of neuropsych patient evaluation scans into spreadsheets to allow for statistical analysis and further research.

# Instructions

Below are instructions for installing and using the package as a user and developer.

## User Instructions

User should create the default output directories utilized in each processing stage.

`$ mkdir case_outputs case_outputs case_outputs/sheets case_outputs/seg case_outputs/tables case_outputs/tables_clean`

First, to select clinical summary sheet scans from full case files, run `select_sheets.py`:

`$ python -m src.select_sheets -i data/full -o case_outputs/sheets -xl case_outputs/patient_info.csv`

Next, to rectify and segment the scans into proper sections, run `segment_sheets.py`:

`$ python -m src.segment_sheets -i case_outputs/sheets -o case_outputs/seg -t instructions/seg_template.json`

In order to scan sheets and enter data into CSV tables, run `scan_sheets.py`:

`$ python -m src.scan_sheets -i case_outputs/seg -o case_outputs/tables`

To process and clean CSV table data, run `postprocessing.py`:

`$ python -m src.postprocessing -i case_outputs/tables -o case_outputs/tables_clean`

Finally, to align and package data to export to XLSX, run `package_data.py`:

`$ python -m src.package_data -i case_outputs/tables_clean -o case_outputs/final_data.xlsx`

## Developer Instructions

For development, clone the repository and install the developer requirements in a virtual environment. Development allows for training of new models using the scripts directory.

`$ pip install -r requirements_dev.txt`

Install pre-commit hooks

`$ pre-commit install`

Run pre-commit hooks to ensure code quality

`$ pre-commit run -a`

# Authors

1. **Michal Brzus**

    github: [mbrzus](https://github.com/mbrzus), email: michal-brzus@uiowa.edu

2. **Ashby Martin**

    github: [Ashby-Martin](https://github.com/Ashby-Martin), email: ashby-martin@uiowa.edu

3. **Zihe (Lucas)  Wang**

    github: [Brocoliman](https://github.com/Brocoliman), email: zihe-wang@uiowa.edu
