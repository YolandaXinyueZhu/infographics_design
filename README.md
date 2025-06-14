# Blood Pressure Infographic Generator

This project processes patient blood pressure data and produces weekly trend plots and infographics for each patient.

## Requirements

* Python 3.7 or higher
* The following Python packages:

  * pandas
  * numpy
  * matplotlib
  * pillow (PIL)

You can install dependencies with:

```bash
pip install pandas numpy matplotlib pillow
```

## Files

* `generate_infographics.py` (or rename your main script to this name)
* `patient_data.csv` – input CSV with patient records
* `infographics_template.png` – background template for infographic
* `arial.ttf` – font file used for text annotations
* `README.md` – this file

## Usage

1. Ensure `patient_data.csv`, `infographics_template.png`, and `arial.ttf` are in the project directory.
2. Run the script:

   ```bash
   python3 generate_infographics.py
   ```

   This will:

   * Load `patient_data.csv`
   * Compute weekly average systolic blood pressure per patient
   * Generate a line plot for each patient (`patient_<id>_plot.png`)

## Updating Patient Data

To update to the latest data:

1. Log in to ITASC and download the newest patient CSV export.
2. Replace the existing `patient_data.csv` in this directory with the downloaded file.
3. Rerun the script:

   ```bash
   python3 generate_infographics.py
   ```

Your new infographics will be generated using the updated data.

---

Feel free to customize the template or font paths as needed. If you run into issues, check that all required files are present and that your CSV columns match the expected names (`patientid`, `measurements_timestamp`, `measurements_systolicbloodpressure_value`).
