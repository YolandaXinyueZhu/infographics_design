#!/usr/bin/env python3
"""
Generate weekly blood pressure trend plots and infographics for patients.
"""
import logging
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Constants
DATA_FILE = Path("patient_data.csv")
TEMPLATE_IMAGE = Path("image2.png")
FONT_PATH = "arial.ttf"
PLOT_DPI = 300
PLOT_SIZE = (4.5, 4)
MAX_WEEKS = 15
GOAL_RANGE = (110, 130)
SPECIAL_LINES = [
    (170, "Very High", "darkred", "dotted", 2.0),
    (140, "High", "#FF8C00", "dotted", 1.8),
    (110, "Low", "orange", "dotted", 1.8),
]
ARROW_SIZE = 40
ARROW_HEAD_SIZE = 8


def load_data() -> pd.DataFrame:
    """
    Load patient data from a CSV file.

    Returns:
        pd.DataFrame: Loaded patient data.
    """
    return pd.read_csv(DATA_FILE)


def calculate_weekly_averages(
    data: pd.DataFrame, patient_id
) -> pd.DataFrame:
    """
    Calculate weekly average systolic blood pressure for a patient.

    Args:
        data (pd.DataFrame): Full dataset.
        patient_id: Identifier for the patient.

    Returns:
        pd.DataFrame: Weekly averages with columns ['week', 'measurements_systolicbloodpressure_value'].
    """
    logging.info("Calculating weekly averages for patient %s", patient_id)
    patient_data = data[data['patientid'] == patient_id].copy()

    if len(patient_data) < 2:
        logging.warning(
            "Patient %s does not have enough data.", patient_id
        )
        return pd.DataFrame()

    patient_data['measurements_timestamp'] = pd.to_datetime(
        patient_data['measurements_timestamp']
    )
    patient_data.sort_values(by='measurements_timestamp', inplace=True)

    baseline_time = patient_data['measurements_timestamp'].iloc[0]
    start_of_week_1 = patient_data['measurements_timestamp'].iloc[1]

    # Compute week index starting from second reading
    delta = patient_data['measurements_timestamp'] - start_of_week_1
    patient_data['week'] = (delta / timedelta(days=7)).astype(int) + 1
    patient_data.loc[
        patient_data['measurements_timestamp'] == baseline_time, 'week'
    ] = 0

    weekly_avg = (
        patient_data
        .groupby('week')['measurements_systolicbloodpressure_value']
        .mean()
        .reset_index()
    )
    return weekly_avg


def plot_data(
    weekly_averages: pd.DataFrame, patient_id
) -> Path | None:
    """
    Plot systolic blood pressure trend and save as PNG.

    Args:
        weekly_averages (pd.DataFrame): Weekly averages.
        patient_id: Identifier for the patient.

    Returns:
        Path or None: File path if plot is created, otherwise None.
    """
    if weekly_averages.empty:
        return None

    weeks = np.arange(MAX_WEEKS)
    values = [np.nan] * MAX_WEEKS

    for _, row in weekly_averages.iterrows():
        w = int(row['week'])
        if 0 <= w < MAX_WEEKS:
            values[w] = row['measurements_systolicbloodpressure_value']

    # Smooth last point if needed
    last_idx = next(
        (i for i in reversed(range(len(values))) if not np.isnan(values[i])),
        None
    )
    if last_idx and last_idx > 1 and not np.isnan(values[last_idx - 1]):
        avg = np.mean([values[last_idx], values[last_idx - 1]])
        values[last_idx], values[last_idx - 1] = avg, np.nan

    mask = ~np.isnan(values)
    x = weeks[mask]
    y = np.array(values)[mask]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.plot(
        x, y, marker='o', linestyle='-', linewidth=1.5,
        markersize=3, zorder=10, color='#2a5674'
    )

    # Goal range shading
    ax.axhspan(
        GOAL_RANGE[0], GOAL_RANGE[1], color='lightgreen', alpha=0.3, zorder=1
    )

    # Special lines
    for y_val, label, color, ls, lw in SPECIAL_LINES:
        ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=lw, zorder=2)
        offset = -2 if label == 'Low' else 2
        va = 'top' if label == 'Low' else 'bottom'
        ax.text(
            MAX_WEEKS - 1 + 0.5, y_val + offset,
            label, fontsize=8, va=va, ha='right', color=color
        )

    # Goal label
    ax.text(
        MAX_WEEKS - 1 + 0.5, GOAL_RANGE[1],
        'Goal', fontsize=8, va='bottom', ha='right', color='green'
    )

    # Highlight data points with larger last marker
    for idx, val in zip(x, y):
        size = 7 if idx == last_idx else 4
        ax.plot(idx, val, 'o', markersize=size, zorder=11, color='#2a5674')

    # Axes and grid
    ax.set_xlim(-0.5, MAX_WEEKS - 0.5)
    ax.set_ylim(70, 220)
    ax.set_xticks(weeks)
    labels = ['Starting'] + [str(i) for i in range(1, MAX_WEEKS - 1)] + ['Last']
    ax.set_xticklabels(labels, fontsize=6, color='#505050')
    ax.tick_params(axis='y', labelsize=7, colors='#505050')
    ax.set_xlabel('Weeks', fontsize=8, color='#505050')
    ax.set_ylabel('Systolic Blood Pressure\n(mmHg)', fontsize=8, color='#505050')
    ax.set_title('Your Blood Pressure Trend', fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='lightgrey')

    # Additional horizontal grid lines
    for y_h in range(80, 220, 10):
        ax.axhline(y=y_h, linestyle='--', linewidth=0.5, color='lightgrey', zorder=0)

    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#505050')

    filename = Path(f'patient_{patient_id}_plot.png')
    fig.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight', transparent=True)
    plt.close(fig)
    return filename


def calculate_x(
    value: float, x1: int = 120, x2: int = 130,
    y1: int = 281, y2: int = 339
) -> float:
    """
    Map a blood pressure value to an x-coordinate on the infographic.
    """
    return y1 + (value - x1) * (y2 - y1) / (x2 - x1)



def main() -> None:
    logging.basicConfig(level=logging.INFO)
    data = load_data()
    patient_ids = data['patientid'].unique()
    logging.info("Found patients: %s", patient_ids)

    for pid in patient_ids:
        if pd.isna(pid):
            continue
        weekly_avg = calculate_weekly_averages(data, pid)
        plot_file = plot_data(weekly_avg, pid)


if __name__ == "__main__":
    main()
