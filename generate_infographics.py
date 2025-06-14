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
TEMPLATE_IMAGE = Path("infographics_template.png")
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


def create_infographic(
    plot_path: Path, baseline: float,
    recent: float, patient_id
) -> Path:
    """
    Overlay plot and annotations onto an infographic template.
    """
    bg = Image.open(TEMPLATE_IMAGE)
    plot_img = Image.open(plot_path)

    # Paste the plot
    x_offset = int(0.07 * bg.width)
    y_offset = int(0.130 * bg.height)
    bg.paste(plot_img, (x_offset, y_offset), plot_img)

    draw = ImageDraw.Draw(bg)
    font_small = ImageFont.truetype(FONT_PATH, 17)
    font_large = ImageFont.truetype(FONT_PATH, 40)
    font_progress = ImageFont.truetype(FONT_PATH, 20)

    # Compute arrow positions
    base_x = int(calculate_x(baseline))
    recent_x = int(calculate_x(recent))
    y_base = int(0.78 * bg.height)
    y_recent = int(0.735 * bg.height)

    # Draw baseline arrow
    draw.polygon([
        (base_x, y_base),
        (base_x - ARROW_HEAD_SIZE, y_base + ARROW_HEAD_SIZE),
        (base_x + ARROW_HEAD_SIZE, y_base + ARROW_HEAD_SIZE)
    ], fill="black")
    draw.line(
        [(base_x, y_base + ARROW_HEAD_SIZE),
         (base_x, y_base + ARROW_SIZE)],
        fill="black", width=2
    )

    # Draw recent arrow
    draw.polygon([
        (recent_x, y_recent),
        (recent_x - ARROW_HEAD_SIZE, y_recent - ARROW_HEAD_SIZE),
        (recent_x + ARROW_HEAD_SIZE, y_recent - ARROW_HEAD_SIZE)
    ], fill="black")
    draw.line(
        [(recent_x, y_recent - ARROW_HEAD_SIZE),
         (recent_x, y_recent - ARROW_SIZE)],
        fill="black", width=2
    )

    # Labels
    text_x = int(0.62 * bg.width)
    y_text_base = int(0.68 * bg.height)
    y_text_recent = int(0.765 * bg.height)
    y_text_progress = int(0.85 * bg.height)

    draw.text((text_x - 0.8 * ARROW_SIZE, y_base + 1.1 * ARROW_SIZE),
              "Baseline", font=font_small, fill="black")
    draw.text((text_x - 1.2 * ARROW_SIZE, y_recent - 1.65 * ARROW_SIZE),
              "Most Recent", font=font_small, fill="black")

    draw.text((text_x, y_text_base), f"{int(baseline)}", font=font_large, fill="black")
    draw.text((text_x, y_text_recent), f"{int(recent)}", font=font_large, fill="black")

    # Progress message
    if GOAL_RANGE[0] <= recent <= GOAL_RANGE[1]:
        message = "Great job! Your blood pressure is in the goal range!"
    elif recent > baseline + 10:
        message = "Your blood pressure is significantly above the goal."
    elif recent > baseline + 5:
        message = "Your blood pressure is above the goal."
    elif recent < baseline - 10:
        message = "Excellent job! Your blood pressure has improved significantly."
    elif recent < baseline - 5:
        message = "Good job! Your blood pressure has improved."
    else:
        if recent > GOAL_RANGE[1] and baseline > GOAL_RANGE[1]:
            message = "Your blood pressure is stable but above the goal."
        elif recent < GOAL_RANGE[0] and baseline < GOAL_RANGE[0]:
            message = "Your blood pressure is stable but below the goal."
        else:
            message = "Your blood pressure is stable."

    # Wrap message
    words = message.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= 35:
            current += f" {w}"
        else:
            lines.append(current.strip())
            current = w
    if current:
        lines.append(current.strip())

    for i, line in enumerate(lines):
        draw.text(
            (text_x, y_text_progress + i * (font_progress.size + 5)),
            line, font=font_progress, fill="black"
        )

    output = Path(f"patient_{patient_id}_infographic.png")
    bg.save(output)
    return output


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
        if plot_file:
            baseline = weekly_avg['measurements_systolicbloodpressure_value'].iloc[0]
            recent = weekly_avg['measurements_systolicbloodpressure_value'].iloc[-1]
            create_infographic(plot_file, baseline, recent, pid)


if __name__ == "__main__":
    main()
