import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# Sample data loading function (replace this with actual data loading)
def load_data():
    data = pd.read_csv("patient_data.csv")
    return data

def calculate_weekly_averages(data, patient_id):
    print(patient_id)
    patient_data = data[data['patientid'] == patient_id].copy()  # Use .copy() to avoid SettingWithCopyWarning
    if len(patient_data) < 2:
        print(f"Patient {patient_id} does not have enough data.")
        return pd.DataFrame()  # Return an empty DataFrame if not enough data
    
    patient_data['measurements_timestamp'] = pd.to_datetime(patient_data['measurements_timestamp'])
    patient_data = patient_data.sort_values(by='measurements_timestamp')
    
    baseline_time = patient_data['measurements_timestamp'].iloc[0]
    start_of_week_1 = patient_data['measurements_timestamp'].iloc[1]
    
    # Calculate weeks starting from the second reading
    patient_data['week'] = ((patient_data['measurements_timestamp'] - start_of_week_1) / timedelta(days=7)).astype(int) + 1
    patient_data.loc[patient_data['measurements_timestamp'] == baseline_time, 'week'] = 0
    
    weekly_averages = patient_data.groupby('week')['measurements_systolicbloodpressure_value'].mean().reset_index()
    return weekly_averages


def plot_data(weekly_averages, patient_id):
    if weekly_averages.empty:
        return None  # If no data, return None
    
    weeks = np.arange(15)
    values = [np.nan] * 15

    for _, row in weekly_averages.iterrows():
        if row['week'] < 15:
            values[int(row['week'])] = row['measurements_systolicbloodpressure_value']

    last_non_nan_index = next((i for i in reversed(range(len(values))) if not np.isnan(values[i])), None)

    if last_non_nan_index is not None and last_non_nan_index > 1:
        if not np.isnan(values[last_non_nan_index - 1]):
            values[last_non_nan_index] = np.mean([values[last_non_nan_index], values[last_non_nan_index - 1]])
            values[last_non_nan_index - 1] = np.nan

    mask = ~np.isnan(values)
    filtered_weeks = weeks[mask]
    filtered_values = np.array(values)[mask]

    fig, ax = plt.subplots(figsize=(4.5, 4))

    # Plotting the data curve
    ax.plot(filtered_weeks, filtered_values, marker='o', color='#2a5674', linestyle='-', linewidth=1.5, markersize=3, zorder=10)
    print("id", patient_id)
    print("values", filtered_values)

    # Shade the area between 110 and 130 in light green (Goal Range)
    ax.axhspan(110, 130, color='lightgreen', alpha=0.3, zorder=1)

    # Updated special lines
    special_lines = [
        (170, 'Very High', 'darkred', 'dotted', 2.0),
        (140, 'High', '#FF8C00', 'dotted', 1.8),
        (110, 'Low', 'orange', 'dotted', 1.8)
    ]

    for y, label, color, linestyle, linewidth in special_lines:
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)
        if label == 'Low':
            ax.text(14.5, y - 2, label, fontsize=8, verticalalignment='top', horizontalalignment='right', color=color)
        else:
            ax.text(14.5, y + 2, label, fontsize=8, verticalalignment='bottom', horizontalalignment='right', color=color)

    # Add a "Goal Range" label
    ax.text(14.5, 130, "Goal", fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='green')

    # Highlighting the data points
    for i, val in enumerate(filtered_values):
        size = 4 if i != last_non_nan_index else 7  # Making the last data point bigger
        ax.plot(filtered_weeks[i], val, marker='o', color='#2a5674', markersize=size, zorder=11)

    # Customizing axes and grid
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(70, 220)
    ax.set_xticks(weeks)
    # Change the last week's label to "Last"
    x_labels = ['Starting'] + [str(i) for i in range(1, 14)] + ['Last']
    ax.set_xticklabels(x_labels, fontsize=6, color='#505050')

    ax.tick_params(axis='y', labelsize=7, colors='#505050')
    ax.set_xlabel('Weeks', fontsize=8, color='#505050')
    ax.set_ylabel('Systolic Blood Pressure\n(mmHg)', fontsize=8, color='#505050')
    ax.set_title('Your Blood Pressure Trend', fontsize=10)
    ax.grid(axis='y', which='major', color='lightgrey', linestyle='--', linewidth=0.5)

    # Adding grey lines every 10 units
    for y in range(80, 220, 10):
        ax.axhline(y=y, color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)

    # Adjusting aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#505050')
    ax.spines['bottom'].set_color('#505050')

    # Save plot
    plot_filename = f'patient_{patient_id}_plot.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight', transparent=True)

    return plot_filename


# Define the scale function based on the given points
def calculate_x(value, x1=120, x2=130, y1=281, y2=339):
    return y1 + (value - x1) * (y2 - y1) / (x2 - x1)
    
# Function to create the infographic
def create_infographic(plot_filename, baseline_value, most_recent_value, patient_id):
    bg_image = Image.open('infographics_template.png')
    plot_image = Image.open(plot_filename)
    
    bg_image.paste(plot_image, (int(0.07 * bg_image.width), int(0.130 * bg_image.height)), plot_image)
    
    draw = ImageDraw.Draw(bg_image)
    font = ImageFont.truetype("arial.ttf", 17)
    
    baseline_x = int(calculate_x(baseline_value))
    most_recent_x = int(calculate_x(most_recent_value))
    
    arrow_color = "black"
    arrow_size = 40
    arrow_y_position = int(0.735 * bg_image.height)
    arrow_y_position_baseline = int(0.78 * bg_image.height)
    
    arrow_head_size = 8
    draw.polygon([(baseline_x, arrow_y_position_baseline), (baseline_x - arrow_head_size, arrow_y_position_baseline + arrow_head_size), (baseline_x + arrow_head_size, arrow_y_position_baseline + arrow_head_size)], fill=arrow_color)
    draw.line([(baseline_x, arrow_y_position_baseline + arrow_head_size), (baseline_x, arrow_y_position_baseline + arrow_size)], fill=arrow_color, width=2)
    
    draw.polygon([(most_recent_x, arrow_y_position), (most_recent_x - arrow_head_size, arrow_y_position - arrow_head_size), (most_recent_x + arrow_head_size, arrow_y_position - arrow_head_size)], fill=arrow_color)
    draw.line([(most_recent_x, arrow_y_position - arrow_head_size), (most_recent_x, arrow_y_position - arrow_size)], fill=arrow_color, width=2)
    
    font_bp = ImageFont.truetype("arial.ttf", 40)
    font_progress = ImageFont.truetype("arial.ttf", 20)
    
    # Draw the labels with a larger font size
    draw.text((baseline_x - 0.8 * arrow_size, arrow_y_position_baseline + 1.1 * arrow_size), "Baseline", fill=arrow_color, font=font)
    draw.text((most_recent_x - 1.2 *arrow_size, arrow_y_position - 1.65 * arrow_size), "Most Recent", fill=arrow_color, font=font)
    
    text_x_position = int(0.62 * bg_image.width)
    baseline_text_y = int(0.68 * bg_image.height)
    most_recent_text_y = int(0.765 * bg_image.height)
    progress_text_y = int(0.85 * bg_image.height)
    

    ### Blood pressure: high, low, goal range, too high
    ### Progress: make progress, did not make progress
    ### One message or two messages. 
    ### Dangerously high
    if 120 <= most_recent_value <= 130:
        progress_message = "Great job! Your blood pressure is in the goal range!"
    elif most_recent_value > baseline_value + 10:
        progress_message = "Your blood pressure is significantly above the goal."
    elif most_recent_value > baseline_value + 5:
        progress_message = "Your blood pressure is above the goal."
    elif most_recent_value < baseline_value - 10:
        progress_message = "Excellent job! Your blood pressure has improved significantly."
    elif most_recent_value < baseline_value - 5:
        progress_message = "Good job! Your blood pressure has improved."
    else:
        if most_recent_value > 130 and baseline_value > 130:
            progress_message = "Your blood pressure is stable but above the goal."
        elif most_recent_value < 120 and baseline_value < 120:
            progress_message = "Your blood pressure is stable but below the goal."
        else:
            progress_message = "Your blood pressure is stable."

    
    draw.text((text_x_position, baseline_text_y), f"{int(baseline_value)}", fill=arrow_color, font=font_bp)
    draw.text((text_x_position, most_recent_text_y), f"{int(most_recent_value)}", fill=arrow_color, font=font_bp)
    # Splitting the progress message into multiple lines
    lines = progress_message.split(' ')
    max_line_length = 35  # Maximum number of characters per line
    wrapped_lines = []
    current_line = ""
    
    for word in lines:
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += f" {word}"
        else:
            wrapped_lines.append(current_line.strip())
            current_line = word
    
    if current_line:
        wrapped_lines.append(current_line.strip())
    
    for i, line in enumerate(wrapped_lines):
        draw.text((text_x_position, progress_text_y + i * (font_progress.size + 5)), line, fill=arrow_color, font=font_progress)
 
    
    infographic_filename = f'patient_{patient_id}_infographic.png'
    bg_image.save(infographic_filename)

# Main function to process all patients and create infographics
def main():
    data = load_data()
    print(data['patientid'].unique())
    
    for patient_id in data['patientid'].unique():
        if pd.notna(patient_id):
            weekly_averages = calculate_weekly_averages(data, patient_id)
            if not weekly_averages.empty:  # Check if the DataFrame is not empty
                plot_filename = plot_data(weekly_averages, patient_id)
                
                if plot_filename:  # Check if plot was created
                    baseline_value = weekly_averages['measurements_systolicbloodpressure_value'].iloc[0]
                    most_recent_value = weekly_averages['measurements_systolicbloodpressure_value'].iloc[-1]
                    
                    create_infographic(plot_filename, baseline_value, most_recent_value, patient_id)

if __name__ == "__main__":
    main()
