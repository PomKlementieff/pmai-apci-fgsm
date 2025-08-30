#!/bin/bash

# Define models
SUBSTITUTE_MODEL="Inc-v3"
TARGET_MODELS=("Inc-v3" "Xce" "IncRes-v2" "Res-101" "Inc-v3_ens3" "Inc-v3_ens4" "IncRes-v2_ens")
ATTACK="hybrid_fgsm"
CASES=("dim" "tim" "sim" "pim" "dim_tim" "dim_sim" "tim_sim" "dim_tim_sim")

OUTPUT_FILE="hybrid_attack_results.csv"

# Create CSV header
echo -n "Case" > $OUTPUT_FILE
for target_model in "${TARGET_MODELS[@]}"; do
    echo -n ",$target_model" >> $OUTPUT_FILE
done
echo ",Average" >> $OUTPUT_FILE

TOTAL_COMBINATIONS=${#CASES[@]}
CASE_INDEX=0
START_TIME=$(date +%s)

for case in "${CASES[@]}"; do
    CASE_INDEX=$((CASE_INDEX + 1))
    echo "Progress: $CASE_INDEX / $TOTAL_COMBINATIONS"
    
    CASE_START_TIME=$(date +%s)
    
    # Prepare base command
    BASE_CMD="python3 blackbox_attack_runner.py --substitute_model $SUBSTITUTE_MODEL --data_dir ./dataset --batch_size 32 --attack $ATTACK --eps 0.3"
    
    # Add case-specific parameters
    case $case in
        "dim")
            CMD="$BASE_CMD --use_dim --resize_rate 1.1 --diversity_prob 0.5"
            CASE_NAME="DIM only"
            ;;
        "tim")
            CMD="$BASE_CMD --use_tim --kernel_size 5"
            CASE_NAME="TIM only"
            ;;
        "sim")
            CMD="$BASE_CMD --use_sim --scale_factors 1.0 0.9 0.8 0.7 0.6"
            CASE_NAME="SIM only"
            ;;
        "pim")
            CMD="$BASE_CMD --use_pim --patch_size 16"
            CASE_NAME="PIM only"
            ;;
        "dim_tim")
            CMD="$BASE_CMD --use_dim --use_tim --resize_rate 1.1 --diversity_prob 0.5 --kernel_size 5"
            CASE_NAME="DIM + TIM"
            ;;
        "dim_sim")
            CMD="$BASE_CMD --use_dim --use_sim --resize_rate 1.1 --diversity_prob 0.5 --scale_factors 1.0 0.9 0.8 0.7 0.6"
            CASE_NAME="DIM + SIM"
            ;;
        "tim_sim")
            CMD="$BASE_CMD --use_tim --use_sim --kernel_size 5 --scale_factors 1.0 0.9 0.8 0.7 0.6"
            CASE_NAME="TIM + SIM"
            ;;
        "dim_tim_sim")
            CMD="$BASE_CMD --use_dim --use_tim --use_sim --resize_rate 1.1 --diversity_prob 0.5 --kernel_size 5 --scale_factors 1.0 0.9 0.8 0.7 0.6"
            CASE_NAME="DIM + TIM + SIM"
            ;;
    esac
    
    result_line="$CASE_NAME"
    sum=0
    count=0
    
    for target_model in "${TARGET_MODELS[@]}"; do
        FULL_CMD="$CMD --target_model $target_model"
        echo "  Running: $FULL_CMD"
        
        result=$(eval $FULL_CMD | grep "Black-box attack success rate:" | awk '{printf "%.2f", $5}')
        if [ -z "$result" ]; then
            result="N/A"
        else
            sum=$(echo "$sum + $result" | bc)
            count=$((count + 1))
        fi
        result_line="$result_line,$result"
    done
    
    # Calculate average
    if [ $count -gt 0 ]; then
        average=$(echo "scale=2; $sum / $count" | bc)
    else
        average="N/A"
    fi
    result_line="$result_line,$average"
    
    echo $result_line >> $OUTPUT_FILE
    
    CASE_END_TIME=$(date +%s)
    CASE_DURATION=$((CASE_END_TIME - CASE_START_TIME))
    
    ELAPSED_TIME=$((CASE_END_TIME - START_TIME))
    AVG_TIME_PER_CASE=$((ELAPSED_TIME / CASE_INDEX))
    REMAINING_CASES=$((TOTAL_COMBINATIONS - CASE_INDEX))
    ESTIMATED_REMAINING_TIME=$((AVG_TIME_PER_CASE * REMAINING_CASES))
    
    echo "Completed case in $CASE_DURATION seconds"
    echo "Estimated remaining time: $(($ESTIMATED_REMAINING_TIME / 60)) minutes"
    echo "----------------------------------------"
done

# Convert CSV to Excel with formatting
python3 - << EOL
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Read the CSV file
df = pd.read_csv('$OUTPUT_FILE')

# Create a new Excel writer object
writer = pd.ExcelWriter('hybrid_attack_results.xlsx', engine='openpyxl')

# Write the DataFrame to Excel
df.to_excel(writer, sheet_name='Results', index=False)

# Get the workbook and the worksheet
workbook = writer.book
worksheet = writer.sheets['Results']

# Define styles
header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
header_font = Font(bold=True, color='FFFFFF')
border = Border(left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'), bottom=Side(style='thin'))

# Apply styles to header row
for cell in worksheet[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')

# Format all cells
for row in worksheet.iter_rows(min_row=2):
    for cell in row:
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
        # Color the average column with light blue
        if cell.column == worksheet.max_column:
            cell.fill = PatternFill(start_color='DCE6F1', end_color='DCE6F1', fill_type='solid')

# Adjust column widths
for column in worksheet.columns:
    max_length = 0
    column = list(column)
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = (max_length + 2)
    worksheet.column_dimensions[get_column_letter(column[0].column)].width = adjusted_width

# Add attack averages at the bottom
worksheet.append([])
avg_row = worksheet.max_row + 1
worksheet.cell(row=avg_row, column=1, value='Average').font = Font(bold=True)

# Calculate averages for each target model
for col in range(2, worksheet.max_column + 1):  # Start from first target model column
    col_letter = get_column_letter(col)
    worksheet[f"{col_letter}{avg_row}"] = f"=AVERAGE({col_letter}2:{col_letter}{worksheet.max_row-2})"
    worksheet[f"{col_letter}{avg_row}"].font = Font(bold=True)
    worksheet[f"{col_letter}{avg_row}"].fill = PatternFill(start_color='DCE6F1', end_color='DCE6F1', fill_type='solid')

workbook.save('hybrid_attack_results.xlsx')
EOL

echo "All combinations completed. Results saved to hybrid_attack_results.xlsx"
