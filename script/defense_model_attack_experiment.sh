#!/bin/bash

# Define models and attack configurations
SUBSTITUTE_MODEL="Inc-v3"
TARGET_MODELS=("hgd" "rs" "r_p" "bit_red" "feature_distill" "jpeg_defense" "nips_r3" "comdefend" "purify")
ATTACK="hybrid_fgsm"
CASES=("dim" "tim" "sim" "pim" "dim_tim" "dim_sim" "tim_sim" "dim_tim_sim")

OUTPUT_FILE="defense_attack_results.csv"

# Create CSV header
echo -n "Attack Method,Case" > $OUTPUT_FILE
for target_model in "${TARGET_MODELS[@]}"; do
    echo -n ",$target_model" >> $OUTPUT_FILE
done
echo ",Average" >> $OUTPUT_FILE

# Calculate total combinations
TOTAL_COMBINATIONS=${#CASES[@]}
CURRENT_COMBINATION=0
START_TIME=$(date +%s)

# Run hybrid attack with different cases
for case in "${CASES[@]}"; do
    CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
    echo "Progress: $CURRENT_COMBINATION / $TOTAL_COMBINATIONS"
    echo "Running hybrid attack: $ATTACK, case: $case"
    
    COMBINATION_START_TIME=$(date +%s)
    
    # Base command
    BASE_CMD="python3 defense_models_attack.py --substitute_model $SUBSTITUTE_MODEL --data_dir ./dataset --batch_size 32 --attack $ATTACK --eps 0.3 --steps 10"
    
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
    
    result_line="$ATTACK,$CASE_NAME"
    sum=0
    count=0
    
    for target_model in "${TARGET_MODELS[@]}"; do
        FULL_CMD="$CMD --target_model $target_model"
        echo "  Running: $FULL_CMD"
        
        result=$(eval $FULL_CMD | grep "Attack success rate:" | awk '{printf "%.2f", $4}')
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
    
    # Progress tracking
    COMBINATION_END_TIME=$(date +%s)
    COMBINATION_DURATION=$((COMBINATION_END_TIME - COMBINATION_START_TIME))
    ELAPSED_TIME=$((COMBINATION_END_TIME - START_TIME))
    AVG_TIME_PER_COMBINATION=$((ELAPSED_TIME / CURRENT_COMBINATION))
    REMAINING_COMBINATIONS=$((TOTAL_COMBINATIONS - CURRENT_COMBINATION))
    ESTIMATED_REMAINING_TIME=$((AVG_TIME_PER_COMBINATION * REMAINING_COMBINATIONS))
    
    echo "Completed combination in $COMBINATION_DURATION seconds"
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
writer = pd.ExcelWriter('defense_attack_results.xlsx', engine='openpyxl')

# Write the DataFrame to Excel
df.to_excel(writer, sheet_name='Results', index=False)

# Get the workbook and the worksheet
workbook = writer.book
worksheet = writer.sheets['Results']

# Define styles
header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
header_font = Font(bold=True, color='FFFFFF')
attack_fill = PatternFill(start_color='B8CCE4', end_color='B8CCE4', fill_type='solid')
case_fill = PatternFill(start_color='E4ECF7', end_color='E4ECF7', fill_type='solid')
border = Border(left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'), bottom=Side(style='thin'))

# Apply styles to header row
for cell in worksheet[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center')

# Format all cells and apply conditional formatting
for row in worksheet.iter_rows(min_row=2):
    row[0].fill = attack_fill
    row[1].fill = case_fill
    
    for cell in row:
        cell.border = border
        cell.alignment = Alignment(horizontal='center', vertical='center')
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
worksheet.cell(row=avg_row, column=1, value='Overall Average').font = Font(bold=True)

# Calculate averages for each target model
for col in range(3, worksheet.max_column + 1):  # Start from first target model column
    col_letter = get_column_letter(col)
    worksheet[f"{col_letter}{avg_row}"] = f"=AVERAGE({col_letter}2:{col_letter}{worksheet.max_row-2})"
    worksheet[f"{col_letter}{avg_row}"].font = Font(bold=True)
    worksheet[f"{col_letter}{avg_row}"].fill = PatternFill(start_color='DCE6F1', end_color='DCE6F1', fill_type='solid')

# Merge the "Overall Average" cells
worksheet.merge_cells(f"A{avg_row}:B{avg_row}")
worksheet[f"A{avg_row}"].alignment = Alignment(horizontal='center', vertical='center')

workbook.save('defense_attack_results.xlsx')
EOL

echo "All experiments completed. Results saved to defense_attack_results.xlsx"
