


import os
import pandas as pd
import json

# Define the dataset-to-task mapping
TASK_MAPPING = {
    "Afrimmlu": "QA",
    "Flores_en_xx": "MT",
    "Flores_xx_en": "MT",
    "XQUAD": "QA",
    "MMLU": "Reasoning",
    "Summarization": "Summarization",
    "Diacritics": "Diacritics Restoration"
}

# Input directory containing CSV files
DATA_DIR = "benchmark/data/"

# Output directory for JSON files
OUTPUT_DIR = "benchmark/leaderboard_json/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary to store task-wise grouped data
task_data = {}

# Process each CSV file
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, filename)
        dataset_name = filename.replace(" - 0-shot.csv", "").replace(" - 5-shot.csv", "")

        # Read CSV
        df = pd.read_csv(file_path)

        # Identify task
        task = TASK_MAPPING.get(dataset_name, "Other")

        # Extract model names and prompts
        models = df["model"].unique()

        # Get language columns (excluding model and prompt)
        language_columns = [col for col in df.columns if col not in ["model", "prompt", "avg_score"]]

        # Initialize task if not already in dictionary
        if task not in task_data:
            task_data[task] = {"task": task, "datasets": {}}

        # Add dataset entry
        task_data[task]["datasets"][dataset_name] = {
            "languages": language_columns,
            "scores": {}
        }

        # Populate model scores per language
        for model in models:
            model_scores = df[df["model"] == model][language_columns].mean().to_list()  # Average across prompts
            task_data[task]["datasets"][dataset_name]["scores"][model] = model_scores

# Save each task as a separate JSON file
for task, data in task_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{task.lower().replace(' ', '_')}.json")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

print("Task-wise JSON files generated successfully!")
