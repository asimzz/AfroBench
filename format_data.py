import os
import pandas as pd
import json

# Define task, subtask, and dataset mapping
TASK_MAPPING = {
    "MasakhaPOS": ("NLU", "TokC"),
    "MasakhaNER": ("NLU", "TokC"),
    "AfriSenti": ("NLU", "Senti"),
    "NollySenti": ("NLU", "Senti"),
    "InjongoIntent": ("NLU", "Intent"),
    "MasakhaNEWS": ("NLU", "Topic"),
    "SIB": ("NLU", "Topic"),
    "AfriHate": ("NLU", "Hate"),
    "AfriXNLI": ("NLU", "NLI"),
    "AfriQA": ("QA", "XQA"),
    "Belebele": ("QA", "RC"),
    "NaijaRC": ("QA", "RC"),
    "UHURA": ("Knowledge", "Arc-E"),
    "OpenAIMMLU": ("Knowledge", "MMLU"),
    "AfriMMLU": ("Knowledge", "MMLU"),
    "AfriMGSM": ("Reasoning", "Math"),
    "SALT - en_xx": ("NLG", "MT"),
    "SALT - xx_en": ("NLG", "MT"),
    "Flores - en_xx": ("NLG", "MT"),
    "Flores - xx_en": ("NLG", "MT"),
    "MAFAND - en_xx": ("NLG", "MT"),
    "MAFAND - xx_en": ("NLG", "MT"),
    "NTREX - en_xx": ("NLG", "MT"),
    "NTREX - xx_en": ("NLG", "MT"),
    "XLSUM": ("NLG", "Summ"),
    "ADR": ("NLG", "Diacritics")

}

# Input directory for CSV files
DATA_DIR = "datasets"

# Output directory for JSON files
OUTPUT_DIR = "leaderboard_json/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary to store task-wise grouped data
task_data = {}
model_map = {
    "AfroLlama-V1": "AfroLLaMa 8B",
    "LLaMAX3-8B-Alpaca": "LLaMAX3 8B",
    "Llama-2-7b-chat-hf": "LLaMa2 7b",
    "Llama-3.1-70B-Instruct": "LLaMa3.1 70B",
    "Llama-3.1-8B-Instruct": "LLaMa3.1 8B",
    "Meta-Llama-3-8B-Instruct": "LLaMa3 8B",
    "aya-101": "Aya 101",
    "gemma-1.1-7b-it": "Gemma1.1 7b",
    "gemma-2-27b-it": "Gemma2 27b",
    "gemma-2-9b-it": "Gemma2 9b",
    "gemini-1.5-pro-002": "Gemini 1.5 pro",
    "gpt-4o-2024-08-06": "GPT-4o (Aug)",
    "Gemma 2 IT 27B": "Gemma2 27b",
    "Gemma 2 IT 9B": "Gemma2 9b",
    "Aya-101": "Aya-101 13B",
    "Meta-Llama-3.1-70B-Instruct": "LLaMa3.1 70B",
    "LLaMAX3-8B": "LLaMAX3 8B",
    "Meta-Llama-3-70B-Instruct": "LLaMa3.1 70B"}

task_map = {key.lower(): value for key, value in TASK_MAPPING.items()}
# Process each CSV file
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, filename)
        dataset_name = filename.replace(" - 0-shot.csv", "").replace(" 0-shot.csv", "")

        # Identify task & subtask
        task_info = task_map.get(dataset_name.lower())
        if not task_info:
            print(dataset_name.lower())
            continue  # Skip if dataset is not mapped

        task, subtask = task_info

        # Read CSV
        df = pd.read_csv(file_path)
        df.loc[df["model"].str.contains("LLaMaX"), "model"] = "LLaMaX 3 8B"
        df = df[df["model"] != "InkubaLM-0.4B"].copy()
        df = df[df["model"] != "Claude 3.5 Sonnet"].copy()
        df.loc[df["model"].str.contains("gpt"), "model"] = "gpt-4o-2024-08-06"
        df.loc[df["model"].str.contains("gemini"), "model"] = "gemini-1.5-pro-002"
        df["model"] = df["model"].map(model_map)

        # Extract models and prompts
        models = df["model"].unique()

        # Get language columns (excluding metadata columns)
        language_columns = [col for col in df.columns if col not in ["model", "prompt", "avg_score", "avg"]]
        avg_col = "avg" if "avg" in df.columns else "avg_score"

        # Initialize task if not exists
        if task not in task_data:
            task_data[task] = {"task": task, "subtasks": {}}

        # Initialize subtask if not exists
        if subtask not in task_data[task]["subtasks"]:
            task_data[task]["subtasks"][subtask] = {"datasets": {}}

        # Add dataset entry
        task_data[task]["subtasks"][subtask]["datasets"][dataset_name] = {
            "languages": language_columns,
            "scores": {}
        }

        # Populate model scores per language
        for model in models:
            # model_scores = df[df["model"] == model][avg_col].idxmax()
            # model_scores = df[df["model"] == model][language_columns].mean().to_list()  # Average across prompts
            best_avg_row = df[df["model"] == model].loc[df[df["model"] == model][avg_col].idxmax()]
            model_scores = best_avg_row[language_columns].to_list()
            task_data[task]["subtasks"][subtask]["datasets"][dataset_name]["scores"][model] = model_scores

# Save each task as a separate JSON file
for task, data in task_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{task.lower().replace(' ', '_')}.json")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

print("Task-wise JSON files with subtasks generated successfully!")