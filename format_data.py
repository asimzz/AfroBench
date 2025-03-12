import os
import json
import glob
import pandas as pd


# Define task, subtask, and dataset mapping
TASK_MAPPING = {
    "MasakhaPOS": ("NLU", "POS"),
    "MasakhaNER": ("NLU", "NER"),
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
    "SALT - en_xx": ("NLG", "MT(en/fr-xx)"),
    "SALT - xx_en": ("NLG", "MT(xx-en/fr)"),
    "Flores - en_xx": ("NLG", "MT(en/fr-xx)"),
    "Flores - xx_en": ("NLG", "MT(xx-en/fr)"),
    "MAFAND - en_xx": ("NLG", "MT(en/fr-xx)"),
    "MAFAND - xx_en": ("NLG", "MT(xx-en/fr)"),
    "NTREX - en_xx": ("NLG", "MT(en/fr-xx)"),
    "NTREX - xx_en": ("NLG", "MT(xx-en/fr)"),
    "XLSUM": ("NLG", "SUMM"),
    "ADR": ("NLG", "ADR")

}

MODEL_MAP = {
    "AfroLlama-V1": "AfroLLaMa 8B",
    "LLaMAX3-8B-Alpaca": "LLaMAX3 8B",
    "Llama-2-7b-chat-hf": "LLaMa2 7b",
    "Llama-3.1-70B-Instruct": "LLaMa3.1 70B",
    "Llama-3.1-8B-Instruct": "LLaMa3.1 8B",
    "Meta-Llama-3-8B-Instruct": "LLaMa3 8B",
    "aya-101": "Aya-101 13B",
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
    "LLaMaX 3 8B": "LLaMAX3 8B",
    "Meta-Llama-3-70B-Instruct": "LLaMa3.1 70B"
}


def generate_json_files(data_dir="datasets", output_dir="leaderboard_json", leaderboard=None):
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store either per-task JSON data or the leaderboard
    task_data = {}
    leaderboard_data = {}

    task_map = {key.lower(): value for key, value in TASK_MAPPING.items()}

    # Afrobench-Lite included datasets
    afrobench_lite_datasets = {"injongointent", "sib", "afrixnli", "belebele", "afrimmlu", "afrimgsm",
                               "flores - en_xx", "flores - xx_en"}

    # Process each CSV file
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            dataset_name = filename.replace(" - 0-shot.csv", "").replace(" 0-shot.csv", "")

            # Identify task & subtask
            task_info = task_map.get(dataset_name.lower())
            if not task_info:
                print(f"Skipping unmapped dataset: {dataset_name.lower()}")
                continue

            task, subtask = task_info

            # Read CSV
            df = pd.read_csv(file_path)

            drop_col = [i for i in df.columns if 'unnamed' in i.lower()]
            df.drop(drop_col, axis=1, inplace=True)

            # Standardize model names
            df.loc[df["model"].str.contains("LLaMaX", case=False), "model"] = "LLaMaX 3 8B"
            df = df[df["model"] != "InkubaLM-0.4B"].copy()
            df = df[df["model"] != "Claude 3.5 Sonnet"].copy()
            df.loc[df["model"].str.contains("gpt", case=False), "model"] = "gpt-4o-2024-08-06"
            df.loc[df["model"].str.contains("gemini", case=False), "model"] = "gemini-1.5-pro-002"
            df["model"] = df["model"].map(MODEL_MAP)

            # Extract models
            models = df["model"].unique()

            all_columns = list(df.columns)
            meta_columns = ["model", "prompt", "avg_score", "avg"]
            language_columns = [col for col in all_columns if col not in meta_columns]
            language_columns = [col for col in language_columns if col.lower() not in {"eng", "fra",
                                                                                       "eng_latn, fra_latn"}]

            avg_col = "avg" if "avg" in df.columns else "avg_score"

            if leaderboard == "afrobench":
                # Initialize leaderboard structure
                if task not in leaderboard_data:
                    leaderboard_data[task] = {}

                if subtask not in leaderboard_data[task]:
                    leaderboard_data[task][subtask] = {"datasets": {}}

                # Store per-model dataset scores
                dataset_scores = {}
                for model in models:
                    best_avg_row = df[df["model"] == model].loc[df[df["model"] == model][avg_col].idxmax()]
                    scores = [best_avg_row[col] for col in language_columns if col in best_avg_row]
                    dataset_scores[model] = round(sum(scores) / len(scores) if scores else None, 1)  # Avoid division by zero

                leaderboard_data[task][subtask]["datasets"][dataset_name] = dataset_scores

            elif leaderboard == "afrobench_lite":
                # Only process datasets included in Afrobench-Lite
                if dataset_name in afrobench_lite_datasets:
                    # Use subtask name as task key (no nested "subtasks" structure)
                    if subtask not in leaderboard_data:
                        leaderboard_data[subtask] = {}

                    # Store per-model dataset scores
                    dataset_scores = {}
                    for model in models:
                        best_avg_row = df[df["model"] == model].loc[df[df["model"] == model][avg_col].idxmax()]
                        scores = [best_avg_row[col] for col in language_columns if col in best_avg_row]
                        dataset_scores[model] = round(sum(scores) / len(scores) if scores else None,
                                                      1)  # Avoid division by zero

                    leaderboard_data[subtask][dataset_name] = dataset_scores

            else:
                # Initialize task & subtask structure
                if task not in task_data:
                    task_data[task] = {"task": task, "subtasks": {}}

                if subtask not in task_data[task]["subtasks"]:
                    task_data[task]["subtasks"][subtask] = {"datasets": {}}

                # Store per-task dataset data
                task_data[task]["subtasks"][subtask]["datasets"][dataset_name] = {
                    "languages": language_columns,
                    "scores": {}
                }

                for model in models:
                    best_avg_row = df[df["model"] == model].loc[df[df["model"] == model][avg_col].idxmax()]
                    model_scores = [round(score, 1) for score in best_avg_row[language_columns].to_list()]
                    task_data[task]["subtasks"][subtask]["datasets"][dataset_name]["scores"][model] = model_scores

    # Save leaderboard JSON if enabled
    if leaderboard:
        output_path = os.path.join(output_dir, f"{leaderboard}.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(leaderboard_data, json_file, indent=4)
        print("Leaderboard JSON generated successfully!")

    # Save per-task JSON files if leaderboard=False
    else:
        for task, data in task_data.items():
            output_path = os.path.join(output_dir, f"{task.lower().replace(' ', '_')}.json")
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
        print("Task-wise JSON files with subtasks generated successfully!")


generate_json_files(leaderboard="afrobench_lite")
