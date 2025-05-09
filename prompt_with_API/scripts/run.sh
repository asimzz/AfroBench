set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
TASK_DIR=$WORK_DIR/tasks

TASKS=(
    "arc-easy"
    "afrimmlu"
    "afrimgsm"
    "afrisenti"
    "afrihate"
    "afrixnli"
    "openai_mmlu"
    "afriqa"
    "belebele"
    "masakhanews"
    "masakhapos"
    "masakhaner"
    "intent"
    "xlsum"
)

MODEL_NAMES=(
    "Jacaranda/UlizaLlama"
    "Jacaranda/AfroLlama_V1"
)

MODEL_ABBRS=(
    "uliza-llama"
    "afro-llama-v1"
)

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}
    echo "Running task with model: $MODEL_NAME"
    for TASK in "${TASKS[@]}"; do
        echo "Running task: $TASK"
        OUTPUT_DIR=$WORK_DIR/output/${TASK}/$MODEL_ABBR
        python3 $WORK_DIR/run.py \
                --tasks $TASK_DIR/$TASK.yaml \
                --model $MODEL_NAME \
                --limit 250 \
                --output $OUTPUT_DIR
    done
done