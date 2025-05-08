set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
TASK_DIR=$WORK_DIR/tasks
OUTPUT_DIR=$WORK_DIR/../results

TASKS=(
    "afrimmlu"
    "afrimgsm"
)

MODEL_NAMES=(
    "gpt-3.5-turbo"
    "gpt-4"
    "gpt-4-32k"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running task with model: $MODEL_NAME"
    for TASK in "${TASKS[@]}"; do
        echo "Running task: $TASK"
        python3 $WORK_DIR/run.py \
                --tasks $TASK_DIR/$TASK.yaml \
                --model $MODEL_NAME \
                --output $OUTPUT_DIR
    done
done