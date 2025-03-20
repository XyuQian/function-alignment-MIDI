for task in $(seq 0 11); do
    echo "Running task $task..."
    python -m shoelace.datasets.preprocess_5_tasks $task
done

echo "All tasks completed."