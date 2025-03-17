#!/bin/bash

# Log output for debugging
LOGFILE="/workspace/InstantID/startup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Starting InstantID setup at $(date)"

# Ensure we're in the correct directory
cd /workspace/InstantID || { echo "Failed to cd into /workspace/InstantID"; exit 1; }

# Run setup.py
echo "Running setup.py..."
python3 setup.py

# Ensure tmux session exists and start UFC_operator.py inside it
SESSION_NAME="instantid"
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s $SESSION_NAME
fi

# Run the Gradio interface inside the tmux session
echo "Launching UFC_operator.py in tmux session..."
tmux send-keys -t $SESSION_NAME "cd /workspace/InstantID && python3 gradio/UFC_operator.py" C-m

echo "Startup script finished at $(date)"
