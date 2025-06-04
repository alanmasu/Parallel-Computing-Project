#!/bin/bash
# This script is used to execute a Python script in a specific environment
# Usage: source executePython.sh <script_name> <args>
#ENV_SCRIPT="${PWD}/utils/env-configuration.sh"
PYTHON_ENV="${PWD}/graphEnv/bin/activate"

# source $ENV_SCRIPT

# Activate the Python environment
source $PYTHON_ENV

# Check if the script name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <script_name> [args]"
    exit 1
fi
# Check if the script exists    
if [ ! -f "$1" ]; then
    echo "Error: Script $1 not found!"
    exit 1
fi

# Execute the Python script with the provided arguments
python3 $1 "${@:2}"

# Find __pycache__ directories and remove them
# find . -type d -name "__pycache__" -exec rm -rf {} +

# Deactivate the Python environment
deactivate