#!/bin/bash

# Source server details
SOURCE_SERVER="di35zis@login.ai.lrz.de"
SOURCE_BASE="/dss/dsshome1/0E/di35zis/lab/verl"

# Target machine details
TARGET_USER="quy"
TARGET_HOST="182.18.90.106"
TARGET_PORT="8092"
TARGET_KEY="~/.ssh/id_ed25519"  # Replace with your actual private key path
TARGET_PATH="/data2/quy/verl/ckpts"

# Create temporary directory for local storage
TEMP_DIR="./temp_ckpts"
mkdir -p $TEMP_DIR

# Function to expand step numbers
expand_steps() {
    local path=$1
    local steps=$2
    echo $steps | tr ',' '\n' | while read step; do
        echo "${path}/global_step_${step}"
    done
}

# Process each checkpoint line
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Extract base path and steps
    if [[ $line =~ ^\"?([^\"]+)\"?$ ]]; then
        base_path="${BASH_REMATCH[1]}"
        # Remove quotes if present
        base_path=${base_path//\"/}
        
        # Extract steps if present in the path
        if [[ $base_path =~ \{([^}]+)\} ]]; then
            steps="${BASH_REMATCH[1]}"
            # Remove the {steps} part from the base path
            base_path=${base_path/\{*\}/}
            
            # Expand and transfer each step
            for step_path in $(expand_steps "$base_path" "$steps"); do
                echo "Processing: $step_path"
                
                # Create local directory
                local_path="$TEMP_DIR/$(dirname $step_path)"
                mkdir -p "$local_path"
                
                # Copy from source server to local
                echo "Copying from server: $SOURCE_SERVER:$SOURCE_BASE/$step_path"
                scp -r "$SOURCE_SERVER:$SOURCE_BASE/$step_path" "$local_path/"
                
                # Copy from local to target machine
                echo "Copying to target: $TARGET_USER@$TARGET_HOST:$TARGET_PATH/$step_path"
                scp -i "$TARGET_KEY" -P "$TARGET_PORT" -r "$local_path/$(basename $step_path)" "$TARGET_USER@$TARGET_HOST:$TARGET_PATH/$step_path"
            done
        else
            # Single path without steps
            echo "Processing: $base_path"
            
            # Create local directory
            local_path="$TEMP_DIR/$(dirname $base_path)"
            mkdir -p "$local_path"
            
            # Copy from source server to local
            echo "Copying from server: $SOURCE_SERVER:$SOURCE_BASE/$base_path"
            scp -r "$SOURCE_SERVER:$SOURCE_BASE/$base_path" "$local_path/"
            
            # Copy from local to target machine
            echo "Copying to target: $TARGET_USER@$TARGET_HOST:$TARGET_PATH/$base_path"
            scp -i "$TARGET_KEY" -P "$TARGET_PORT" -r "$local_path/$(basename $base_path)" "$TARGET_USER@$TARGET_HOST:$TARGET_PATH/$base_path"
        fi
    fi
done < checkpoint.txt

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf $TEMP_DIR

echo "Transfer completed!"