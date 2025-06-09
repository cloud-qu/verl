#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Source server details
SOURCE_SERVER="di35zis@login.ai.lrz.de"
SOURCE_BASE="/dss/dsshome1/0E/di35zis/lab/verl"
# SOURCE_PASS is now loaded from .env file

# Target machine details
TARGET_USER="quy"
TARGET_HOST="182.18.90.106"
TARGET_PORT="8092"
TARGET_KEY="~/.ssh/id_ed25519"  # Replace with your actual private key path
TARGET_PATH="/data2/quy/verl/ckpts"

# Create temporary directory for local storage
TEMP_DIR="./temp_ckpts"
mkdir -p $TEMP_DIR

# Process each checkpoint line
while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Remove any quotes and trim whitespace
    base_path=$(echo "$line" | tr -d '"' | xargs)
    
    echo "Processing: $base_path"
    
    # Create local directory
    local_path="$TEMP_DIR/$(dirname $base_path)"
    mkdir -p "$local_path"
    
    # Copy from source server to local
    echo "Copying from server: $SOURCE_SERVER:$SOURCE_BASE/$base_path"
    if ! sshpass -p "$SOURCE_PASS" scp -r "$SOURCE_SERVER:$SOURCE_BASE/$base_path" "$local_path/"; then
        echo "Error: Failed to copy from source server"
        continue
    fi
    
    # Copy from local to target machine
    echo "Copying to target: $TARGET_USER@$TARGET_HOST:$TARGET_PATH/$base_path"
    if ! scp -i "$TARGET_KEY" -P "$TARGET_PORT" -r "$local_path/$(basename $base_path)" "$TARGET_USER@$TARGET_HOST:$TARGET_PATH/$base_path"; then
        echo "Error: Failed to copy to target machine"
        continue
    fi
    
done < checkpoint.txt

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf $TEMP_DIR

echo "Transfer completed!"