#!/bin/bash

# Define the file ID and destination
FILE_ID="1BhTPLUkfN6e2xkqR8LEm9lByXbLY1IYd"
DESTINATION="/mnt/disk1/users/afshari/wavlmModel/wavlm_model.zip"

# Step 1: Get the confirmation token
curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CONFIRM=$(awk '/_warning_/ {print $NF}' /tmp/gcookie)

# Step 2: Use the confirmation token to download the file
curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o ${DESTINATION}

# Unzip the downloaded file
unzip ${DESTINATION} -d /mnt/disk1/users/afshari/wavlmModel/
