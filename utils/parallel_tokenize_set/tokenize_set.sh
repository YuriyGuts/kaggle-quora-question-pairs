#!/usr/bin/env bash

INPUT_FILE=${1:-test.csv}
OUTPUT_FILE=${2:-test.json}
LINES_PER_CHUNK=${3:-100000}


# Split the file and spawn subjobs.
tail -n +2 "$INPUT_FILE" | split -l $LINES_PER_CHUNK
SUBPROCESSES=""

for file in x*
do
    (head -n 1 "$INPUT_FILE"; cat "$file") > "$file".new
    mv "$file".new "$file"
    mv "$file" "$file.csv"

    ./tokenize_chunk.py "$file.csv" "$file.json" &
    SUBPROCESSES="$SUBPROCESSES $!"
done


# Wait for jobs to complete.
for pid in $SUBPROCESSES; do
    wait $pid || let "RESULT=1"
done

if [ "$RESULT" == "1" ];
    then
       exit 1
fi


# Merge the chunks into a single JSON.
./merge_chunks.py "$OUTPUT_FILE"
