#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath -s "${SCRIPT_DIR}/../") # Assuming color_env.sh is in scripts/ relative to this test file's parent

# release-excluded File Path
EXCLUDE_FILE=".github/release-excluded"

pushd "$PROJECT_ROOT" || exit 1

# File existence check
if [ ! -f "$EXCLUDE_FILE" ]; then
  echo "$EXCLUDE_FILE not found!"
  exit 1
fi

# Read and delete each path
while IFS= read -r path; do
  # Handle Windows CRLF line endings: remove \r
  path=$(echo "$path" | tr -d '\r')

  # Skip empty lines
  [ -z "$path" ] && continue

  # Wildcard pattern handling
  if [[ "$path" == *"*"* ]]; then
    # Expand glob patterns and delete matching files
    shopt -s nullglob
    matches=("${PROJECT_ROOT}/${path}")
    shopt -u nullglob
    
    if [ ${#matches[@]} -gt 0 ]; then
      for match in "${matches[@]}"; do
        echo "Deleting: ${match#${PROJECT_ROOT}/} ..."
        rm -rf "$match" && {
          echo "[Deleted] ${match#${PROJECT_ROOT}/}"
        } || {
          echo "[Failed to delete] ${match#${PROJECT_ROOT}/}"
          exit 1
        }
      done
    else
      echo "[SKIP] No matches found: $path"
    fi
  elif [ -e "${PROJECT_ROOT}/$path" ]; then
    echo "Deleting: $path ..."
    rm -rf "${PROJECT_ROOT}/$path" && {
        echo "[Deleted] $path"
    } || {
        echo "[Failed to delete] $path"
        exit 1
    }
  else
    echo "[SKIP] Not found: $path"
  fi
done < "$EXCLUDE_FILE"

popd || exit 1
exit 0
