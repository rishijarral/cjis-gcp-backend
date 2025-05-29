#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.


echo "Changing ownership of /app/data to appuser:appgroup..."
chown -R appuser:appgroup /app/data
echo "Ownership change complete."

echo "Executing command as appuser: $@"
exec gosu appuser "$@"