#!/bin/bash
#
# Stop Progressive Teacher Server
#
# Use this to gracefully stop the teacher server
#

echo "Stopping progressive teacher server..."

# Find and kill the teacher server process
pkill -f "progressive_teacher_server.py"

if [ $? -eq 0 ]; then
    echo "✓ Teacher server stopped"
else
    echo "✗ No teacher server process found"
fi

# Optional: clean up any zombie processes
sleep 1
pkill -9 -f "progressive_teacher_server.py" 2>/dev/null

echo "Done."
