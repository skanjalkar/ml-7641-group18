#!/bin/bash

# Kill all background processes started by the parser script
pkill -f "python3 src/parser/parser.py"

echo "All parser processes have been terminated."
