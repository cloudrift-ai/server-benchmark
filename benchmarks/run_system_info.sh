#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Collecting system information..."
$SCRIPT_DIR/collect_system_info.sh

echo "System information collection complete!"
