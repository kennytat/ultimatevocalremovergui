#!/bin/bash

# Check if USERNAME and PASSWORD variables are not empty
if [ -n "$MS_USER" ] && [ -n "$MS_PASS" ]; then
	# Modify main.py file using sed
	sed -i "s/admin/$MS_USER/g; s/mypassword/$MS_PASS/g" main.py
	echo "main.py file updated successfully."
else
	echo "MS_USER or MS_PASS variable is empty."
fi

python main.py & python UVR-webui.py "$@"