#!/bin/bash
# lint.sh sorts, cleans and lints the code in this repository

set -e 
VENV_PATH="./venv"

if [ -f "./venv/bin/activate" ]; then
    source $VENV_PATH/bin/activate
elif [ -f "./venv/Scripts/activate" ]; then
    source $VENV_PATH/Scripts/activate
else
    echo "No virtual environment found"
fi

echo "Installing packages for linting..."
for pkg in black isort flake8 autoflake; do
    if ! pip show "$pkg" > /dev/null 2>&1; then 
        echo "Installing $pkg"
        pip install "$pkg"
    fi
done
echo "Done installing..."

TARGET_DIR="."
echo ""
echo "1: Removing unused imports and variables with autoflake..."
find . -type f -name "*.py" ! -path "./venv/*" -print0 | xargs -0 autoflake --remove-all-unused-imports --remove-unused-variables --in-place
echo "Autoflake done."

echo ""
echo "2: Sorting imports"
isort --profile black "$TARGET_DIR"
echo "Isort done"

echo ""
echo "3: Formatting code with Black"
black "$TARGET_DIR"
echo "Black done"

echo ""
echo "4: Linting with Flake9"
find . -type f -name "*.py" ! -path "./venv/*" -print0 | xargs -0 flake8 --max-line-length=88 --ignore=E203,W503,E501
echo "Flake8 done"