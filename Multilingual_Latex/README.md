# Language Switcher for LaTeX

## Introduction
This repository contains a Python script that automatically adds `\selectlanguage` commands to LaTeX documents when switching between Greek and English text. It is designed to process plain text files and output the modified text with the necessary LaTeX commands for language switching.

## Features
- Detects Greek and English characters.
- Automatically inserts `\selectlanguage{language}` in LaTeX format.
- Handles text files as input.
- Outputs processed text to a new file.

## Setup
To use this script, you need to have Python installed on your system. You can download and install Python from [here](https://www.python.org/downloads/).

## Usage
The script can be used in two ways:

### As a Python Script
1. Clone this repository or download the script.
2. Run the script using Python and pass the path of the text file you want to process. For example:
python language_switcher.py yourfile.txt
3. The script will create a new file with the processed text in the same directory as the input file.

### As an Executable
1. A standalone executable version of the script is available in the `dist` folder.
2. Simply drag and drop a text file onto the executable.
3. The executable will process the file and output the result in the same directory with `_processed.txt` appended to the original filename.

## Building the Executable
If you wish to build the executable yourself, follow these steps:

1. Install PyInstaller:
pyinstaller --onefile language_switcher.py

## Additional Notes

The script currently supports only Greek and English languages. Additional language support can be added in the future.
This project was created as a utility tool for LaTeX users and is not a full-fledged language detection system.