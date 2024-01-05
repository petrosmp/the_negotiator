import sys

def is_greek_char(char):
    """
    Check if the character is a Greek character. 
    Greek characters fall within the Unicode range 0370 to 03FF and 1F00 to 1FFF.
    """
    return '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF'

def is_english_char(char):
    """
    Check if the character is an English character. 
    English characters are ASCII characters from 0041 to 007A.
    """
    return '\u0041' <= char <= '\u007A' or '\u0061' <= char <= '\u007A'

def append_language_commands(text):
    """
    Process the input text and append \selectlanguage{language} commands 
    when a change in language is detected.
    """
    if not text:
        return ""

    # Determine the starting language
    current_language = "greek" if is_greek_char(text[0]) else "english"
    result = f"\\selectlanguage{{{current_language}}}"

    for char in text:
        if is_greek_char(char) and current_language != "greek":
            current_language = "greek"
            result += f"\\selectlanguage{{greek}}"
        elif is_english_char(char) and current_language != "english":
            current_language = "english"
            result += f"\\selectlanguage{{english}}"
        result += char

    return result

# Example usage
# example_text = "Σήμερα είδα για πρώτη φορά το site του Γκουσγκούνη."
# processed_text = append_language_commands(example_text)

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        processed_text = append_language_commands(text)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(processed_text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: <script> <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = input_file_path + "_processed.txt"
    
    process_file(input_file_path, output_file_path)
