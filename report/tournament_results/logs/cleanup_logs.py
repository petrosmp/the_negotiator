INPUT_FILE = "TheNegotiator_no_tracebacks.log"
OUTPUT_FILE = "TheNegotiator_cleaned.log"

with open(INPUT_FILE, "r") as input_file, open(OUTPUT_FILE, "w") as output_file:
    
    last_line = None
    
    for line in input_file:

        if line != last_line:
            output_file.write(line)
            last_line = line