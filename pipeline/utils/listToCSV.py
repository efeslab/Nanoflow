import csv
import re
import argparse

def list_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read()

    # Use regex to split data into cycles (both regular and skip cycles)
    cycle_pattern = r'(------------------ (?:Cycle|Skip Cycle) (\d+) ------------------)'
    cycles = re.split(cycle_pattern, data)[1:]

    rows = []
    for i in range(0, len(cycles), 3):
        cycle_type = cycles[i].strip()  # Cycle type (Cycle or Skip Cycle)
        cycle_number = cycles[i + 1].strip()  # Cycle number
        fields = cycles[i + 2].strip().splitlines()  # Fields within the cycle

        # Determine if it's a skip cycle
        is_skip_cycle = 'Skip' in cycle_type

        # Parse fields and values
        field_values = {'Cycle': cycle_number, 'is_skip_cycle': is_skip_cycle}
        for field in fields:
            key, value = map(str.strip, field.split(':', 1))
            field_values[key] = value

        rows.append(field_values)

    # Get all possible field names
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(all_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Parse cycle logs and convert them to CSV.')
    parser.add_argument('input_file', type=str, help='Path to the input log file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the parser function with input and output paths
    list_to_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
