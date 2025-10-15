import csv
import glob
import os
import re

# Expected format: 200 timesteps * 3 axes = 600 values per sample
EXPECTED_VALUES_PER_SAMPLE = 600


def _is_valid_float(s):
    """Check if string can be converted to float"""
    try:
        float(s.strip())
        return True
    except (ValueError, TypeError):
        return False


def validate_data(data_points):
    """
    Validate data points. Return (is_valid, cleaned_list, errors).
    If any value is invalid or count wrong, is_valid=False.
    """
    if len(data_points) != EXPECTED_VALUES_PER_SAMPLE:
        return False, [], [(None, None, f"count={len(data_points)}, expected {EXPECTED_VALUES_PER_SAMPLE}")]

    cleaned = []
    errors = []
    for i, dp in enumerate(data_points):
        stripped = dp.strip()
        if not stripped:
            errors.append((i, dp[:20], "empty"))
            return False, [], errors
        if not _is_valid_float(stripped):
            errors.append((i, dp[:20], "non-numeric"))
            return False, [], errors
        cleaned.append(stripped)
    return True, cleaned, []


def convert_txt_to_csv(dataset_dir="dataset"):
    """
    Automatically traverse all .txt files in the dataset directory and generate corresponding .csv files
    """
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' does not exist")
        return

    # Get all .txt files
    txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in directory '{dataset_dir}'")
        return

    print(f"Found {len(txt_files)} .txt files")

    # Regular expression to extract data
    pattern = r"=== Data Collection Results ===(.*?)=== End of Data Collection ==="

    # Process each .txt file
    for txt_file in txt_files:
        # Get filename without extension
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        csv_file = os.path.join(dataset_dir, f"{base_name}.csv")

        print(f"\nProcessing: {txt_file}")

        try:
            # Read txt file
            with open(txt_file, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract data
            matches = re.findall(pattern, content, re.DOTALL)
            print(f"  Found {len(matches)} data collections")

            if len(matches) == 0:
                print(f"  Warning: No data found in {txt_file}")
                continue

            # Write to CSV file (only valid samples)
            kept = 0
            discarded = 0
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                for i, match in enumerate(matches, start=1):
                    match = match.strip()
                    data_points = match.split(",")

                    is_valid, cleaned_points, errors = validate_data(data_points)

                    if not is_valid:
                        discarded += 1
                        if errors:
                            idx, val, msg = errors[0]
                            reason = msg if idx is None else f"idx {idx}: '{val}' ({msg})"
                            print(f"    Discarded sample {i}: {reason}")
                        continue

                    writer.writerow(cleaned_points)
                    kept += 1

            print(f"  Successfully generated: {csv_file}")
            print(f"  Kept: {kept}, Discarded: {discarded}")

        except Exception as e:
            print(f"  Error: Failed to process {txt_file}: {str(e)}")

    print(f"\nConversion completed! Processed {len(txt_files)} files")


if __name__ == "__main__":
    convert_txt_to_csv("shuttle_dataset")
