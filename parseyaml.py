#!/usr/bin/env python3
import yaml
import sys
import json

# Function to recursively process the YAML and concatenate keys and values
def process_yaml(data, parent_key='', parent_val=''):
    # Store the result dictionary
    result = {}

    # Accumulate parent values from non-dictionary entries
    accumulated_parent_value = parent_val

    # Flag to check if there is any nested dictionary
    has_nested_dict = False

    # Iterate over the dictionary entries
    # First, process the values that are not dictionaries
    for key, value in data.items():
        if not isinstance(value, dict):
            # Accumulate non-dictionary values (for parent key)
            accumulated_parent_value += f" {value}"

    # Then, process the values that are dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            # Set flag if nested dictionary is found
            has_nested_dict = True
            # Recurse deeper for dictionary values
            result.update(process_yaml(value, f"{parent_key}_{key}" if parent_key else key, accumulated_parent_value))

    # Only add to the result if we are at a leaf node (no nested dictionaries)
    if not has_nested_dict and parent_key:
        result[parent_key] = accumulated_parent_value.strip()

    return result

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: parseyaml.py <yaml_file>")
        sys.exit(1)

    # 1st argument is the YAML file path
    yaml_file = sys.argv[1]

    try:
        # Read the YAML file
        with open(yaml_file, 'r') as file:
            yaml_string = file.read()

        # Load the YAML string into a Python dictionary
        data = yaml.safe_load(yaml_string)
        # data = yaml.load(yaml_string, Loader=yaml.FullLoader)
        print(json.dumps(data, indent=2))
        

        # Process the YAML data to flatten it into a dictionary
        result_dict = process_yaml(data)

        # Print the resulting dictionary
        for key, value in result_dict.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
