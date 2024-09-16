import pandas as pd
import ast
import re

# Replace this with your file's path
input_file_path = './excel_files/FORTE_evaluated.xlsx'
output_file_path = './excel_files/FORTE_negationremoval.xlsx'

# Load the Excel file
xls = pd.ExcelFile(input_file_path)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(output_file_path, engine='openpyxl')

# Function to check if 'no' is in the list-like string
def contains_no(value):
    # This pattern matches 'no' as a complete word, allowing for specific non-word characters before it
    pattern = r'(^|\s|\>|\-)(no)(\s|$|[.,;!?])'

    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            # Safely evaluate the string as a list
            lst = ast.literal_eval(value)
            # Check if 'no' is in the list (case-insensitive)
            return any(re.search(pattern, item.lower()) for item in lst if isinstance(item, str))
        except (ValueError, SyntaxError):
            # If there's a ValueError or SyntaxError, it's not a list
            pass
    # Use regular expression to find 'no' as a standalone word or with specified characters before it
    return bool(re.search(pattern, value.lower()))


# Iterate over each sheet
for sheet_name in xls.sheet_names:
    print(f'Processing sheet "{sheet_name}"...')
    # Read each sheet
    df = xls.parse(sheet_name)
    
    # Filter out the rows where 'gt_degree' column contains 'no'
    df_filtered = df[~df['gt_degree'].astype(str).apply(contains_no)]
    
    # Filter out the rows where 'gt_degree' column contains 'no'
    df_filtered2 = df_filtered[~df_filtered['out_degree'].astype(str).apply(contains_no)]
    
    # Write the modified DataFrame to the new file
    df_filtered2.to_excel(writer, sheet_name=sheet_name, index=False)

# Save the new file
writer.close()
