import pandas as pd

def get_first_column(csv_file_path):
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file_path)
        
        # Get the name of the first column
        first_column_name = df.columns[0]
        
        # Get the first column independent of its name
        first_column = df[first_column_name].tolist()
        return first_column, first_column_name
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None

def print_first_column(texts, column_name, num_rows=10):
    if texts is None:
        return
    
    # Ensure num_rows does not exceed the size of the list
    num_rows = min(num_rows, len(texts))
    
    print(f"Showing {num_rows} rows of column '{column_name}':")
    print("-" * 50)
    
    for i, text in enumerate(texts[:num_rows], 1):
        print(f"Row {i}:")
        print(text)
        print("-" * 50)

if __name__ == "__main__":
    # CSV file path
    csv_file_path = "csv/test10.csv"
    
    # Get all texts and the name of the column
    texts, column_name = get_first_column(csv_file_path)
    
    # Print the first 5 texts as an example
    print_first_column(texts, column_name, 5)
