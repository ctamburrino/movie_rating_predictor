import json
import os
import pickle

def create_index_file(input_file, output_file):
    """
    Creates an index file that maps line numbers to file positions.
    This allows for fast random access to specific lines in the JSON file.
    """
    print("Creating index file...")
    positions = []
    current_position = 0
    
    with open(input_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            positions.append(current_position)
            current_position += len(line)
    
    # Save the positions to a file
    with open(output_file, 'wb') as f:
        pickle.dump(positions, f)
    
    print(f"Index created with {len(positions)} entries")
    print(f"Index file saved to {output_file}")

if __name__ == "__main__":
    input_file = "Movies_and_TV.json"
    output_file = "Movies_and_TV.index"
    
    if os.path.exists(output_file):
        print(f"Index file {output_file} already exists. Delete it if you want to recreate it.")
    else:
        create_index_file(input_file, output_file) 