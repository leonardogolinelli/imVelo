import os

# Define the directory where the files are located
directory = "matrix_folder"

# Loop through all files in the specified directory
for filename in os.listdir(directory):
    # Check if "lsvelo" is part of the filename
    if "lsvelo" in filename:
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        try:
            # Check if it's a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Skipping non-file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

print("Deletion of 'lsvelo' files complete.")
