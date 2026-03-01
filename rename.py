import os
folder_path = r"C:\Users\User\Desktop\Projects\Face recognition\dataset\Muhammad Nashib Sajjad"
person_name = "Muhammad Nashib Sajjad"  # change this

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Sort files to make sure numbering is consistent
files.sort()

# Rename files in a loop
for idx, file in enumerate(files, start=1):
    # Get the file extension
    ext = os.path.splitext(file)[1]
    
    # Create new filename
    new_name = f"{person_name}_{idx}{ext}"
    
    # Full paths
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Renaming completed!")