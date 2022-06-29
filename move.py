import os

for folder in ["train", "test", "val"]:
    for file in os.listdir(os.path.join("data", folder, "images")):
        
        filename = file.split(".")[0]+".json"
        existing_filepath = os.path.join("data", "labels", filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join("data", folder, "labels", file.split(".")[0]+".json")
            os.replace(existing_filepath, new_filepath)