import os

labels_dir = "C:/Users/thuta/Downloads/Uni/Senior Project/Image Conversion/Roboflow/dice/test/labels"

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(labels_dir, file)

    new_lines = []

    with open(path, "r") as f:
        for line in f.readlines():
            parts = line.split()
            parts[0] = "1"   # force change class id 
            new_lines.append(" ".join(parts) + "\n")

    with open(path, "w") as f:
        f.writelines(new_lines)