import csv
import json

# File paths
csv_file_path = "/home/perry/nlp2025/HippoRAG/reproduce/dataset/synthetic_knowledge_items.csv"
json_file_path = "/home/perry/nlp2025/HippoRAG/reproduce/dataset/itsupport_corpus.json"

def add_csv_data_to_json(csv_path, json_path):
    # Load the existing JSON data
    with open(json_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    # Read the CSV file and extract relevant columns
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Create a new entry for each row in the CSV
            new_entry = {
                "title": row["ki_topic"],
                "text": row["ki_text"],
                "idx": len(json_data)  # Assign a new index based on the current length of the JSON data
            }
            json_data.append(new_entry)

    # Write the updated JSON data back to the file
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)

# Execute the function
add_csv_data_to_json(csv_file_path, json_file_path)