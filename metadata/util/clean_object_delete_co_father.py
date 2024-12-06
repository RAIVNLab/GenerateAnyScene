import json

def flatten_json(data, root_key="object.n.01"):
    flat_dict = {}

    def _flatten(node, key, parent_key, level):
        flat_dict[key] = {
            "definition": node["definition"],
            "hypernyms": [parent_key] if parent_key else [],
            "level": level
        }
        for hyponym_key, hyponym_node in node["hyponyms"].items():
            _flatten(hyponym_node, hyponym_key, key, level + 1)

    for key, node in data["hyponyms"].items():
        _flatten(node, key, root_key, data["level"]+1)

    return flat_dict



with open("object.n.01.json") as f:
    data = json.load(f)
    
    
flattened_json = flatten_json(data, "object.n.01")

with open("object_n_01_flattened.json", 'w') as f:
    json.dump(flattened_json, f, indent=4)
    
    
co_father_file = "/Users/roy/Desktop/comp-visgen/fathers_polysemous_words.json"
words_to_remove = []
with open(co_father_file) as f:
    co_fathers_words = json.load(f)
    for father in co_fathers_words:
        for child in co_fathers_words[father]:
            # print(co_fathers_words[father][child])
            for polysemous_word in co_fathers_words[father][child]:
                # print(polysemous_word)  
                words_to_remove.append(polysemous_word)
                
                
import json

def remove_words(data, words_to_remove):
    # Find words to remove and their hypernyms
    hypernym_map = {}
    for word in words_to_remove:
        if word in data:
            hypernym_map[word] = data[word]["hypernyms"]

    # Update the hypernyms and levels of nodes that have hypernyms in words_to_remove
    for key, value in data.items():
        new_hypernyms = []
        new_level = value["level"]
        for hypernym in value["hypernyms"]:
            if hypernym in hypernym_map:
                new_hypernyms.extend(hypernym_map[hypernym])
                new_level = min(new_level, data[hypernym]["level"]) if hypernym in data else new_level
            else:
                new_hypernyms.append(hypernym)
        value["hypernyms"] = list(set(new_hypernyms))
        value["level"] = new_level
    
    all_levels_updated = False
    while not all_levels_updated:
        all_levels_updated = True
        for key, value in data.items():
            if value["hypernyms"] == ["object.n.01"]:
                value["level"] = 3
            else:
                if value["level"] != data[value["hypernyms"][0]]["level"]+1:
                    all_levels_updated = False
                    value["level"] = data[value["hypernyms"][0]]["level"]+1

    # Remove the specified words
    for word in words_to_remove:
        if word in data:
            del data[word]

    return data

# Load the flattened JSON file
with open('object_n_01_flattened.json', 'r') as file:
    flattened_data = json.load(file)

# List of words to remove
# words_to_remove = ["minor_planet.n.01", "planet.n.01","star.n.01"]

# Remove specified words and update the JSON structure
updated_data = remove_words(flattened_data, words_to_remove)

# Save the updated JSON to a file
with open('object_n_01_flattened_updated.json', 'w') as file:
    json.dump(updated_data, file, indent=4)
    



import json
from collections import defaultdict

# Load the flattened JSON file
with open('object_n_01_flattened_updated.json', 'r') as file:
    data = json.load(file)

# Sort the entries by their level
sorted_entries = sorted(data.items(), key=lambda item: item[1]['level'])

# Create a nested dictionary structure to hold the tree
tree = {}
node_map = {}

# Function to insert a node into the tree
def insert_node(node_id, definition, level, hypernyms):
    node = {
        "definition": definition,
        "level": level,
        "hyponyms": {}
    }
    if node_id not in node_map:
        node_map[node_id] = node
    else:
        node_map[node_id].update(node)
    
    for hypernym in hypernyms:
        if hypernym in node_map:
            node_map[hypernym]["hyponyms"][node_id] = node_map[node_id]
        else:
            node_map[hypernym] = {
                "hyponyms": {
                    node_id: node_map[node_id]
                }
            }

# Insert all nodes into the tree
for node_id, node_data in sorted_entries:
    insert_node(node_id, node_data["definition"], node_data["level"], node_data["hypernyms"])

# Extract the root node
root = "object.n.01"
final_tree = node_map[root]

# Save the final hierarchical JSON to a file
with open('object.n.01.new.json', 'w') as file:
    json.dump(final_tree, file, indent=4)
