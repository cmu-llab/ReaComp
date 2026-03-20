import json

def write_jsonl(data, filename: str):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")