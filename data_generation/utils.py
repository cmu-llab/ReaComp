import json

def write_jsonl(data, filename: str):
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def read_jsonl(filename: str):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data