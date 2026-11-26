"""Write CSV and JSON files through temporary files in the same directory."""

import json


def save_csv(data, path, **kwargs):
    temporary = path.with_name(f".{path.name}.partial")
    data.to_csv(temporary, **kwargs)
    temporary.replace(path)


def save_json(data, path):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
