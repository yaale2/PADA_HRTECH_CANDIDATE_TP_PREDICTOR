import json
import numpy as np

def load_resumes(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total resumes: {len(data)}")
    return data

def filter_resumes(data, min_jobs=2):
    filtered = [r for r in data if len(r.get('jobs', [])) >= min_jobs]
    print(f"Filtered: {len(filtered)} resumes (min {min_jobs} jobs)")
    return filtered
