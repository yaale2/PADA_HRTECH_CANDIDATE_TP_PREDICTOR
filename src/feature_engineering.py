import re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MAX_JOBS = 10
EMB_DIM = 384

def parse_period(period_str):
    if not period_str:
        return None
    years = re.findall(r'20\d{2}', str(period_str))
    if len(years) >= 2:
        return (int(years[-1]) - int(years[0])) * 12
    elif len(years) == 1:
        return (2025 - int(years[0])) * 12
    return None

def extract_features(resumes):
    model_bert = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    sequences = []
    num_features = []
    durations = []
    events = []
    
    for resume in tqdm(resumes):
        jobs = resume.get('jobs', [])
        
        if len(jobs) < 2:
            continue
        
        job_texts = []
        job_durations = []
        
        for job in jobs:
            dur = parse_period(job.get('period'))
            if dur:
                job_texts.append(str(job.get('position', '')))
                job_durations.append(dur)
        
        if len(job_durations) < 2:
            continue
        
        past_texts = job_texts[:-1]
        past_durations = job_durations[:-1]
        target_duration = job_durations[-1]
        
        emb = model_bert.encode(past_texts)
        
        if len(emb) < MAX_JOBS:
            pad = np.zeros((MAX_JOBS - len(emb), EMB_DIM))
            emb = np.vstack([emb, pad])
        else:
            emb = emb[-MAX_JOBS:]
        
        sequences.append(emb)
        
        num_features.append([
            np.mean(past_durations),
            len(past_durations)
        ])
        
        durations.append(target_duration)
        
        last_period = str(jobs[-1].get('period', '')).lower()
        events.append(0 if 'настоящее' in last_period else 1)
    
    X_seq = np.array(sequences)
    X_num = np.array(num_features)
    T = np.array(durations)
    E = np.array(events)
    
    return X_seq, X_num, T, E
