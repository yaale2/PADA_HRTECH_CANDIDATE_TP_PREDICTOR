import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_resumes, filter_resumes
from src.feature_engineering import extract_features, MAX_JOBS, EMB_DIM
from src.model import HybridAttritionModel

def cox_loss(preds, durations, events):
    order = torch.argsort(durations, descending=True)
    preds = preds[order]
    events = events[order]
    hazard = torch.exp(preds)
    log_risk = torch.log(torch.cumsum(hazard, dim=0))
    loss = -(preds - log_risk) * events
    return loss.sum() / (events.sum() + 1e-8)

def concordance_index(preds, durations, events):
    n = len(preds)
    concordant = 0
    permissible = 0
    for i in range(n):
        for j in range(i+1, n):
            if durations[i] != durations[j]:
                permissible += 1
                if (preds[i] < preds[j]) == (durations[i] > durations[j]):
                    concordant += 1
    return concordant / permissible if permissible > 0 else 0.5

def train_model(json_path, epochs=25, lr=1e-3, test_size=0.2):
    # Load data
    data = load_resumes(json_path)
    data = filter_resumes(data, min_jobs=2)
    
    # Extract features
    X_seq, X_num, T, E = extract_features(data)
    
    # Normalize
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    
    # Train-test split
    X_seq_train, X_seq_test, X_num_train, X_num_test, T_train, T_test, E_train, E_test = train_test_split(
        X_seq, X_num, T, E, test_size=test_size, random_state=42
    )
    
    # Prepare tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
    X_num_train = torch.tensor(X_num_train, dtype=torch.float32).to(device)
    T_train = torch.tensor(T_train, dtype=torch.float32).to(device)
    E_train = torch.tensor(E_train, dtype=torch.float32).to(device)
    
    X_seq_test_t = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_num_test_t = torch.tensor(X_num_test, dtype=torch.float32).to(device)
    
    # Model
    model = HybridAttritionModel(EMB_DIM, X_num.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    for epoch in range(epochs):
        model.train()
        preds = model(X_seq_train, X_num_train).squeeze()
        loss = cox_loss(preds, T_train, E_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_test = model(X_seq_test_t, X_num_test_t).cpu().numpy().flatten()
    
    c_index = concordance_index(preds_test, T_test, E_test)
    print(f"\nFinal C-index: {c_index:.4f}")
    
    return model, {'c_index': c_index, 'scaler': scaler}
