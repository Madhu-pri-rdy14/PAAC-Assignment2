import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os

from data import QuantumDataset
from model import DensityMatrixReconstructor

def calculate_fidelity(rho_pred, rho_true):
    """
    Calculates Quantum Fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
    For 1 qubit, we can use a simpler approach or direct matrix algebra.
    Here we compute it explicitly for the batch.
    """
    fidelities = []
    pred_np = rho_pred.detach().cpu().numpy()
    true_np = rho_true.detach().cpu().numpy()
    
    for i in range(len(pred_np)):
        p = pred_np[i]
        t = true_np[i]
        try:
            evals, evecs = np.linalg.eigh(p)
            
            evals = np.clip(evals, 0, None) 
            sqrt_p = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
            
            term = sqrt_p @ t @ sqrt_p
            
            evals_term, evecs_term = np.linalg.eigh(term)
            evals_term = np.clip(evals_term, 0, None)
            sqrt_term = evecs_term @ np.diag(np.sqrt(evals_term)) @ evecs_term.conj().T
            
            f = np.real(np.trace(sqrt_term))**2
            fidelities.append(f)
        except np.linalg.LinAlgError:
            
            fidelities.append(0.0)

    return np.mean(fidelities)

def calculate_trace_distance(rho_pred, rho_true):
    """
    Trace distance T(rho, sigma) = 0.5 * Tr(|rho - sigma|)
    Where |A| = sqrt(A_dagger * A)
    """
    diff = (rho_pred - rho_true).detach().cpu().numpy()
    distances = []
    for i in range(len(diff)):
        d = diff[i]
        
        s = np.linalg.svd(d, compute_uv=False)
        distances.append(0.5 * np.sum(s))
    return np.mean(distances)

def train():
    
    N_QUBITS = 1
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    dataset = QuantumDataset(num_samples=2500, n_qubits=N_QUBITS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  
    model = DensityMatrixReconstructor(n_qubits=N_QUBITS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    criterion = nn.MSELoss() 
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_fidelity = 0
        total_trace_dist = 0
        batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            reconstructed_rho = model(data)
            
            
            loss = criterion(reconstructed_rho.real, target.real) + \
                   criterion(reconstructed_rho.imag, target.imag)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_fidelity += calculate_fidelity(reconstructed_rho, target)
            total_trace_dist += calculate_trace_distance(reconstructed_rho, target)
            batches += 1
            
        avg_loss = total_loss / batches
        avg_fid = total_fidelity / batches
        avg_td = total_trace_dist / batches
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Fidelity: {avg_fid:.4f} | Trace Dist: {avg_td:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.2f}s")
    
    output_path = os.path.join("outputs", "model_track1.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train()