import torch
import time
from model import DensityMatrixReconstructor
from data import random_density_matrix, simulate_measurement

def measure_latency():
    device = torch.device("cpu")
    model = DensityMatrixReconstructor(n_qubits=1).to(device)
    model.load_state_dict(torch.load("outputs/model_track1.pt"))
    model.eval()

    rho = random_density_matrix(1)
    data = simulate_measurement(rho, n_shots=50) 
    input_tensor = torch.tensor(data).unsqueeze(0).to(device)

    for _ in range(10):
        _ = model(input_tensor)

    start_time = time.time()
    num_runs = 1000
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()

    avg_latency = (end_time - start_time) / num_runs
    print(f"Inference Latency: {avg_latency * 1000:.4f} ms per reconstruction")

if __name__ == "__main__":
    measure_latency()