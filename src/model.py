import torch
import torch.nn as nn

class DensityMatrixReconstructor(nn.Module):
    def __init__(self, n_qubits, embed_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits  
        
        self.embedding = nn.Linear(2, embed_dim) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_size = self.dim * self.dim * 2 
        self.fc_out = nn.Linear(embed_dim, self.output_size)

    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer(x)
        x = x.mean(dim=1) 

        raw_params = self.fc_out(x)
        
        raw_reshaped = raw_params.view(-1, 2, self.dim, self.dim)
        
        real_part = raw_reshaped[:, 0, :, :]
        imag_part = raw_reshaped[:, 1, :, :]
        
        
        L_real = torch.tril(real_part)
        L_imag = torch.tril(imag_part)
        
        L = torch.complex(L_real, L_imag)

        L_dagger = torch.transpose(L.conj(), 1, 2)
        unnormalized_rho = torch.bmm(L, L_dagger) 

        trace = torch.einsum('bii->b', unnormalized_rho).real.unsqueeze(1).unsqueeze(2)
        rho = unnormalized_rho / (trace + 1e-6)
        
        return rho