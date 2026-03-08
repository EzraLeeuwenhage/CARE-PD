import torch
import torch.nn as nn
import torch.optim as optim

from model import FlowMatchingVectorField
from dataloader import load_patient_walks, create_fixed_length_windows, get_h36m_edge_index


def save_model(model, filepath="flow_matching_stgcn.pth"):
    """Saves the model weights to disk."""
    torch.save(model.state_dict(), filepath)
    print(f"Model successfully saved to {filepath}")


def train_flow_matching(model, real_data, edge_index, epochs=1000, lr=1e-3):
    """
    Trains the Flow Matching vector field model on our single-patient data.
    """
    # Move everything to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    real_data = real_data.to(device)
    edge_index = edge_index.to(device)
    
    # Standard Adam optimizer and MSE loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    batch_size = real_data.shape[0]

    # 1. choose x_0
    # 2. interpolation: x_t = t * x_1 + (1-t) * x_0
    # 3. conditionen, hoe doe je sampling (ODE)
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Sample real data (x_1)
        # For this single-patient test, our entire dataset is one batch
        x_1 = real_data 
        
        # 2. Sample pure noise (x_0)
        x_0 = torch.randn_like(x_1) 
        
        # 3. Sample a random timestep 't' for every item in the batch
        # t is between 0 and 1. Shape: (Batch_size,)
        t = torch.rand(batch_size, device=device)
        
        # We need to reshape 't' so it can broadcast over (N, C, T, V)
        # Reshape to (N, 1, 1, 1)
        t_reshaped = t.view(batch_size, 1, 1, 1)
        
        # 4. Compute intermediate noisy data (x_t) via linear interpolation
        x_t = t_reshaped * x_1 + (1.0 - t_reshaped) * x_0
        
        # 5. Compute the true target velocity (u_t)
        target_velocity = x_1 - x_0
        
        # 6. Model prediction (v_theta)
        predicted_velocity = model(x_t, t, edge_index)
        
        # 7. Calculate Flow Matching Loss (MSE)
        loss = loss_fn(predicted_velocity, target_velocity)
        
        # 8. Backpropagation
        loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")
            
    print("Training complete!")
    return model


if __name__ == "__main__":
    npz_path = r"..\assets\datasets\h36m\PD-GaM\h36m_3d_world_floorXZZplus_30f_or_longer.npz"
    
    # Get walk data for patient 003
    patient_003_data = load_patient_walks(npz_path, patient_prefix="003")
    data = create_fixed_length_windows(
        patient_003_data, 
        window_size=60, 
        step_size=20
    )
    edge_index = get_h36m_edge_index()
    
    # init model
    model = FlowMatchingVectorField()

    # train model
    trained_model = train_flow_matching(model, data, edge_index, epochs=1000)

    # save model
    save_model(trained_model)