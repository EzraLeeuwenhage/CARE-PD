import torch
from model import FlowMatchingVectorField
from dataloader import get_h36m_edge_index
import numpy as np


def save_generated_to_npz(generated_tensor, output_filename="generated_PD_walk.npz"):
    """
    Reverses the GNN formatting and saves the generated skeleton 
    into a CARE-PD compatible .npz archive.
    """
    print(f"Original generated tensor shape: {generated_tensor.shape}")
    
    # 1. Remove the batch dimension -> Shape becomes (3, 60, 17)
    tensor_sq = generated_tensor.squeeze(0)
    
    # 2. Permute from (C, T, V) back to (T, V, C) -> Shape becomes (60, 17, 3)
    tensor_permuted = tensor_sq.permute(1, 2, 0)
    
    # 3. Detach from graph and convert to a standard NumPy array
    numpy_array = tensor_permuted.cpu().detach().numpy()
    
    # 4. Save to .npz using a mock sequence ID key that fits their naming convention
    # We use "003__" to pretend it belongs to the patient we trained on
    sequence_key = "003__generated_walk_01"
    
    # np.savez takes keyword arguments to name the arrays inside the archive
    save_dict = {sequence_key: numpy_array}
    np.savez(output_filename, **save_dict)
    
    print(f"Successfully saved to: {output_filename}")
    print(f"Array shape inside .npz: {numpy_array.shape} under key: '{sequence_key}'")


def euler_ode_solver(model, edge_index, num_steps=100, device="cpu"):
    """
    Starts from pure noise and iteratively applies the model's predicted 
    velocity to generate a 60-frame walking sequence.
    """
    model.eval() # Set model to evaluation mode
    
    # 1. Define the shapes based on our training data setup
    N, C, T_frames, V = 1, 3, 60, 17 
    
    # 2. Start with pure Gaussian noise (x_0) at time t=0
    x = torch.randn(N, C, T_frames, V, device=device)
    
    # 3. Calculate the step size (dt)
    dt = 1.0 / num_steps
    
    print(f"Starting Euler ODE solver with {num_steps} steps...")
    
    # We don't need to track gradients during generation
    with torch.no_grad():
        for step in range(num_steps):
            # Current time t
            t_val = step * dt
            
            # Create a time tensor of shape (Batch,)
            t_tensor = torch.full((N,), t_val, device=device)
            
            # 4. Ask the model for the velocity at the current position and time
            velocity = model(x, t_tensor, edge_index)
            
            # 5. Take a small step in the direction of the velocity
            x = x + velocity * dt
            
            # Optional: Print progress every 20 steps
            if step % 20 == 0:
                print(f"Step {step}/{num_steps} (t={t_val:.2f}) completed.")
                
    print("Generation complete! Reached t=1.0")
    
    return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Re-initialize the architecture (from Step 3)
    model = FlowMatchingVectorField().to(device)
    edge_index = get_h36m_edge_index().to(device)
    
    # 2. Load the trained weights we just saved
    model.load_state_dict(torch.load("flow_matching_stgcn.pth", map_location=device))
    
    # 3. Generate a new sequence
    generated_skeleton = euler_ode_solver(model, edge_index, num_steps=100, device=device)
    
    print(f"Final generated tensor shape: {generated_skeleton.shape}")

    # 4. Save the generated sequence to a .npz file
    save_generated_to_npz(generated_skeleton, output_filename="generated_PD_walk.npz")