import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from thesis.src.model import FlowMatchingMLP
from thesis.src.dataloader import get_dataloader
from thesis.src.generate_prior import generate_prior_from_prefix

CONFIG_PATH = "thesis/configs/dataloader.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(model, dataloader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    epochs = cfg['training']['epochs']
    loss_weight = cfg['training']['loss_weight']
    
    save_path = Path(cfg['training']['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0
        
        for _, (prefix, target, severity) in enumerate(pbar):
            optimizer.zero_grad()
            
            # move data to device
            batch_size = severity.shape[0]
            prefix = {k: v.to(device) for k, v in prefix.items()}
            target = {k: v.to(device) for k, v in target.items()}
            severity = severity.to(device)
            
            # sample prior and time step
            x_0 = generate_prior_from_prefix(prefix, target)
            t = torch.rand(batch_size, 1).to(device)
            
            # construct x_t using linear interpolation
            t_pose = t.view(batch_size, 1, 1, 1)
            t_trans = t.view(batch_size, 1, 1)
            
            x_t = {
                'pose': (1 - t_pose) * x_0['pose'] + t_pose * target['pose'],
                'trans': (1 - t_trans) * x_0['trans'] + t_trans * target['trans']
            }
            
            # compute target velocity u_true
            u_true = {
                'pose': target['pose'] - x_0['pose'],
                'trans': target['trans'] - x_0['trans']
            }
            
            # predict velocity field and compute (weighted) loss
            u_pred = model(x_t, prefix, t, severity)

            # TODO: Replace nn.functional.mse_loss with Geodesic Loss for pose
            loss_pose = nn.functional.mse_loss(u_pred['pose'], u_true['pose'])
            loss_trans = nn.functional.mse_loss(u_pred['trans'], u_true['trans'])
            loss = ((1.0 - loss_weight) * loss_pose) + (loss_weight * loss_trans)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Calculate average loss over epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % cfg['training']['log_interval'] == 0:
            print(f"Epoch {epoch+1:04d}/{epochs} | Avg Loss: {avg_epoch_loss:.6f}")
            
    # save model params
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")

if __name__ == "__main__":
    cfg = load_config()
    dataloader = get_dataloader(config_path=CONFIG_PATH)

    # init model, run training loop
    model = FlowMatchingMLP(config_path=CONFIG_PATH)
    train(model, dataloader, cfg)