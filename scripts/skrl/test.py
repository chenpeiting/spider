import torch
data = torch.load("teacher_data_23dim.pt", weights_only=True)
actions = data["act_seq"]
print(f"Teacher action range: [{actions.min():.2f}, {actions.max():.2f}]")
print(f"Teacher action std: {actions.std():.3f}")
print(f"Teacher action abs mean: {actions.abs().mean():.3f}")