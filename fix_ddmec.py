"""Script to fix ddmec_1d.py with all necessary changes."""

import re

# Read the file
with open('ddmec_1d.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Change SimpleMLP to SmallMLP in import
content = content.replace('from train_ddpm_1d import DDPM1D, DDPMConfig, SimpleMLP',
                         'from train_ddpm_1d import DDPM1D, DDPMConfig, SmallMLP')

# Fix 2: Change all alpha_bar to alphas_cumprod
content = re.sub(r'\.alpha_bar\[', '.alphas_cumprod[', content)
content = re.sub(r'\.alpha_bar(\s)', r'.alphas_cumprod\1', content)

# Fix 3: Change .alpha[ to .alphas[
content = re.sub(r'\.alpha\[', '.alphas[', content)

# Fix 4: Change .beta[ to .betas[
content = re.sub(r'\.beta\[', '.betas[', content)

# Fix 5: Add ConditionalMLP class after imports, before DDMEC1D class
conditional_mlp_class = '''

class ConditionalMLP(nn.Module):
    """Wrapper to make SmallMLP conditional by concatenating condition."""
    
    def __init__(self, base_model: SmallMLP):
        super().__init__()
        self.base_model = base_model
        # New input layer that takes [x, condition, time_emb]
        self.input_proj = nn.Linear(2 + 32, 64)  # 2 inputs + 32 time_emb -> 64
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, 1] - noisy input
        t: [B] - timestep
        condition: [B, 1] - conditioning variable (optional)
        """
        # Get time embedding from base model
        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.base_model.time_embed(t_norm)
        
        if condition is None:
            # Unconditional: use zeros
            condition = torch.zeros_like(x)
        
        # Concatenate x and condition, then with time embedding
        inp = torch.cat([x, condition, t_emb], dim=-1)
        h = self.input_proj(inp)
        h = torch.relu(h)
        
        # Use rest of base model network
        h = self.base_model.net[2](h)  # Second Linear layer
        h = self.base_model.net[3](h)  # SiLU
        h = self.base_model.net[4](h)  # Output layer
        
        return h

'''

# Find the position to insert ConditionalMLP (before class DDMEC1D)
if 'class ConditionalMLP' not in content:
    class_pos = content.find('class DDMEC1D:')
    if class_pos != -1:
        content = content[:class_pos] + conditional_mlp_class + content[class_pos:]

# Fix 6: Add wrapping code in _load_model
# Find the _load_model function and add wrapping after load_state_dict
pattern = r'(ddpm\.model\.load_state_dict\(checkpoint\["model_state_dict"\]\))\s*(ddpm\.model\.to\(self\.device\))'
replacement = r'\1\n        \n        # Wrap with conditional wrapper\n        ddpm.model = ConditionalMLP(ddpm.model).to(self.device)'

if 'Wrap with conditional wrapper' not in content:
    content = re.sub(pattern, replacement, content)

# Write the fixed file
with open('ddmec_1d.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed ddmec_1d.py successfully!")
print("Changes made:")
print("  - Fixed SimpleMLP -> SmallMLP")
print("  - Fixed alpha_bar -> alphas_cumprod")
print("  - Fixed alpha -> alphas")
print("  - Fixed beta -> betas")
print("  - Added ConditionalMLP class")
print("  - Added model wrapping in _load_model")

