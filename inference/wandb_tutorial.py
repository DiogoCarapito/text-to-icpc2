import torch
from safetensors.torch import load_file
import wandb

# Define your model architecture (example)
class MyModel(torch.nn.Module):
    def __init__(self, input_size=10, output_size=10):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 128)
        self.layer2 = torch.nn.Linear(128, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

# Initialize W&B
run = wandb.init(project="text-to-icpc2")

# Correctly specify the artifact reference
artifact = run.use_artifact('diogo-carapito/wandb-registry-model/text-to-icpc2:v1', type='model')

# Download the artifact
artifact_dir = artifact.download()

# Load the .safetensors file
model_path = f"{artifact_dir}/model.safetensors"
state_dict = load_file(model_path)

# Initialize the model and load the state dict
model = MyModel(input_size=10, output_size=10)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Prepare the input data (example)
input_data = torch.randn(1, 10)  # Example input tensor

# Perform inference
with torch.no_grad():
    output = model(input_data)

# Get the top 5 results
topk_values, topk_indices = torch.topk(output, k=5, dim=1)

# Print the top 5 results
print("Top 5 values:", topk_values)
print("Top 5 indices:", topk_indices)