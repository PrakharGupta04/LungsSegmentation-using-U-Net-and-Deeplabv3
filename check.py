#things to do before running 
# 1.Create a Conda Environment
# conda create -n LungSeg python=3.10 -y
# conda activate LungSeg

# 2.inatell dependencies
# pip install torch torchvision torchaudio
# pip install tqdm numpy opencv-python albumentations scikit-learn pandas matplotlib timm monai



# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())  # should be True if GPU is available


import torch

print("PyTorch version:", torch.__version__)
print("CUDA available? ", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA detected")
