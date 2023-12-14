import torch

from model.R2L import R2L  
from data import (
    select_and_load_dataset
)
from utils import get_rank
from os.path import join
import torch.quantization
import argparse
import torch.nn.utils.prune as prune

##Setup parameters for correct model loading

root_dir="dataset/nerf_synthetic"
scene= "lego"
dataset_type = 'Blender'
input_height=100
input_width =100
output_height =800
output_width =800
test_skip = 8
factor = 4
bd_factor = 0.75
llffhold = 8
ff = False
use_sr_module = True
camera_convention = 'openGL'
ndc = False
device = get_rank()
n_sample_per_ray =8 

netdepth = 60
netwidth = 256
activation_fn = 'gelu'
use_sr_module = True
num_conv_layers = 2
num_sr_blocks = 3
sr_kernel = (64,64,16)
dataset_type = 'Blender'
multires = 6 

args = argparse.ArgumentParser(description='Description of your script.')


# Add an optional argument with a default value
args.add_argument('--netdepth', default=netdepth)
args.add_argument('--netwidth', default=netwidth)
args.add_argument('--activation_fn', default=activation_fn)
args.add_argument('--use_sr_module', default=use_sr_module)

args.add_argument('--num_conv_layers', default=num_conv_layers)
args.add_argument('--num_sr_blocks', default=num_sr_blocks)
args.add_argument('--sr_kernel', default=sr_kernel)
args.add_argument('--dataset_type', default=dataset_type)
args.add_argument('--multires',default=multires)


# Parse the command-line arguments
args = args.parse_args()

embed_dim = 2*args.multires +1 
## R2L module is loaded and saved in MobileR2L
model = R2L(
            args,
            3*n_sample_per_ray*embed_dim,
            3
        )



# load data    
# dataset_info = select_and_load_dataset(
#     basedir=join(root_dir, scene),
#     dataset_type=dataset_type,
#     input_height=input_height,
#     input_width=input_width,
#     output_height=output_height,
#     output_width=output_width,
#     scene=scene,
#     test_skip=test_skip,
#     factor=factor,
#     bd_factor=bd_factor,
#     llffhold=llffhold,
#     ff=ff,
#     use_sr_module=use_sr_module,
#     camera_convention=camera_convention,
#     ndc=ndc,
#     device=device,
#     n_sample_per_ray= n_sample_per_ray
# )

# Load the pre-trained model
checkpoint = torch.load("./ckpt.tar")


# model = R2LEngine(dataset_info=dataset_info, logger=None, args=None)  # Instantiate your model
model.load_state_dict(checkpoint["network_fn_state_dict"])
model.eval()

print(model)
parameters_to_prune = ()
for name,param in model.named_parameters():
    # print(name)

    if name.split('.')[-1] == 'weight':
        parameters_to_prune += ((name,"weight"),)



# print(parameters_to_prune)
# print(model.head.0)
parameters_to_prune = (
    (getattr(model.head,'0'), 'weight'),
    (getattr(model.body,'1').conv_block[0],'weight')
    
)


print('Before Pruning')
print(model.head[0].weight)



prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# prune.remove(parameters_to_prune[0],'weight')
# prune.remove(parameters_to_prune[1],'weight')
print('After Pruning')
print(model.head[0].weight)

# print(list(model.head[0].named_parameters()))

checkpoint["network_fn_state_dict"] = model.state_dict()
torch.save(checkpoint,'ckpt_pruned.tar')