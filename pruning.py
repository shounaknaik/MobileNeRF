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


# Load the pre-trained model
checkpoint = torch.load("./ckpt.tar")

model.load_state_dict(checkpoint["network_fn_state_dict"])
model.eval()

def remove_char_at_index(input_str, index):
    if 0 <= index < len(input_str):
        return input_str[:index] + input_str[index + 1:]
    else:
        return "Index out of range"

def convert_to_code_form(name):
    name= f"model.{name}"
    split_name = name.split('.')[:-1]
    new_name=[]
    for ch in split_name:

        
        if ch.isdigit():
            new_ch = f"[{ch}]"
            new_name.append(new_ch)
        else:
            new_name.append(ch)
    new_string = '.'.join(new_name)

    indices=[]
    for i,c in enumerate(new_string):
        if c=="[":
            indices.append(i-1)

    offset=0
    for idx in indices:
        new_string=remove_char_at_index(new_string,idx-offset)
        offset+=1

    return new_string


parameters_to_prune = ()
for name,param in model.named_parameters():

    wanted_name = convert_to_code_form(name)

    if name.split('.')[-1]=='weight':
        parameters_to_prune+=((wanted_name,'weight'),)




parameters_to_prune=((model.head[0], 'weight'), (model.body[0].conv_block[0], 'weight'), 
(model.body[0].conv_block[1], 'weight'), (model.body[0].conv_block[3], 'weight'), (model.body[0].conv_block[4], 'weight'), 
(model.body[1].conv_block[0], 'weight'), (model.body[1].conv_block[1], 'weight'), (model.body[1].conv_block[3], 'weight'), 
(model.body[1].conv_block[4], 'weight'), (model.body[2].conv_block[0], 'weight'), (model.body[2].conv_block[1], 'weight'), 
(model.body[2].conv_block[3], 'weight'), (model.body[2].conv_block[4], 'weight'), (model.body[3].conv_block[0], 'weight'), 
(model.body[3].conv_block[1], 'weight'), (model.body[3].conv_block[3], 'weight'), (model.body[3].conv_block[4], 'weight'), 
(model.body[4].conv_block[0], 'weight'), (model.body[4].conv_block[1], 'weight'), (model.body[4].conv_block[3], 'weight'), 
(model.body[4].conv_block[4], 'weight'), (model.body[5].conv_block[0], 'weight'), (model.body[5].conv_block[1], 'weight'), 
(model.body[5].conv_block[3], 'weight'), (model.body[5].conv_block[4], 'weight'), (model.body[6].conv_block[0], 'weight'), 
(model.body[6].conv_block[1], 'weight'), (model.body[6].conv_block[3], 'weight'), (model.body[6].conv_block[4], 'weight'), 
(model.body[7].conv_block[0], 'weight'), (model.body[7].conv_block[1], 'weight'), (model.body[7].conv_block[3], 'weight'), 
(model.body[7].conv_block[4], 'weight'), (model.body[8].conv_block[0], 'weight'), (model.body[8].conv_block[1], 'weight'), 
(model.body[8].conv_block[3], 'weight'), (model.body[8].conv_block[4], 'weight'), (model.body[9].conv_block[0], 'weight'), 
(model.body[9].conv_block[1], 'weight'), (model.body[9].conv_block[3], 'weight'), (model.body[9].conv_block[4], 'weight'),
(model.body[10].conv_block[0], 'weight'), (model.body[10].conv_block[1], 'weight'), (model.body[10].conv_block[3], 'weight'), 
(model.body[10].conv_block[4], 'weight'), (model.body[11].conv_block[0], 'weight'), (model.body[11].conv_block[1], 'weight'),
(model.body[11].conv_block[3], 'weight'), (model.body[11].conv_block[4], 'weight'), (model.body[12].conv_block[0], 'weight'), 
(model.body[12].conv_block[1], 'weight'), (model.body[12].conv_block[3], 'weight'), (model.body[12].conv_block[4], 'weight'), 
(model.body[13].conv_block[0], 'weight'), (model.body[13].conv_block[1], 'weight'), (model.body[13].conv_block[3], 'weight'), 
(model.body[13].conv_block[4], 'weight'), (model.body[14].conv_block[0], 'weight'), (model.body[14].conv_block[1], 'weight'), 
(model.body[14].conv_block[3], 'weight'), (model.body[14].conv_block[4], 'weight'), (model.body[15].conv_block[0], 'weight'), 
(model.body[15].conv_block[1], 'weight'), (model.body[15].conv_block[3], 'weight'), (model.body[15].conv_block[4], 'weight'), 
(model.body[16].conv_block[0], 'weight'), (model.body[16].conv_block[1], 'weight'), (model.body[16].conv_block[3], 'weight'), 
(model.body[16].conv_block[4], 'weight'), (model.body[17].conv_block[0], 'weight'), (model.body[17].conv_block[1], 'weight'), 
(model.body[17].conv_block[3], 'weight'), (model.body[17].conv_block[4], 'weight'), (model.body[18].conv_block[0], 'weight'),
(model.body[18].conv_block[1], 'weight'), (model.body[18].conv_block[3], 'weight'), (model.body[18].conv_block[4], 'weight'), 
(model.body[19].conv_block[0], 'weight'), (model.body[19].conv_block[1], 'weight'), (model.body[19].conv_block[3], 'weight'), 
(model.body[19].conv_block[4], 'weight'), (model.body[20].conv_block[0], 'weight'), (model.body[20].conv_block[1], 'weight'), 
(model.body[20].conv_block[3], 'weight'), (model.body[20].conv_block[4], 'weight'), (model.body[21].conv_block[0], 'weight'), 
(model.body[21].conv_block[1], 'weight'), (model.body[21].conv_block[3], 'weight'), (model.body[21].conv_block[4], 'weight'), 
(model.body[22].conv_block[0], 'weight'), (model.body[22].conv_block[1], 'weight'), (model.body[22].conv_block[3], 'weight'), 
(model.body[22].conv_block[4], 'weight'), (model.body[23].conv_block[0], 'weight'), (model.body[23].conv_block[1], 'weight'),
(model.body[23].conv_block[3], 'weight'), (model.body[23].conv_block[4], 'weight'), (model.body[24].conv_block[0], 'weight'), 
(model.body[24].conv_block[1], 'weight'), (model.body[24].conv_block[3], 'weight'), (model.body[24].conv_block[4], 'weight'), 
(model.body[25].conv_block[0], 'weight'), (model.body[25].conv_block[1], 'weight'), (model.body[25].conv_block[3], 'weight'),
(model.body[25].conv_block[4], 'weight'), (model.body[26].conv_block[0], 'weight'), (model.body[26].conv_block[1], 'weight'), 
(model.body[26].conv_block[3], 'weight'), (model.body[26].conv_block[4], 'weight'), (model.body[27].conv_block[0], 'weight'), 
(model.body[27].conv_block[1], 'weight'), (model.body[27].conv_block[3], 'weight'), (model.body[27].conv_block[4], 'weight'), 
(model.body[28].conv_block[0], 'weight'), (model.body[28].conv_block[1], 'weight'), (model.body[28].conv_block[3], 'weight'))





print('Before Pruning')
# print(model.head[0].weight)



prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)


for idx in range(len(parameters_to_prune)):
    prune.remove(parameters_to_prune[idx][0],'weight')

print('After Pruning')


checkpoint["network_fn_state_dict"] = model.state_dict()
torch.save(checkpoint,'ckpt_pruned.tar')