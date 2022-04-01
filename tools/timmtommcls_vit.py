from numpy import load
from os.path import join as join
import torch

def np2th(weights, conv=False,cls=False,mult=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    if cls:
        weights = weights.transpose([1,0])
    if mult:
        weights = weights.transpose([1,2,0])
    return torch.from_numpy(weights)

checkpoint = load('ViT-B_16-224.npz')
new_sd = {}
print(type(checkpoint))
for k in checkpoint:
    if k == 'cls':
        new_sd['backbone.cls_token']=np2th(checkpoint[k])
    if k == 'embedding/bias':
        new_sd['backbone.patch_embed.projection.bias']=np2th(checkpoint[k])
    if k == 'embedding/kernel':
        new_sd['backbone.patch_embed.projection.weight']=np2th(checkpoint[k],conv=True)
    if k == 'head/bias':
        new_sd['head.layers.head.bias']=np2th(checkpoint[k])
    if k == 'head/kernel':
        new_sd['head.layers.head.weight']=np2th(checkpoint[k],cls=True)
    if k == 'Transformer/encoder_norm/scale':
        new_sd['backbone.ln1.weight']=np2th(checkpoint[k])
    if k == 'Transformer/encoder_norm/bias':
        new_sd['backbone.ln1.bias']=np2th(checkpoint[k])
    if k == 'Transformer/posembed_input/pos_embedding':
        print(checkpoint[k].shape)
        new_sd['backbone.pos_embed']=np2th(checkpoint[k])
for i in range(12):
    for k in checkpoint:
        if k == f'Transformer/encoderblock_{i}/LayerNorm_0/bias':
            new_sd[f'backbone.layers.{i}.ln1.bias']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/LayerNorm_2/bias':
            new_sd[f'backbone.layers.{i}.ln2.bias']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/LayerNorm_0/scale':
            new_sd[f'backbone.layers.{i}.ln1.weight']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/LayerNorm_2/scale':
            new_sd[f'backbone.layers.{i}.ln2.weight']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias':
            new_sd[f'backbone.layers.{i}.ffn.layers.0.0.bias']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel':
            new_sd[f'backbone.layers.{i}.ffn.layers.0.0.weight']=np2th(checkpoint[k],cls=True)
        if k == f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias':
            new_sd[f'backbone.layers.{i}.ffn.layers.1.bias']=np2th(checkpoint[k])
        if k == f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel':
            new_sd[f'backbone.layers.{i}.ffn.layers.1.weight']=np2th(checkpoint[k],cls=True)
    for name in ['bias','kernel']:
        if name == 'bias':
            q = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/{name}']).view(-1)
            k = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/{name}']).view(-1)
            v = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/{name}']).view(-1)
            new_sd[f'backbone.layers.{i}.attn.proj.bias']=np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
            cat = torch.cat([q, k, v], dim=0)
            new_sd[f'backbone.layers.{i}.attn.qkv.{name}'] = cat
        if name == 'kernel':
            q = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/{name}']).view(-1,768).t()
            k = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/{name}']).view(-1,768).t()
            v = np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/{name}']).view(-1,768).t()
            new_sd[f'backbone.layers.{i}.attn.proj.weight']=np2th(checkpoint[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel']).view(-1,768).t()
            cat = torch.cat([q, k, v], dim=0)
            new_sd[f'backbone.layers.{i}.attn.qkv.weight'] = cat

for k, v in new_sd.items():
    print(k,v.size(),sep="  ")

torch.save({'state_dict': new_sd}, 'timm_models/vit-b.pth')
#print(type(data))

#for item in lst:
 #   print(item,data[item].shape,sep=" ")