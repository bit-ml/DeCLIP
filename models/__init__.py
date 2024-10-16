from .clip_models import CLIPModelLocalisation


VALID_NAMES = [
    'CLIP:RN50',  
    'CLIP:ViT-L/14',
    'CLIP:xceptionnet',
    'CLIP:ViT-L/14,RN50', 
]

def get_model(opt):
    name, layer, decoder_type = opt.arch, opt.feature_layer, opt.decoder_type

    assert name in VALID_NAMES

    return CLIPModelLocalisation(name.split(':')[1], intermidiate_layer_output = layer, decoder_type=decoder_type) 
    
