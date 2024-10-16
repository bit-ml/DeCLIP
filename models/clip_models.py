from .clip import clip
import torch.nn as nn
import torch
import torch.nn.functional as F
import models.networks.customnet as customnetworks
from models.clip.model import ResidualAttentionBlock
import re

# Model for localisation
class CLIPModelLocalisation(nn.Module):
    def __init__(self, name, intermidiate_layer_output = None, decoder_type = "conv-4"):
        super(CLIPModelLocalisation, self).__init__()
        
        self.intermidiate_layer_output = intermidiate_layer_output
        self.decoder_type = decoder_type
        self.name = name # architecure
        
        if self.intermidiate_layer_output:
            assert "layer" in self.intermidiate_layer_output or "all" in self.intermidiate_layer_output or "xceptionnet" in self.intermidiate_layer_output

        self._set_backbone()
        self._set_decoder()

    def _set_backbone(self):    
        # Set up the backbone model architecture and parameters
        if self.name in ["RN50", "ViT-L/14"]:
            self.model, self.preprocess = clip.load(self.name, device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None))            
        elif self.name == "xceptionnet":
            # XceptionNet
            layername = 'block%d' % 2
            extra_output = 'block%d' % 1
            self.model = customnetworks.make_patch_xceptionnet(
                layername=layername, extra_output=extra_output, num_classes=2)
        # ViT+RN fusion
        elif "RN50" in self.name and "ViT-L/14" in self.name:
            name = self.name.split(",")
            model1, self.preprocess = clip.load(name[0], device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None)) 
            model2, self.preprocess = clip.load(name[1], device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None))            
            self.model = [model1.to("cuda"), model2.to("cuda")]
        
    def _set_decoder(self):
        # Set up decoder architecture
        upscaling_layers = []
        if "conv" in self.decoder_type:
            filter_sizes = self._get_conv_filter_sizes(self.name, self.intermidiate_layer_output, self.decoder_type)
            num_convs = int(re.search(r'\d{0,3}$', self.decoder_type).group())
            
            for i in range(1, len(filter_sizes)):
                upscaling_layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=5, padding=2))
                upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                upscaling_layers.append(nn.ReLU())
                for _ in range(num_convs//4 - 1):
                    upscaling_layers.append(nn.Conv2d(filter_sizes[i], filter_sizes[i], kernel_size=5, padding=2))
                    upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                    upscaling_layers.append(nn.ReLU())

                # skip some upscaling layers if the input is too large (case for CNNs)
                skip_upscaling = (
                    self.intermidiate_layer_output == "layer2" and i == 1
                    or self.intermidiate_layer_output == "layer1" and i <= 2
                    ) and ("RN50" in self.name or "xceptionnet" in self.name)
                if skip_upscaling:
                    continue

                upscaling_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

            # CNNs output may not be in (256, 256) - usually a (224, 224) size
            if "RN50" in self.name or "xceptionnet" in self.name:
                upscaling_layers.append(nn.Upsample(size=(256, 256), mode='bilinear'))

            upscaling_layers.append(nn.Conv2d(64, 1, kernel_size=5, padding=2))

        elif self.decoder_type == "linear":
            # Xceptionnet
            if self.name == "xceptionnet":
                upscaling_layers.append(nn.Linear(784, 1))
            # CLIP
            else:
                upscaling_layers.append(nn.Linear(1024, 1))

        elif self.decoder_type == "attention":
            transformer_width = 1024
            transformer_heads = transformer_width // 64
            attn_mask = self._build_attention_mask()
            self.att1 = ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask)
            self.att2 = ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask)
            upscaling_layers.append(nn.Linear(1024, 1))

        self.fc = nn.Sequential(*upscaling_layers)

    def _get_conv_filter_sizes(self, name, intermidiate_layer_output, decoder_type):
        assert "conv" in decoder_type

        if "RN50" in name and "ViT-L/14" in name:
            num_layers = len(name.split(","))
            return [1024*num_layers, 512, 256, 128, 64]
        elif "RN50" in name:
            if intermidiate_layer_output == "layer1":
                return [256, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer2":
                return [512, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer3":
                return [1024, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer4":
                return [2048, 512, 256, 128, 64]
        elif "xceptionnet" in name:
            return [256, 512, 256, 128, 64]
        else:
            return [1024, 512, 256, 128, 64]
    
    def _unify_linear_layer_outputs(self, linear_outputs):
        output = torch.cat(linear_outputs, dim=1)
        output = output.view(output.size()[0],  int(output.size()[1]**0.5), int(output.size()[1]**0.5))
        output = torch.nn.functional.interpolate(output.unsqueeze(1), size = (256, 256), mode = 'bicubic')
        return output

    # standard CLIPs method
    def _build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        context_length = 257
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def _feature_map_transform(self, input):
        output = input.permute(1, 2, 0)
        output = output.view(output.size()[0], output.size()[1], int(output.size()[2]**0.5), int(output.size()[2]**0.5))
        return output

    def feature_extraction(self, x):
        if self.name == "RN50" or self.name=="ViT-L/14":
            features = self.model.encode_image(x)
            if self.intermidiate_layer_output:
                features = features[self.intermidiate_layer_output]
            # choose the last layer
            else:
                if self.name == "RN50":
                    features = features["layer4"]
                else:
                    features = features["layer23"]
        # ViT+RN fusion
        elif "RN50" in self.name and "ViT-L/14" in self.name:
            # given ViT feature layer
            features_vit = self.model[0].encode_image(x)[self.intermidiate_layer_output]
            features_vit = self._feature_map_transform(features_vit[1:])
            # explicit RN50 3rd layer to match the feature dimension
            features_rn50 = self.model[1].encode_image(x)["layer3"]
            features_rn50 = F.interpolate(features_rn50, size=(16, 16), mode='bilinear', align_corners=False)
            features = torch.cat([features_vit, features_rn50], 1)
        # for xceptionnet
        else:
            features = self.model(x)
            features = features[0]
        
        return features
                
    def forward(self, x):
        # Feature extraction
        features = self.feature_extraction(x)
        
        # Forward step
        # ViT+RN fusion convolutional decoder
        if "RN50" in self.name and "ViT-L/14" in self.name and "conv" in self.decoder_type:
            output = self.fc(features)
        
        # Linear decoder
        elif self.decoder_type == "linear":
            # xceptionnet + linear
            if self.name == "xceptionnet":
                features = features.view(features.size()[0], features.size()[1], -1)
                features = features.permute(1, 0, 2)
                linear_outputs = [self.fc(input_part) for input_part in features[0:]]
            # CLIP + linear
            else:
                linear_outputs = [self.fc(input_part) for input_part in features[1:]]

            output = self._unify_linear_layer_outputs(linear_outputs)

        # Attention decoder
        elif self.decoder_type == "attention":
            features = self.att1(features)
            features = self.att2(features)
            linear_outputs = [self.fc(input_part) for input_part in features[1:]]
            output = self._unify_linear_layer_outputs(linear_outputs)

        # Convolutional decoder over RN
        elif "conv" in self.decoder_type and "RN50" == self.name:
            output = self.fc(features)

        # Convolutional decoder over ViT
        else:
            features = features[1:]
            output = self._feature_map_transform(features)
            output = self.fc(output)

        output = torch.flatten(output, start_dim =1)
        return output