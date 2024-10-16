from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
# import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # d_model => (token) embed_dim
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # Yup, pytorch's forward for MultideadAttention expects arguments (query, key, value, ...)
        # query => [L, N, E]; key and value => [S, N, E]
        # L: target dim; S: source dim; E: (token) embedding dim; N: batch
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=False)

    def forward(self, x: torch.Tensor):
        attention_res = self.attention(self.ln_1(x))
        x, weights = x + attention_res[0], attention_res[1]
        # x => attn_output => shape = [L, N, E]
        # weights => attn_output_weights => shape = [layers, N, heads, L, S]
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        weights_all_blocks = []

        # Go through all the blocks (layers)
        for block in self.resblocks:
            x, weight = block(x)
            weights_all_blocks.append(weight)

        return x, torch.stack(weights_all_blocks)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.get_cls = False

    def forward(self, x: torch.Tensor):
        # The conv1 uses kernel_size=patch_size, stride=patch_size, therefore stride==kernel_size and
        # that will do the equivalent of chopping the image into patches and converting those into "tokens"
        # with a depth (number of output layers / channels in the conv1) equals to "width".
        # Note1: grid => input_resolution/patch_size
        # Note2: width => d_model => (token) embedding dim
        x = self.conv1(x)  # shape = [*, width, grid, grid] or [batch, embedding_dim, grid, grid]

        # Reshape it to flatten the grid (it will look more like a "sentence" made of "tokens")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid**2] or [batch, embedding_dim, grid**2]

        # Swap axis so each token (patch) leads to its own learned high dim embedding "juice"
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] or [batch, grid**2, embedding_dim]

        # Concatenate an extra token (learned) that will represent the class (it will become the first token, it's the [CLS] for BERT (?) stuff)
        # Finally, we get to what is usual for transformers with a final shape = [batch, sentence_length, embedding_dim]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, 1 + grid**2, width]

        # Add to the previous tensor (it's not concat now!) the positional embeddings (learned)
        x = x + self.positional_embedding.to(x.dtype)

        # The normalization layer learns a gain and a bias during training,
        # but it also "uses statistics computed from input data (x) in both training and evaluation modes".
        x = self.ln_pre(x)

        # Pytorch transformer stuff expects the batch as the second dimension
        x = x.permute(1, 0, 2)  # shape: [N, L, E] -> [L, N, E]

        x, weights = self.transformer(x)
        # x => attn_output => shape = [L, N, E]
        # weights => attn_output_weights => shape = [layers, N, heads, L, S]
        # N: batch
        # L = S: 1 + grid**2
        # E: width or (token) embedding dim

        # Undo the mess with the dimensions as explained...
        x = x.permute(1, 0, 2)  # shape: [L, N, E] -> [N, L, E]

        # At this point we have x.shape = [*, 1 + grid ** 2, width] or [batch, sentence_length, embedding_dim]
        # but the line below will keep only the class embedding token (learned!), the index=0 below,
        # and normalize it (as before, the normalization layer learns a gain and a bias...)
        x = self.ln_post(x[:, 0, :])  # shape = [*, width] or [batch, emdedding_dim]
        # Explanation about the CLS from the ViT creators:
        # https://github.com/google-research/vision_transformer/issues/61#issuecomment-802233921
        # https://github.com/google-research/vision_transformer/issues/83#issuecomment-805661088
        # "After training the embedding is fixed and it will be exactly the same for all inputs 
        # in the first layer - but due to interactions with the other tokens in every layer, the 
        # value will be input-dependent (and strongly correlate with the input's class) at the output layer."

        # self.proj.shape = [width, output_dim]
        if self.get_cls:
            return x

        x = x @ self.proj  # Project the output from the transformer (the CLS token only) into the choosen output dimension
                           # The operation is [batch, width] x [width, output_dim] = [batch, output_dim]
                           # CLS embeddings, "x", has size "width" and (for the available models) it's always bigger than the output (output_dim).
                           # Therefore we are compressing "x" into the output (smaller number of dimensions).

        return x, weights


class CLIP(nn.Module):
    def __init__(self,
                 # output dimension (embeddings generated by text and image encoders)
                 output_embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=self.vision_heads,
            output_dim=output_embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, output_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # Lazily create the causal attention mask.
        # Pytorch uses additive attention mask (it adds the mask inside the softmax, remember the exponentials!),
        # therefore we fill it with "-inf" and make the lower diagonal part all zeros.
        # Note1: This is used only for the text encoder. For the visual transformer we want
        # full attention between the vision tokens (non-causal) because the image is 2D
        # and we only use a 1D thing like for text to reuse the transformer architecture...
        # Note2: A system is called "causal" when its output depends only on present and past inputs.
        # For text, the "sentence" is seen as a time-series, so the positions are usually referred to as T ("time").
        # That's why we apply this weird mask and it will have the effect that all 
        # elements ABOVE the main diagonal will be zeroed.
        #    t0  t1  t2  t3
        # t0 w11 0   0   0   => At t0, you can only use t <= t0
        # t1 w21 w22 0   0   => ...
        # t2 w31 w32 w33 0   => ...
        # t3 w41 w42 w43 w45 => At t3, the final row, you can use everything because <= t3
        # Apparently it's "cheaper" to use a full matrix and at the end chop off half of it...
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, get_cls=False):
        self.visual.get_cls = get_cls
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # Here text is the input "tokenized" text (clip.tokenize). 
        # It has shape = [batch, context_length]
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, context_length, transformer_width]
                                                         # transformer_width => d_model => token embeddings
                                                         # context_length => sentence_length

        # A clear difference between this and the vision one is the lack of a CLS token.
        x = x + self.positional_embedding.type(self.dtype)

        # Pytorch transformer stuff expects the batch as the second dimension
        x = x.permute(1, 0, 2)  # shape: [N, L, E] -> [L, N, E]

        x, weights = self.transformer(x)
        # x => attn_output => shape = [L, N, E]
        # weights => attn_output_weights => shape = [layers, N, heads, L, S]
        # N: batch
        # L = S: context_length
        # E: transformer_width or (token) embedding_dim

        # Undo the mess with the dimensions as explained...
        x = x.permute(1, 0, 2)  # shape: [L, N, E] -> [N, L, E]

        # x.shape = [batch_size, context_length, transformer_width]
        x = self.ln_final(x).type(self.dtype)

        # Take features ONLY from the eot embedding. The eot_token is the token with the highest value (index) in each sequence.
        # Although we are only using that token, it went through all the transformer layers,
        # so it carries lots if information (but I don't know why they chose exactly the last one)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1), :] @ self.text_projection  # I added the ":" for clarity...
        # The operation is [batch_size, transformer_width] x [transformer_width, output_embed_dim] = [batch_size, output_embed_dim]

        return x, weights

    def forward(self, image, text):
        image_features, _ = self.encode_image(image)
        text_features, _ = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_features(state_dict: dict, name: str = "", fp16: bool = True):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise ValueError("Modified to work only with ViT image encoders...")

    output_embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        output_embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if fp16:
        convert_weights(model)

    model.load_state_dict(state_dict)

    print(f"Model stats for {name}")
    print(f"- output_embed_dim: {output_embed_dim}")
    print(f"- vision_width: {vision_width}")
    print(f"- vision_layers: {vision_layers}")
    print(f"- vision_patch_size: {vision_patch_size}")
    print(f"- vision_heads: {model.vision_heads}")
    print(f"- grid_size: {grid_size}")
    print(f"- image_resolution: {image_resolution}")
    print(f"- context_length: {context_length}")
    print(f"- vocab_size: {vocab_size}")
    print(f"- transformer_width: {transformer_width}")
    print(f"- transformer_heads: {transformer_heads}")
    print(f"- transformer_layers: {transformer_layers}")
    print(f"- total number of parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    return model.eval()