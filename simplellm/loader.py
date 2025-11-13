import torch
from pathlib import Path
from safetensors.torch import load_file
from simplellm.config import get_model_config
from simplellm.model import Qwen3Model
class Qwen3ModelLoader:
    def __init__(self,model_path="",load_model=True):
        # assert "0.6B" in model_path
        self.model_path = Path(model_path).resolve()
        self.config = get_model_config("0.6B")
        self.model = Qwen3Model(self.config)
        print(f"model:{self.model}")
        with torch.no_grad():
            res = self.model(torch.tensor([1, 2, 3]).unsqueeze(0))
        print(f"warmup res shape: {res.shape}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params:,}")
        total_params_normalized = total_params - self.model.tok_emb.weight.numel()
        print(f"Total number of unique parameters: {total_params_normalized:,}")
        print(f"float32 (PyTorch default): {self.get_model_memory_size(self.model, input_dtype=torch.float32):.2f} GB")
        print(f"bfloat16: {self.get_model_memory_size(self.model, input_dtype=torch.bfloat16):.2f} GB")
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.npu.is_available():
                device = torch.device("npu:0")
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
        print("use device:",device)
        self.device = device
        self.model.to(device)
        if load_model:
            self.load_model()
        self.tokenizer=None

    @staticmethod
    def get_model_memory_size(model, input_dtype=torch.float32):
        total_params = 0
        total_grads = 0
        for param in model.parameters():
            # Calculate total number of elements per parameter
            param_size = param.numel()
            total_params += param_size
            # Check if gradients are stored for this parameter
            if param.requires_grad:
                total_grads += param_size

        # Calculate buffer size (non-parameters that require memory)
        total_buffers = sum(buf.numel() for buf in model.buffers())

        # Size in bytes = (Number of elements) * (Size of each element in bytes)
        # We assume parameters and gradients are stored in the same type as input dtype
        element_size = torch.tensor(0, dtype=input_dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

        # Convert bytes to gigabytes
        total_memory_gb = total_memory_bytes / (1024 ** 3)

        return total_memory_gb
    def load_model(self):
        weights_file = self.model_path/"model.safetensors"
        params = load_file(weights_file)
        model = self.model
        def assign(left, right, tensor_name="unknown"):
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

            with torch.no_grad():
                if isinstance(right, torch.Tensor):
                    left.copy_(right)
                else:
                    left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

            return left

        model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"],
                                      "model.embed_tokens.weight")

        for l in range(self.config["n_layers"]):
            block = model.trf_blocks[l]
            att = block.att

            # Q, K, V projections
            att.W_query.weight = assign(
                att.W_query.weight,
                params[f"model.layers.{l}.self_attn.q_proj.weight"],
                f"model.layers.{l}.self_attn.q_proj.weight"
            )
            att.W_key.weight = assign(
                att.W_key.weight,
                params[f"model.layers.{l}.self_attn.k_proj.weight"],
                f"model.layers.{l}.self_attn.k_proj.weight"
            )
            att.W_value.weight = assign(
                att.W_value.weight,
                params[f"model.layers.{l}.self_attn.v_proj.weight"],
                f"model.layers.{l}.self_attn.v_proj.weight"
            )

            # Output projection
            att.out_proj.weight = assign(
                att.out_proj.weight,
                params[f"model.layers.{l}.self_attn.o_proj.weight"],
                f"model.layers.{l}.self_attn.o_proj.weight"
            )

            # QK norms
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.scale = assign(
                    att.q_norm.scale,
                    params[f"model.layers.{l}.self_attn.q_norm.weight"],
                    f"model.layers.{l}.self_attn.q_norm.weight"
                )
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.scale = assign(
                    att.k_norm.scale,
                    params[f"model.layers.{l}.self_attn.k_norm.weight"],
                    f"model.layers.{l}.self_attn.k_norm.weight"
                )

            # Attention layernorm
            block.norm1.scale = assign(
                block.norm1.scale,
                params[f"model.layers.{l}.input_layernorm.weight"],
                f"model.layers.{l}.input_layernorm.weight"
            )

            # Feedforward weights
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )
            block.norm2.scale = assign(
                block.norm2.scale,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
                f"model.layers.{l}.post_attention_layernorm.weight"
            )

        # Final normalization and output head
        model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

        if "lm_head.weight" in params:
            model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.out_head.weight = model.tok_emb.weight
            print("Model uses weight tying.")
        self.model.to(self.device)
        # del params