from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
import torch

# --- H100-friendly defaults ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # speeds up matmuls on Hopper


bits = 8
model_id=" /opt/models/models--deepseek-ai--deepseek-moe-16b-chat/snapshots/eefd8ac7e8dc90e095129fe1a537d5e236b2e57c/"
quant_path = f"DeepSeek-MoE-16B-Chat-gptq-{bits}bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"] # change to full dataset if your GPU has enough VRAM


quant_config = QuantizeConfig(bits=bits, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=4)

# --- Save ---
print(f"model savaed at: {quant_path}")
model.save(quant_path)

