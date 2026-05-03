
import json 
from gptqmodel.utils.eval import EVAL
import torch  # Import PyTorch for GPU memory management
from lm_eval import evaluator



batch_size = 2  # Adjust based on your GPU memory
MODEL_NAME="DeepSeek-MoE-16B-Chat-gptq-8bit"
MODEL_PATH="/opt/models/DeepSeek-MoE-16B-Chat-gptq-8bit"


# Define evaluation parameters, list inspired by this leaderboard: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/
tasks_list = ["arc_challenge", "gpqa", "ifeval", "gpqa", "mmlu_pro"]  # Benchmark dataset





print(f"#######################")
print(f"#### model name: {MODEL_NAME} ####")
print(f"#### model path: {MODEL_PATH} ####")
print(f"#######################")

print(f"#######################")
print(f"#### tasks list: {', '.join(tasks_list)} ####")
print(f"#######################")

# Run evaluation
results = evaluator.simple_evaluate(
    model="hf",  # Hugging Face model
    cache_requests=False,
    # model_args=f"pretrained={model_path},dtype=bfloat16,load_in_8bit=True",
    model_args=f"pretrained={MODEL_PATH}",
    tasks=tasks_list, 
    batch_size=batch_size,
    device="cuda:0" 
)

# Extract results
results = results['results']
json_string = json.dumps(results, indent=4)

# Save to a file
json_filename = f"llm-eval-{MODEL_NAME}.json"
with open(json_filename, "w") as file:
    file.write(json_string)

print(f"### Results saved to {json_filename} ###")

# 🧹 Clear GPU Memory
del results
torch.cuda.empty_cache()
torch.cuda.synchronize()  # Ensures all GPU processes are finished before the next iteration

print(f"### GPU memory cleared for next iteration ###")
