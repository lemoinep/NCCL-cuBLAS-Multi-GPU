import json

with open("topology_mapping.json") as f:
    data = json.load(f)

print("Num GPUs:", data["num_gpus"])
print("GPUs:")
for g in data["gpus"]:
    print(g)

print("Rank mapping:")
for rm in data["rank_mapping"]:
    print(f"rank {rm['rank']} -> GPU {rm['gpu_id']}")