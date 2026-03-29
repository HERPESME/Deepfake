import json
import matplotlib.pyplot as plt
import numpy as np

# Load Phase 5 results
with open('experiments/phase5_results/adversarial_results.json') as f:
    adv_data = json.load(f)

epsilons_str = ["0.0", "0.01", "0.02", "0.05", "0.1"]
epsilons = [float(e) for e in epsilons_str]
models = ["efficientnet_b0", "xception", "hybrid", "resnet50"]
labels = {"efficientnet_b0":"EfficientNet-B0", "xception":"XceptionNet", "hybrid":"Hybrid CNN-Trans", "resnet50":"ResNet50"}
colors = {"efficientnet_b0":"red", "xception":"blue", "hybrid":"purple", "resnet50":"orange"}

plt.figure(figsize=(8, 5))
for m in models:
    accs = []
    for eps in epsilons_str:
        # PGD accuracies
        acc = adv_data[m]['pgd'][eps]['accuracy']
        accs.append(acc * 100)
    plt.plot(epsilons, accs, marker='o', label=labels[m], color=colors[m], linewidth=2)

plt.title('Adversarial Robustness (PGD Attack)', fontsize=14)
plt.xlabel(r'Perturbation Strength ($\epsilon$)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('SpringConference_Package/figure2/adversarial_pgd.png', dpi=300)

