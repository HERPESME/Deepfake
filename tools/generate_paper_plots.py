import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('experiments/phase5_results/adversarial_results.json') as f:
    adv_data = json.load(f)

epsilons_str = ["0.0", "0.01", "0.02", "0.05", "0.1"]
epsilons = [float(e) for e in epsilons_str]

all_models = ["efficientnet_b0", "xception", "resnet50", "vit", "hybrid", "multiscale", "frequency_aware"]
labels = {
    "efficientnet_b0": "EfficientNet-B0",
    "xception": "XceptionNet",
    "resnet50": "ResNet50",
    "vit": "ViT",
    "hybrid": "Hybrid CNN-Trans",
    "multiscale": "Multi-Scale",
    "frequency_aware": "Freq-Aware CNN",
}
colors = {
    "efficientnet_b0": "#e74c3c",
    "xception": "#3498db",
    "resnet50": "#e67e22",
    "vit": "#2ecc71",
    "hybrid": "#9b59b6",
    "multiscale": "#1abc9c",
    "frequency_aware": "#f39c12",
}
markers = {
    "efficientnet_b0": "o",
    "xception": "s",
    "resnet50": "D",
    "vit": "^",
    "hybrid": "v",
    "multiscale": "P",
    "frequency_aware": "X",
}

# --- PGD plot (all 7 models) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in all_models:
    accs = [adv_data[m]['pgd'][eps]['accuracy'] * 100 for eps in epsilons_str]
    ax.plot(epsilons, accs, marker=markers[m], label=labels[m], color=colors[m], linewidth=2, markersize=7)

ax.set_title('Adversarial Robustness under PGD Attack', fontsize=14, fontweight='bold')
ax.set_xlabel(r'Perturbation Budget $\epsilon$ ($L_\infty$)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(-5, 105)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=9, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig('SpringConference_Package/figure2/adversarial_pgd.png', dpi=300, bbox_inches='tight')
plt.close()

# --- FGSM plot (all 7 models) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for m in all_models:
    accs = [adv_data[m]['fgsm'][eps]['accuracy'] * 100 for eps in epsilons_str]
    ax.plot(epsilons, accs, marker=markers[m], label=labels[m], color=colors[m], linewidth=2, markersize=7)

ax.set_title('Adversarial Robustness under FGSM Attack', fontsize=14, fontweight='bold')
ax.set_xlabel(r'Perturbation Budget $\epsilon$ ($L_\infty$)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(-5, 105)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=9, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig('SpringConference_Package/figure2/adversarial_fgsm.png', dpi=300, bbox_inches='tight')
plt.close()

print("Both plots generated successfully.")
