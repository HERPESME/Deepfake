"""
Generate architecture diagrams for 4 novel deepfake detection models.
Style inspired by 'Attention is All You Need' paper figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color palette (pastel, academic style)
COLORS = {
    'input':    '#FFD6D6',  # light pink
    'backbone': '#FFE0B2',  # light orange
    'conv':     '#B3E5FC',  # light blue
    'attn':     '#C8E6C9',  # light green
    'norm':     '#E1BEE7',  # light purple
    'pool':     '#FFF9C4',  # light yellow
    'fc':       '#D1C4E9',  # lavender
    'output':   '#FFCCBC',  # light coral
    'fft':      '#B2DFDB',  # light teal
    'concat':   '#F0F4C3',  # light lime
    'vote':     '#BBDEFB',  # pastel blue
    'stack':    '#C5CAE9',  # pastel indigo
    'group_bg': '#F5F5F5',  # very light gray
    'group_bd': '#BDBDBD',  # gray border
}

def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False, text_color='#333333'):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#555555', linewidth=1.0,
                         zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=4,
            fontfamily='sans-serif')
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', style='->', lw=1.2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)

def draw_group_box(ax, x, y, w, h, label=None, label_side='left'):
    """Draw a grouping rectangle (like the Nx block in Transformer)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=COLORS['group_bg'], edgecolor=COLORS['group_bd'],
                         linewidth=1.5, linestyle='-', zorder=1)
    ax.add_patch(box)
    if label:
        lx = x - w/2 - 0.15 if label_side == 'left' else x + w/2 + 0.15
        ax.text(lx, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='#666666', zorder=4)

def draw_brace_label(ax, x, y, text, fontsize=8):
    """Draw a label at a position."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='#444444', fontfamily='sans-serif', style='italic')

# ============================================================
# DIAGRAM 1: Hybrid CNN-Transformer
# ============================================================
def draw_hybrid(ax):
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Hybrid CNN-Transformer\n(30.7M params)', fontsize=10, fontweight='bold', pad=10)
    
    W = 1.6; H = 0.38
    
    # Input
    draw_box(ax, 0, 0, W, H, 'Input Image (224×224×3)', COLORS['input'], fontsize=7.5)
    
    # CNN Backbone
    draw_group_box(ax, 0, 1.5, W+0.3, 1.5, label=None)
    draw_box(ax, 0, 1.0, W, H, 'EfficientNet-B0\nBackbone', COLORS['backbone'], fontsize=7.5)
    draw_box(ax, 0, 1.9, W, H, 'Feature Map\n(7×7×1280)', COLORS['conv'], fontsize=7.5)
    draw_arrow(ax, 0, 0.19, 0, 0.81)
    draw_arrow(ax, 0, 1.19, 0, 1.71)
    ax.text(-1.15, 1.5, 'CNN\nStage', ha='center', va='center', fontsize=7, color='#888', style='italic')
    
    # Patch Embedding
    draw_box(ax, 0, 2.7, W, H, 'Patch Embedding\n+ Positional Enc.', COLORS['norm'], fontsize=7.5)
    draw_arrow(ax, 0, 2.09, 0, 2.51)
    
    # CLS Token
    draw_box(ax, 0, 3.5, W, H, '[CLS] Token\nConcatenation', COLORS['concat'], fontsize=7.5)
    draw_arrow(ax, 0, 2.89, 0, 3.31)
    
    # Transformer block
    draw_group_box(ax, 0, 5.0, W+0.3, 2.2, label='6×')
    draw_box(ax, 0, 4.2, W, H, 'Multi-Head\nSelf-Attention (8H)', COLORS['attn'], fontsize=7.5)
    draw_box(ax, 0, 4.9, W, H, 'Add & Norm', COLORS['norm'], fontsize=7)
    draw_box(ax, 0, 5.55, W, H, 'Feed Forward', COLORS['conv'], fontsize=7.5)
    draw_box(ax, 0, 6.2, W, H, 'Add & Norm', COLORS['norm'], fontsize=7)
    draw_arrow(ax, 0, 3.69, 0, 4.01)
    draw_arrow(ax, 0, 4.39, 0, 4.71)
    draw_arrow(ax, 0, 5.09, 0, 5.36)
    draw_arrow(ax, 0, 5.74, 0, 6.01)
    
    # Classification
    draw_box(ax, 0, 7.0, W, H, 'Dropout + Linear\n(512→256→2)', COLORS['fc'], fontsize=7.5)
    draw_arrow(ax, 0, 6.39, 0, 6.81)
    
    # Output
    draw_box(ax, 0, 7.8, W*0.7, H, 'Real / Fake', COLORS['output'], fontsize=8, bold=True)
    draw_arrow(ax, 0, 7.19, 0, 7.61)


# ============================================================
# DIAGRAM 2: Multi-Scale Feature Fusion
# ============================================================
def draw_multiscale(ax):
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Multi-Scale Feature Fusion\n(4.2M params)', fontsize=10, fontweight='bold', pad=10)
    
    W = 1.2; H = 0.38
    
    # Input
    draw_box(ax, 0, 0, 1.6, H, 'Input Image (224×224×3)', COLORS['input'], fontsize=7.5)
    
    # Backbone
    draw_box(ax, 0, 0.8, 1.6, H, 'EfficientNet-B0\n(features_only)', COLORS['backbone'], fontsize=7.5)
    draw_arrow(ax, 0, 0.19, 0, 0.61)
    
    # Four scale branches
    scales = [
        (-1.5, 'Stage 1\n(56×56)', '#BBDEFB'),
        (-0.5, 'Stage 2\n(28×28)', '#B3E5FC'),
        (0.5, 'Stage 3\n(14×14)', '#80DEEA'),
        (1.5, 'Stage 4\n(7×7)', '#4DD0E1'),
    ]
    
    for sx, label, color in scales:
        draw_box(ax, sx, 1.8, W, H, label, color, fontsize=7)
        draw_arrow(ax, 0, 0.99, sx, 1.61)
    
    # 1x1 Conv + BN + ReLU
    for sx, _, _ in scales:
        draw_box(ax, sx, 2.7, W, H, 'Conv 1×1\n→ 256-d', COLORS['conv'], fontsize=7)
        draw_arrow(ax, sx, 1.99, sx, 2.51)
    
    # GAP
    for sx, _, _ in scales:
        draw_box(ax, sx, 3.5, W, H, 'Global Avg\nPooling', COLORS['pool'], fontsize=7)
        draw_arrow(ax, sx, 2.89, sx, 3.31)
    
    # Concatenation
    draw_box(ax, 0, 4.6, 1.8, H, 'Concatenate\n(1024-d)', COLORS['concat'], fontsize=7.5)
    for sx, _, _ in scales:
        draw_arrow(ax, sx, 3.69, 0, 4.41)
    
    # FC layers
    draw_box(ax, 0, 5.5, 1.6, H, 'Linear (1024→512)\n+ ReLU + Dropout', COLORS['fc'], fontsize=7.5)
    draw_arrow(ax, 0, 4.79, 0, 5.31)
    
    draw_box(ax, 0, 6.3, 1.6, H, 'Linear (512→2)', COLORS['fc'], fontsize=7.5)
    draw_arrow(ax, 0, 5.69, 0, 6.11)
    
    # Output
    draw_box(ax, 0, 7.1, 1.1, H, 'Real / Fake', COLORS['output'], fontsize=8, bold=True)
    draw_arrow(ax, 0, 6.49, 0, 6.91)


# ============================================================
# DIAGRAM 3: Ensemble Learning
# ============================================================
def draw_ensemble(ax):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Ensemble Learning\n(Soft Vote / Stacking)', fontsize=10, fontweight='bold', pad=10)
    
    W = 1.1; H = 0.38
    
    # Input
    draw_box(ax, 0, 0, 1.6, H, 'Input Image (224×224×3)', COLORS['input'], fontsize=7.5)
    
    # Model branches
    models = [
        (-1.8, 'EfficientNet\n-B0', COLORS['backbone']),
        (-0.9, 'Xception\nNet', COLORS['conv']),
        (0.0, 'Hybrid\nCNN-T', COLORS['attn']),
        (0.9, 'Multi-Scale\nFusion', COLORS['pool']),
        (1.8, 'ViT', COLORS['norm']),
    ]
    
    for mx, label, color in models:
        draw_box(ax, mx, 1.5, W, 0.5, label, color, fontsize=6.5)
        draw_arrow(ax, 0, 0.19, mx, 1.25)
    
    # Predictions
    for mx, _, _ in models:
        draw_box(ax, mx, 2.7, W, H, 'p(fake|x)', COLORS['fc'], fontsize=7)
        draw_arrow(ax, mx, 1.75, mx, 2.51)
    
    # Aggregation methods (grouped)
    draw_group_box(ax, 0, 4.5, 4.2, 2.8)
    ax.text(-2.3, 4.5, 'Aggregation\nStrategies', ha='center', va='center', fontsize=7, color='#888', style='italic')
    
    draw_box(ax, 0, 3.5, 2.0, H, 'Hard Voting\nmode({I[pm > 0.5]})', COLORS['vote'], fontsize=7)
    draw_box(ax, 0, 4.4, 2.0, H, 'Soft Voting\n1/M * Sum pm(y|x)', COLORS['vote'], fontsize=7)
    draw_box(ax, 0, 5.3, 2.0, H, 'Learned Stacking\nsigma(Sum wm * pm + b)', COLORS['stack'], fontsize=7)
    
    for mx, _, _ in models:
        draw_arrow(ax, mx, 2.89, 0, 3.31)
    draw_arrow(ax, 0, 3.69, 0, 4.21)
    draw_arrow(ax, 0, 4.59, 0, 5.11)
    
    # Final prediction
    draw_box(ax, 0, 6.3, 1.6, H, 'Ensemble\nPrediction', COLORS['concat'], fontsize=7.5)
    draw_arrow(ax, 0, 5.49, 0, 6.11)
    
    # Output
    draw_box(ax, 0, 7.1, 1.1, H, 'Real / Fake', COLORS['output'], fontsize=8, bold=True)
    draw_arrow(ax, 0, 6.49, 0, 6.91)


# ============================================================
# DIAGRAM 4: Frequency-Aware Dual-Branch CNN
# ============================================================
def draw_frequency(ax):
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(d) Frequency-Aware Dual-Branch\n(5.5M params)', fontsize=10, fontweight='bold', pad=10)
    
    W = 1.5; H = 0.38
    
    # Input
    draw_box(ax, 0, 0, 1.6, H, 'Input Image (224×224×3)', COLORS['input'], fontsize=7.5)
    
    # Two branches
    SX = -1.0  # spatial branch x
    FX = 1.0   # frequency branch x
    BW = 1.4
    
    # Branch labels
    ax.text(SX, 0.6, 'Spatial Branch', ha='center', va='center', fontsize=7.5, fontweight='bold', color='#555')
    ax.text(FX, 0.6, 'Frequency Branch', ha='center', va='center', fontsize=7.5, fontweight='bold', color='#555')
    
    # Spatial branch
    draw_group_box(ax, SX, 1.8, BW+0.2, 1.7)
    draw_box(ax, SX, 1.2, BW, H, 'EfficientNet-B0\nBackbone', COLORS['backbone'], fontsize=7)
    draw_box(ax, SX, 2.0, BW, H, 'Linear (1280→256)\n+ BatchNorm', COLORS['fc'], fontsize=7)
    draw_arrow(ax, -0.3, 0.19, SX, 1.01)
    draw_arrow(ax, SX, 1.39, SX, 1.81)
    ax.text(SX, 2.75, 'v_spat (256-d)', ha='center', va='center', fontsize=7, color='#666', style='italic')
    
    # Frequency branch
    draw_group_box(ax, FX, 2.4, BW+0.2, 3.0)
    draw_box(ax, FX, 1.2, BW, H, '2D FFT\n+ fftshift', COLORS['fft'], fontsize=7)
    draw_box(ax, FX, 2.0, BW, H, 'Log-Magnitude\nSpectrum', COLORS['fft'], fontsize=7)
    draw_box(ax, FX, 2.8, BW, H, '5-Layer CNN\n(3→32→...→256)', COLORS['conv'], fontsize=7)
    draw_box(ax, FX, 3.6, BW, H, 'GAP + Linear\n+ BatchNorm', COLORS['fc'], fontsize=7)
    draw_arrow(ax, 0.3, 0.19, FX, 1.01)
    draw_arrow(ax, FX, 1.39, FX, 1.81)
    draw_arrow(ax, FX, 2.19, FX, 2.61)
    draw_arrow(ax, FX, 2.99, FX, 3.41)
    ax.text(FX, 4.25, 'v_freq (256-d)', ha='center', va='center', fontsize=7, color='#666', style='italic')
    
    # Concatenation
    draw_box(ax, 0, 5.0, 1.8, H, 'Concatenate\n[v_spat || v_freq] (512-d)', COLORS['concat'], fontsize=7)
    draw_arrow(ax, SX, 2.19, 0, 4.81)
    draw_arrow(ax, FX, 3.79, 0, 4.81)
    
    # Fusion MLP
    draw_group_box(ax, 0, 6.1, 1.8+0.2, 1.5)
    draw_box(ax, 0, 5.7, 1.6, H, 'Linear (512→256)\n+ ReLU + Dropout', COLORS['fc'], fontsize=7)
    draw_box(ax, 0, 6.4, 1.6, H, 'Linear (256→128)\n+ ReLU + Dropout', COLORS['fc'], fontsize=7)
    draw_arrow(ax, 0, 5.19, 0, 5.51)
    draw_arrow(ax, 0, 5.89, 0, 6.21)
    ax.text(-1.25, 6.1, 'Fusion\nMLP', ha='center', va='center', fontsize=7, color='#888', style='italic')
    
    draw_box(ax, 0, 7.1, 1.6, H, 'Linear (128→2)', COLORS['fc'], fontsize=7)
    draw_arrow(ax, 0, 6.59, 0, 6.91)
    
    # Output
    draw_box(ax, 0, 7.8, 1.1, H, 'Real / Fake', COLORS['output'], fontsize=8, bold=True)
    draw_arrow(ax, 0, 7.29, 0, 7.61)


# ============================================================
# MAIN: Generate 2x2 layout
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 16))

draw_hybrid(axes[0, 0])
draw_multiscale(axes[0, 1])
draw_ensemble(axes[1, 0])
draw_frequency(axes[1, 1])

plt.subplots_adjust(wspace=0.15, hspace=0.20, top=0.97, bottom=0.02, left=0.02, right=0.98)
plt.savefig('papers/bridging_the_gap/figures/architecture_diagrams.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Architecture diagrams generated successfully!")
