#GITHUB REPO LINK: https://github.com/UmarRasheed007/mlp-hidden-layer-study#

import os
import numpy as np
import textwrap
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import pearsonr

# defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = False

RND = 42
np.random.seed(RND)

# ------------- Config -------------
widths = [4, 8, 16, 32, 64, 128]
epochs = 40
n_samples = 1200
n_features = 20
n_classes = 3
output_dir = "mlp_width_outputs"
os.makedirs(output_dir, exist_ok=True)

# ------------- Dataset -------------
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_informative=10, n_redundant=5, n_classes=n_classes,
                           class_sep=1.2, random_state=RND)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=RND, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------- Training loop -------------
results = {}
classes = np.unique(y_train)

for w in widths:
    print(f"Training width={w}")
    clf = MLPClassifier(hidden_layer_sizes=(w,), activation='relu', solver='adam',
                        learning_rate_init=0.001, max_iter=1, warm_start=True, random_state=RND)
    train_acc = []
    test_acc = []
    losses = []
    clf.partial_fit(X_train, y_train, classes=classes)
    train_acc.append(np.mean(clf.predict(X_train) == y_train))
    test_acc.append(np.mean(clf.predict(X_test) == y_test))
    losses.append(clf.loss_ if hasattr(clf, "loss_") else np.nan)
    for ep in range(1, epochs):
        clf.partial_fit(X_train, y_train)
        train_acc.append(np.mean(clf.predict(X_train) == y_train))
        test_acc.append(np.mean(clf.predict(X_test) == y_test))
        losses.append(clf.loss_ if hasattr(clf, "loss_") else np.nan)
    results[w] = {"train_acc": np.array(train_acc),
                  "test_acc": np.array(test_acc),
                  "loss": np.array(losses)}

with open(os.path.join(output_dir, "mlp_width_results.pkl"), "wb") as f:
    pickle.dump(results, f)

# ------------- Plots -------------
figure_paths = []
colors = plt.cm.tab10(np.linspace(0, 1, len(widths)))

for i, w in enumerate(widths):
    epochs_range = np.arange(1, epochs+1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    
    # Accuracy plot
    ax[0].plot(epochs_range, results[w]["train_acc"], 'o-', linewidth=2, 
               markersize=4, label="Train", color='steelblue', alpha=0.8)
    ax[0].plot(epochs_range, results[w]["test_acc"], 's-', linewidth=2, 
               markersize=4, label="Test", color='coral', alpha=0.8)
    ax[0].set_title(f'Accuracy (Width = {w})', fontsize=12, fontweight='bold')
    ax[0].set_xlabel('Epoch', fontsize=11)
    ax[0].set_ylabel('Accuracy', fontsize=11)
    ax[0].legend(loc='lower right', fontsize=10)
    ax[0].grid(True, alpha=0.3, linestyle='--')
    ax[0].set_ylim([0, 1.05])
    
    # Loss plot
    ax[1].semilogy(epochs_range, results[w]["loss"], 'o-', linewidth=2, 
                   markersize=4, color='darkgreen', alpha=0.8)
    ax[1].set_title(f'Training Loss (Width = {w})', fontsize=12, fontweight='bold')
    ax[1].set_xlabel('Epoch', fontsize=11)
    ax[1].set_ylabel('Loss (log scale)', fontsize=11)
    ax[1].grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    p = os.path.join(output_dir, f"acc_loss_w{w}.png")
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    figure_paths.append(p)


# ------------- Additional Analysis Plots -------------


# 1. Class Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
unique, counts = np.unique(y_train, return_counts=True)
bars = ax.bar(unique, counts, color=colors[:len(unique)], edgecolor='black', 
              linewidth=1.5, alpha=0.8, width=0.6)
ax.set_xlabel('Class Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Training Set Class Distribution', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_xticks(unique)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
p = os.path.join(output_dir, "class_distribution.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# 2. Feature Histograms
fig, axes = plt.subplots(4, 5, figsize=(14, 10), dpi=100)
axes = axes.flatten()
for i in range(n_features):
    axes[i].hist(X_train[:, i], bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Feature {i}', fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=8)
    axes[i].set_ylabel('Frequency', fontsize=8)
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')
    axes[i].tick_params(labelsize=7)

plt.suptitle('Feature Distributions (Training Set)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
p = os.path.join(output_dir, "feature_histograms.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# 3. Feature Box Plots
fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
bp = ax.boxplot([X_train[:, i] for i in range(n_features)], labels=[f'F{i}' for i in range(n_features)],
                 patch_artist=True, widths=0.6, notch=False)
for patch, color in zip(bp['boxes'], plt.cm.tab20(np.linspace(0, 1, n_features))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.2)
ax.set_ylabel('Standardized Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Box Plots (Training Set)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
p = os.path.join(output_dir, "feature_boxplots.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# 4. Correlation Heatmap
corr_matrix = np.corrcoef(X_train.T)
fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(np.arange(n_features))
ax.set_yticks(np.arange(n_features))
ax.set_xticklabels([f'F{i}' for i in range(n_features)], fontsize=8)
ax.set_yticklabels([f'F{i}' for i in range(n_features)], fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation', fontsize=10, fontweight='bold')
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
p = os.path.join(output_dir, "correlation_heatmap.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# 5. Train-Test Scatter Plot (PCA projection)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                           cmap='tab10', s=30, alpha=0.6, label='Train', edgecolors='black', linewidth=0.5)
scatter_test = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                          cmap='tab10', s=60, alpha=0.8, label='Test', marker='^', 
                          edgecolors='darkred', linewidth=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12, fontweight='bold')
ax.set_title('Train-Test Data Distribution (PCA Projection)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
p = os.path.join(output_dir, "train_test_scatter.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)



# Summary: final accuracy with error bars
final_acc = [results[w]["test_acc"][-1] for w in widths]
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
bars = ax.bar([str(w) for w in widths], final_acc, color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Hidden Layer Width', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Final Test Accuracy by Hidden Layer Width', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.05])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
p = os.path.join(output_dir, "final_accuracy_summary.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# ------------- Dataset Summary Table/Diagram -------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.axis('off')

# Dataset info table
dataset_info = [
    ['Metric', 'Value'],
    ['Total Samples', f'{n_samples}'],
    ['Training Samples', f'{len(X_train)}'],
    ['Test Samples', f'{len(X_test)}'],
    ['Number of Features', f'{n_features}'],
    ['Number of Classes', f'{n_classes}'],
    ['Test Size Ratio', '0.2 (20%)'],
    ['Preprocessing', 'StandardScaler'],
    ['Class Distribution', 'Stratified'],
    ['Random Seed', f'{RND}']
]

table = ax.table(cellText=dataset_info, cellLoc='center', loc='center',
                colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header styling
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(dataset_info)):
    for j in range(2):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

plt.title('Dataset Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
p = os.path.join(output_dir, "dataset_summary.png")
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
figure_paths.append(p)

# ------------- PDF Report (multipage) -------------
pdf_path = os.path.join(output_dir, "MLP_width_tutorial.pdf")
pp = PdfPages(pdf_path)

def add_text_page(title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')
    plt.text(0.02, 0.96, title, fontsize=16, weight='bold')
    y = 0.92
    for para in paragraphs:
        wrapped = textwrap.fill(para, 90)
        plt.text(0.02, y, wrapped, fontsize=10, va='top', family='serif')
        y -= 0.08 * (wrapped.count('\n') + 3)
        if y < 0.08:
            break
    pp.savefig(fig)
    plt.close(fig)

title = "How Hidden Layer Width Influences the Performance of an MLP"
intro = ("This tutorial studies how the number of neurons in a single hidden layer "
         "affects learning, generalisation and training behaviour. We test widths: " + ", ".join(map(str,widths)) + ".")
background = ("Key ideas: width controls model capacity. Small widths lead to underfitting, "
              "very large widths often lead to overfitting. The universal approximation theorem "
              "guarantees representational ability with sufficient width but says nothing about generalisation.")
method = ("Experimental setup: synthetic classification dataset (sklearn.make_classification), "
          f"{n_samples} examples, {n_features} features, {n_classes} classes. StandardScaler applied. "
          f"Each MLP trained for {epochs} epochs with Adam optimizer (lr=0.001). Metrics: train/test accuracy and training loss.")
results_text = ("See figures: per-width accuracy & loss curves, and a summary final-accuracy bar chart. "
                "Typical finding: medium widths (e.g. 32–64) often provide optimal trade-off; very wide layers "
                "achieve low training loss but exhibit larger train-test gap (overfitting).")
conclusion = ("Practical rules: start with modest widths, increase only if underfitting detected, apply L2 regularisation "
              "for large widths, consider depth vs width trade-offs for optimal architecture design.")
refs = ("References: Hornik et al. (1989); Lu et al. (2017); Montúfar et al. (2014); Jacot et al. (2018).")

add_text_page(title, [intro, background, method, results_text, conclusion, refs])

for p in figure_paths:
    img = plt.imread(p)
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.imshow(img)
    plt.axis('off')
    pp.savefig(fig)
    plt.close(fig)

pp.close()
print("All done. Outputs in:", output_dir)
print("Open:", pdf_path)
