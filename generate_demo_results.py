"""
Script de génération des graphiques de résultats démonstratifs.
Simule les résultats typiques obtenus après entraînement complet.
Utile pour préparer la soutenance sans attendre l'entraînement GPU.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

os.makedirs('./results', exist_ok=True)

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid')

# ─────────────────────────────────────────────
# 1. Résultats finaux sur le jeu de test
# ─────────────────────────────────────────────
results = {
    'Modèle':       ['U-Net Standard', 'Attention U-Net', 'ResU-Net'],
    'Dice':         [0.8124, 0.8743, 0.8612],
    'IoU':          [0.6867, 0.7756, 0.7591],
    'Précision':    [0.8356, 0.8891, 0.8734],
    'Rappel':       [0.8012, 0.8612, 0.8501],
    'F1-Score':     [0.8124, 0.8743, 0.8612],
    'Accuracy':     [0.9712, 0.9821, 0.9798],
    'Paramètres':   ['31.0 M', '34.9 M', '32.6 M'],
}
df = pd.DataFrame(results).set_index('Modèle')

print('Tableau des résultats :')
print(df.to_string())
df.to_csv('./results/comparaison_modeles.csv')


# ─────────────────────────────────────────────
# 2. Simulation de l'historique d'entraînement
# ─────────────────────────────────────────────
def smooth(data, alpha=0.15):
    s = [data[0]]
    for x in data[1:]:
        s.append(alpha * x + (1 - alpha) * s[-1])
    return np.array(s)

epochs = np.arange(1, 51)


def make_history(final_dice, noise_std=0.015, convergence=25):
    """Génère des courbes d'entraînement réalistes."""
    decay     = np.exp(-epochs / convergence)
    train_l   = 0.55 * decay + 0.16 + np.random.normal(0, noise_std, 50)
    val_l     = 0.60 * decay + 0.18 + np.random.normal(0, noise_std + 0.005, 50)
    train_d   = final_dice + 0.08 - 0.12 * decay + np.random.normal(0, noise_std, 50)
    val_d     = final_dice + 0.00 - 0.14 * decay + np.random.normal(0, noise_std + 0.003, 50)
    train_iou = train_d * 0.86
    val_iou   = val_d   * 0.86
    return {
        'train_loss': smooth(np.clip(train_l, 0.12, 0.65)),
        'val_loss':   smooth(np.clip(val_l,   0.14, 0.70)),
        'train_dice': smooth(np.clip(train_d, 0.50, 0.98)),
        'val_dice':   smooth(np.clip(val_d,   0.45, 0.97)),
        'train_iou':  smooth(np.clip(train_iou, 0.40, 0.90)),
        'val_iou':    smooth(np.clip(val_iou,   0.35, 0.89)),
    }


hist_unet   = make_history(0.8124, noise_std=0.018, convergence=28)
hist_attn   = make_history(0.8743, noise_std=0.013, convergence=22)
hist_resunet= make_history(0.8612, noise_std=0.014, convergence=24)

histories = {
    'U-Net':           hist_unet,
    'Attention U-Net': hist_attn,
    'ResU-Net':        hist_resunet,
}
palette = ['#3498db', '#e74c3c', '#2ecc71']


# ─────────────────────────────────────────────
# 3. Courbes d'entraînement
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Courbes d'Entraînement — Comparaison des 3 Modèles", fontsize=14, fontweight='bold')

titles_train = ['Loss — Entraînement', 'Dice — Entraînement', 'IoU — Entraînement']
titles_val   = ['Loss — Validation',   'Dice — Validation',   'IoU — Validation']
keys_train   = ['train_loss', 'train_dice', 'train_iou']
keys_val     = ['val_loss',   'val_dice',   'val_iou']

for (name, hist), col in zip(histories.items(), palette):
    for j, (key_t, key_v, t_t, t_v) in enumerate(
            zip(keys_train, keys_val, titles_train, titles_val)):
        axes[0, j].plot(epochs, hist[key_t], label=name, color=col, linewidth=2)
        axes[1, j].plot(epochs, hist[key_v], label=name, color=col, linewidth=2)

for j in range(3):
    axes[0, j].set_title(titles_train[j])
    axes[1, j].set_title(titles_val[j])
    for row in range(2):
        axes[row, j].set_xlabel('Epoch')
        axes[row, j].legend(fontsize=9)
        axes[row, j].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./results/courbes_entrainement.png', bbox_inches='tight')
plt.show()
print('Courbes sauvegardées.')


# ─────────────────────────────────────────────
# 4. Graphique comparatif barres + heatmap
# ─────────────────────────────────────────────
metrics_names = ['Dice', 'IoU', 'Précision', 'Rappel', 'F1-Score', 'Accuracy']
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Comparaison des Modèles de Segmentation', fontsize=14, fontweight='bold')

x     = np.arange(len(metrics_names))
width = 0.25

for i, (model_name, color) in enumerate(zip(df.index, palette)):
    vals = [df.loc[model_name, m] for m in metrics_names]
    bars = axes[0].bar(x + i * width, vals, width, label=model_name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

axes[0].set_xlabel('Métriques')
axes[0].set_ylabel('Score')
axes[0].set_title('Scores par métrique')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_names, rotation=25, ha='right', fontsize=9)
axes[0].set_ylim(0, 1.10)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

hm = df[metrics_names].astype(float)
sns.heatmap(hm, annot=True, fmt='.4f', cmap='YlOrRd',
            linewidths=0.5, ax=axes[1], vmin=0.60, vmax=1.0,
            annot_kws={'size': 11})
axes[1].set_title('Heatmap des performances')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig('./results/comparaison_modeles.png', bbox_inches='tight')
plt.show()
print('Graphique comparatif sauvegardé.')


# ─────────────────────────────────────────────
# 5. Diagramme radar
# ─────────────────────────────────────────────
def radar_chart(df_r, metrics, title='Comparaison Radar'):
    N      = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)

    for (name, row), color in zip(df_r[metrics].iterrows(), palette):
        vals = list(row) + [row[0]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_ylim(0.60, 1.0)
    ax.set_title(title, size=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    plt.savefig('./results/radar_chart.png', bbox_inches='tight')
    plt.show()


radar_chart(df, metrics_names, 'Comparaison Radar des Modèles')
print('Radar chart sauvegardé.')


# ─────────────────────────────────────────────
# 6. Boxplots distribution Dice par image
# ─────────────────────────────────────────────
def sim_dice_distribution(mean, std, n=589):
    vals = np.random.beta(
        a=mean * 20,
        b=(1 - mean) * 20,
        size=n
    )
    return np.clip(vals, 0, 1)


dice_u = sim_dice_distribution(0.8124, 0.12)
dice_a = sim_dice_distribution(0.8743, 0.09)
dice_r = sim_dice_distribution(0.8612, 0.10)

dice_df = pd.DataFrame({'U-Net': dice_u, 'Attention U-Net': dice_a, 'ResU-Net': dice_r})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribution des Scores par Image (Jeu de Test)', fontsize=13, fontweight='bold')

dice_df.boxplot(ax=axes[0], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Distribution du Dice Coefficient')
axes[0].set_ylabel('Score Dice')
axes[0].grid(alpha=0.3)

iou_df = dice_df * 0.88
iou_df.boxplot(ax=axes[1], patch_artist=True,
               boxprops=dict(facecolor='lightgreen', color='darkgreen'),
               medianprops=dict(color='red', linewidth=2))
axes[1].set_title("Distribution de l'IoU")
axes[1].set_ylabel('Score IoU')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./results/distribution_scores.png', bbox_inches='tight')
plt.show()

print('\nStatistiques Dice par image :')
print(dice_df.describe().round(4))


# ─────────────────────────────────────────────
# 7. Tableau HTML stylisé pour rapport
# ─────────────────────────────────────────────
styled = df[metrics_names].style \
    .background_gradient(cmap='YlOrRd', vmin=0.6, vmax=1.0) \
    .format('{:.4f}') \
    .set_caption('Tableau Comparatif — Jeu de Test')
styled.to_html('./results/tableau_comparatif.html')

print('\n=== RÉSUMÉ FINAL ===')
print(df[metrics_names].to_string())
print(f'\nMeilleur modèle : {df["Dice"].idxmax()} (Dice = {df["Dice"].max():.4f})')
print('Tous les résultats sauvegardés dans ./results/')
