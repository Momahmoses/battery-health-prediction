"""
Exploratory Data Analysis — Battery Health Dataset
Covers SOH, RUL, and Fault Type insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/battery_health_dataset.csv')

print("="*60)
print("  BATTERY HEALTH — EXPLORATORY DATA ANALYSIS")
print("="*60)
print(f"\nShape : {df.shape}")
print(f"\nSOH   : Mean={df['soh_percent'].mean():.1f}%  Std={df['soh_percent'].std():.1f}%")
print(f"RUL   : Mean={df['rul_cycles'].mean():.0f}  Std={df['rul_cycles'].std():.0f}")
print(f"\nFaults:\n{df['fault_label'].value_counts()}")

# ── Plot 1: SOH & RUL Distributions ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(df['soh_percent'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('State of Health (SOH) Distribution')
axes[0].set_xlabel('SOH (%)')
axes[0].axvline(df['soh_percent'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['soh_percent'].mean():.1f}%")
axes[0].axvline(80, color='orange', linestyle=':', label='Replacement threshold (80%)')
axes[0].legend(fontsize=8)

axes[1].hist(df['rul_cycles'], bins=50, color='green', edgecolor='white', alpha=0.85)
axes[1].set_title('Remaining Useful Life (RUL) Distribution')
axes[1].set_xlabel('RUL (Cycles)')
axes[1].axvline(df['rul_cycles'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['rul_cycles'].mean():.0f}")
axes[1].legend(fontsize=8)

fault_counts = df['fault_label'].value_counts()
colors = ['green','crimson','darkorange','purple','navy']
bars = axes[2].bar(fault_counts.index, fault_counts.values, color=colors, alpha=0.85)
axes[2].set_title('Fault Type Distribution')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, fault_counts.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', fontsize=8)
plt.suptitle('Target Variable Distributions', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_target_distributions.png', dpi=150)
plt.close()

# ── Plot 2: SOH Degradation Over Cycles ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for chem, color in zip(['LFP','NMC','NCA','LTO'], ['steelblue','crimson','green','purple']):
    sub = df[df['battery_chemistry'] == chem].sample(min(500, len(df[df['battery_chemistry']==chem])),
                                                      random_state=42)
    axes[0].scatter(sub['cycle_count'], sub['soh_percent'],
                    alpha=0.3, s=8, color=color, label=chem)
axes[0].axhline(80, color='black', linestyle='--', alpha=0.6, label='80% replacement threshold')
axes[0].set_xlabel('Cycle Count')
axes[0].set_ylabel('SOH (%)')
axes[0].set_title('SOH Degradation by Battery Chemistry')
axes[0].legend(fontsize=9)

for chem, color in zip(['LFP','NMC','NCA','LTO'], ['steelblue','crimson','green','purple']):
    sub = df[df['battery_chemistry'] == chem].sample(min(500, len(df[df['battery_chemistry']==chem])),
                                                      random_state=42)
    axes[1].scatter(sub['cycle_count'], sub['rul_cycles'],
                    alpha=0.3, s=8, color=color, label=chem)
axes[1].set_xlabel('Cycle Count')
axes[1].set_ylabel('RUL (Cycles Remaining)')
axes[1].set_title('Remaining Useful Life by Chemistry')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/02_degradation_by_chemistry.png', dpi=150)
plt.close()

# ── Plot 3: Internal Resistance vs SOH ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
s = df.sample(2000, random_state=1)
sc = axes[0].scatter(s['internal_resistance'], s['soh_percent'],
                     c=s['cycle_count'], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(sc, ax=axes[0], label='Cycle Count')
axes[0].set_xlabel('Internal Resistance (Ω)')
axes[0].set_ylabel('SOH (%)')
axes[0].set_title('Internal Resistance vs SOH')

sc2 = axes[1].scatter(s['internal_resistance'], s['rul_cycles'],
                      c=s['soh_percent'], cmap='RdYlGn', alpha=0.5, s=10)
plt.colorbar(sc2, ax=axes[1], label='SOH (%)')
axes[1].set_xlabel('Internal Resistance (Ω)')
axes[1].set_ylabel('RUL (Cycles)')
axes[1].set_title('Internal Resistance vs RUL')
plt.tight_layout()
plt.savefig('plots/03_resistance_analysis.png', dpi=150)
plt.close()

# ── Plot 4: Temperature Impact ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
temp_bins = pd.cut(df['max_temp_seen'], bins=[-20,0,25,45,60,80],
                   labels=['<0°C','0-25°C','25-45°C','45-60°C','>60°C'])
soh_by_temp = df.groupby(temp_bins, observed=True)['soh_percent'].mean()
rul_by_temp = df.groupby(temp_bins, observed=True)['rul_cycles'].mean()

axes[0].bar(soh_by_temp.index.astype(str), soh_by_temp.values, color='darkorange', alpha=0.85)
axes[0].set_title('Average SOH by Max Temperature Seen')
axes[0].set_xlabel('Max Temperature')
axes[0].set_ylabel('Average SOH (%)')
for i, v in enumerate(soh_by_temp.values):
    axes[0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)

axes[1].bar(rul_by_temp.index.astype(str), rul_by_temp.values, color='crimson', alpha=0.85)
axes[1].set_title('Average RUL by Max Temperature Seen')
axes[1].set_xlabel('Max Temperature')
axes[1].set_ylabel('Average RUL (Cycles)')
for i, v in enumerate(rul_by_temp.values):
    axes[1].text(i, v+5, f'{v:.0f}', ha='center', fontsize=9)
plt.suptitle('Temperature Impact on Battery Health', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_temperature_impact.png', dpi=150)
plt.close()

# ── Plot 5: Fault Analysis ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fault_order = ['Normal','Thermal_Runaway_Risk','Overcharge','Deep_Discharge','Internal_Short']
clrs = ['green','crimson','darkorange','purple','navy']

soh_by_fault = df.groupby('fault_label')['soh_percent'].mean().reindex(fault_order)
axes[0].bar(fault_order, soh_by_fault.values, color=clrs, alpha=0.85)
axes[0].set_title('Average SOH by Fault Type')
axes[0].set_ylabel('SOH (%)')
axes[0].tick_params(axis='x', rotation=20)

rul_by_fault = df.groupby('fault_label')['rul_cycles'].mean().reindex(fault_order)
axes[1].bar(fault_order, rul_by_fault.values, color=clrs, alpha=0.85)
axes[1].set_title('Average RUL by Fault Type')
axes[1].set_ylabel('RUL (Cycles)')
axes[1].tick_params(axis='x', rotation=20)

temp_by_fault = df.groupby('fault_label')['temperature_c'].mean().reindex(fault_order)
axes[2].bar(fault_order, temp_by_fault.values, color=clrs, alpha=0.85)
axes[2].set_title('Average Temperature by Fault Type')
axes[2].set_ylabel('Temperature (°C)')
axes[2].tick_params(axis='x', rotation=20)
plt.suptitle('Fault Type Analysis', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_fault_analysis.png', dpi=150)
plt.close()

# ── Plot 6: DoD & C-Rate Impact ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
s = df.sample(2000, random_state=2)
sc = axes[0].scatter(s['depth_of_discharge'], s['soh_percent'],
                     c=s['cycle_count'], cmap='plasma', alpha=0.4, s=10)
plt.colorbar(sc, ax=axes[0], label='Cycle Count')
axes[0].set_xlabel('Depth of Discharge')
axes[0].set_ylabel('SOH (%)')
axes[0].set_title('Depth of Discharge vs SOH')

sc2 = axes[1].scatter(s['avg_charge_current'], s['soh_percent'],
                      c=s['cycle_count'], cmap='plasma', alpha=0.4, s=10)
plt.colorbar(sc2, ax=axes[1], label='Cycle Count')
axes[1].set_xlabel('Avg Charge C-Rate')
axes[1].set_ylabel('SOH (%)')
axes[1].set_title('Charge Rate vs SOH')
plt.tight_layout()
plt.savefig('plots/06_dod_crate_impact.png', dpi=150)
plt.close()

# ── Plot 7: Correlation Heatmap ────────────────────────────────────────────────
num_cols = ['cycle_count','voltage_v','current_a','temperature_c',
            'internal_resistance','depth_of_discharge','avg_charge_current',
            'coulombic_efficiency','overcharge_events','overdischarge_events',
            'soh_percent','rul_cycles','fault_type']
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.4, annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/07_correlation_heatmap.png', dpi=150)
plt.close()

# ── Plot 8: SOH Health Bands by Chemistry ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bp_data = [df[df['battery_chemistry']==c]['soh_percent'].values
           for c in ['LFP','NMC','NCA','LTO']]
bp = ax.boxplot(bp_data, tick_labels=['LFP','NMC','NCA','LTO'], patch_artist=True,
                notch=True, showfliers=False)
colors = ['steelblue','crimson','green','purple']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(80, color='black', linestyle='--', label='Replacement threshold (80%)')
ax.set_title('SOH Distribution by Battery Chemistry')
ax.set_ylabel('SOH (%)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/08_soh_by_chemistry_boxplot.png', dpi=150)
plt.close()

print("\nAll EDA plots saved to plots/")
