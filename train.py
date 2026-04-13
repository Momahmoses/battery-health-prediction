"""
Battery Health Prediction — Professional ML Pipeline
Three model tasks:
  Task 1: SOH Regression  — predict State of Health %
  Task 2: RUL Regression  — predict Remaining Useful Life (cycles)
  Task 3: Fault Classification — detect battery fault type

Used in: EV manufacturing, BMS, predictive maintenance, grid storage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load & encode ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/battery_health_dataset.csv')
le_chem = LabelEncoder()
df['chemistry_enc'] = le_chem.fit_transform(df['battery_chemistry'])

features = [
    'cycle_count', 'nominal_capacity_ah', 'voltage_v', 'current_a',
    'temperature_c', 'charge_voltage', 'discharge_cutoff',
    'avg_charge_current', 'avg_discharge_curr', 'depth_of_discharge',
    'charge_time_h', 'rest_time_h', 'coulombic_efficiency',
    'internal_resistance', 'max_temp_seen', 'min_temp_seen',
    'humidity_pct', 'overcharge_events', 'overdischarge_events',
    'battery_age_days', 'chemistry_enc'
]

X = df[features]

# ── Shared train/test split ────────────────────────────────────────────────────
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
idx_train = X_train.index
idx_test  = X_test.index

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

def run_regression(y, task_name, unit=''):
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression':  (Ridge(alpha=1.0), True),
        'Decision Tree':     (DecisionTreeRegressor(max_depth=12, random_state=42), False),
        'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
    }
    results = {}
    print(f"\n── {task_name} ─────────────────────────────────────────────")
    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print("-"*55)
    for name, (model, scaled) in models.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'scaled': scaled,
                         'mae': mae, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
        print(f"{name:<22} {mae:>10.3f} {rmse:>10.3f} {r2:>8.4f}")

    best_name = max(results, key=lambda k: results[k]['r2'])
    best = results[best_name]
    print(f"\nBest: {best_name}  MAE={best['mae']:.3f}{unit}  R²={best['r2']:.4f}")
    return results, best_name, best, y_test

def run_classification(y, task_name):
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    # SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    X_bal_sc = scaler.transform(X_bal)

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree':       (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    }
    results = {}
    print(f"\n── {task_name} ─────────────────────────────────────────────")
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 (macro)':>12} {'AUC (OvR)':>12}")
    print("-"*60)
    for name, (model, scaled) in models.items():
        Xtr = X_bal_sc if scaled else X_bal
        Xte = X_test_sc if scaled else X_test
        model.fit(Xtr, y_bal)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        results[name] = {'model': model, 'scaled': scaled,
                         'acc': acc, 'f1': f1, 'auc': auc,
                         'y_pred': y_pred, 'y_prob': y_prob}
        print(f"{name:<22} {acc:>10.3f} {f1:>12.3f} {auc:>12.3f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    print(f"\nBest: {best_name}  Acc={best['acc']:.3f}  F1={best['f1']:.3f}  AUC={best['auc']:.3f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_test, best['y_pred'])}")
    return results, best_name, best, y_test

# ── Run all three tasks ────────────────────────────────────────────────────────
soh_results, soh_best_name, soh_best, soh_test = run_regression(
    df['soh_percent'], 'TASK 1: SOH Regression', unit='%')

rul_results, rul_best_name, rul_best, rul_test = run_regression(
    df['rul_cycles'], 'TASK 2: RUL Regression', unit=' cycles')

fault_results, fault_best_name, fault_best, fault_test = run_classification(
    df['fault_type'], 'TASK 3: Fault Classification')

fault_names = ['Normal','Thermal_Risk','Overcharge','Deep_Discharge','Internal_Short']

# ── Plot 9: SOH Actual vs Predicted ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_arr = np.array(soh_test)
idx   = np.random.choice(len(y_arr), 1000, replace=False)
axes[0].scatter(y_arr[idx], soh_best['y_pred'][idx], alpha=0.4, s=10, color='steelblue')
mn, mx = y_arr.min(), y_arr.max()
axes[0].plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('Actual SOH (%)')
axes[0].set_ylabel('Predicted SOH (%)')
axes[0].set_title(f'SOH: Actual vs Predicted — {soh_best_name}')
axes[0].legend()

res = y_arr - soh_best['y_pred']
axes[1].hist(res, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title(f'SOH Residuals (MAE={soh_best["mae"]:.2f}%)')
axes[1].set_xlabel('Residual')
plt.tight_layout()
plt.savefig('plots/09_soh_predictions.png', dpi=150)
plt.close()

# ── Plot 10: RUL Actual vs Predicted ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_arr = np.array(rul_test)
idx   = np.random.choice(len(y_arr), 1000, replace=False)
axes[0].scatter(y_arr[idx], rul_best['y_pred'][idx], alpha=0.4, s=10, color='green')
mn, mx = y_arr.min(), y_arr.max()
axes[0].plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('Actual RUL (cycles)')
axes[0].set_ylabel('Predicted RUL (cycles)')
axes[0].set_title(f'RUL: Actual vs Predicted — {rul_best_name}')
axes[0].legend()

res = y_arr - rul_best['y_pred']
axes[1].hist(res, bins=50, color='green', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title(f'RUL Residuals (MAE={rul_best["mae"]:.0f} cycles)')
axes[1].set_xlabel('Residual')
plt.tight_layout()
plt.savefig('plots/10_rul_predictions.png', dpi=150)
plt.close()

# ── Plot 11: Fault Confusion Matrix ───────────────────────────────────────────
cm = confusion_matrix(fault_test, fault_best['y_pred'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=fault_names, yticklabels=fault_names, ax=ax)
ax.set_title(f'Fault Detection Confusion Matrix — {fault_best_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/11_fault_confusion_matrix.png', dpi=150)
plt.close()

# ── Plot 12: Feature Importance for SOH ───────────────────────────────────────
for task, res, name, color, fname in [
    ('SOH',   soh_results,   soh_best_name,   'steelblue', '12_soh_feature_importance.png'),
    ('Fault', fault_results, fault_best_name, 'crimson',   '13_fault_feature_importance.png'),
]:
    model = res[name]['model']
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(10, 7))
        imp.plot(kind='barh', color=color, ax=ax, alpha=0.85)
        ax.set_title(f'Feature Importance — {task} ({name})')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'plots/{fname}', dpi=150)
        plt.close()

# ── Save all models ────────────────────────────────────────────────────────────
joblib.dump(soh_results[soh_best_name]['model'],     'models/soh_model.pkl')
joblib.dump(rul_results[rul_best_name]['model'],     'models/rul_model.pkl')
joblib.dump(fault_results[fault_best_name]['model'], 'models/fault_model.pkl')
joblib.dump(scaler,   'models/scaler.pkl')
joblib.dump(le_chem,  'models/chemistry_encoder.pkl')
joblib.dump({
    'features':        features,
    'soh_model':       soh_best_name,
    'rul_model':       rul_best_name,
    'fault_model':     fault_best_name,
    'soh_scaled':      soh_results[soh_best_name]['scaled'],
    'rul_scaled':      rul_results[rul_best_name]['scaled'],
    'fault_scaled':    fault_results[fault_best_name]['scaled'],
    'fault_labels':    fault_names,
}, 'models/metadata.pkl')

print("\n\nAll models saved to models/")
print("All plots  saved to plots/")
