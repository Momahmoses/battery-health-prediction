"""
Generates a synthetic professional Battery Health dataset.
Targets:
  1. SOH (State of Health %)     — regression
  2. RUL (Remaining Useful Life) — regression (cycles remaining)
  3. Fault Type                  — multi-class classification

Used in: EV manufacturing, BMS, grid storage, predictive maintenance.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 12000

# ── Raw battery parameters ─────────────────────────────────────────────────────
cycle_count         = np.random.randint(0, 2000, N)
nominal_capacity_ah = np.random.choice([50, 60, 75, 100, 150, 200], N,
                          p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
voltage_v           = np.random.uniform(2.8, 4.25, N)
current_a           = np.random.normal(0, 20, N).clip(-80, 80)
temperature_c       = np.random.normal(30, 10, N).clip(-20, 70)
charge_voltage      = np.random.uniform(4.0, 4.25, N)    # max charge voltage
discharge_cutoff    = np.random.uniform(2.5, 3.0, N)     # min discharge voltage
avg_charge_current  = np.random.uniform(0.2, 3.0, N)     # C-rate
avg_discharge_curr  = np.random.uniform(0.2, 5.0, N)     # C-rate
depth_of_discharge  = np.random.uniform(0.3, 1.0, N)     # DoD (fraction)
charge_time_h       = np.random.uniform(0.3, 8.0, N)
rest_time_h         = np.random.uniform(0, 12, N)
coulombic_efficiency= np.random.normal(0.98, 0.02, N).clip(0.85, 1.0)
internal_resistance = (0.04 + cycle_count * 0.00008 +
                       np.random.normal(0, 0.005, N)).clip(0.01, 0.8)
max_temp_seen       = temperature_c + np.random.uniform(0, 20, N)
min_temp_seen       = temperature_c - np.random.uniform(0, 15, N)
humidity_pct        = np.random.normal(55, 15, N).clip(10, 100)
overcharge_events   = np.random.poisson(cycle_count * 0.001, N).clip(0, 50)
overdischarge_events= np.random.poisson(cycle_count * 0.0005, N).clip(0, 30)
battery_chemistry   = np.random.choice(
                          ['LFP', 'NMC', 'NCA', 'LTO'],
                          N, p=[0.30, 0.40, 0.20, 0.10])
battery_age_days    = (cycle_count * 0.7 + np.random.normal(0, 30, N)).clip(0).astype(int)

# ── Target 1: SOH (State of Health %) ─────────────────────────────────────────
# SOH = actual capacity / nominal capacity * 100
# Degrades with cycles, temperature abuse, overcharge/discharge
soh = (
    100
    - 0.020  * cycle_count
    - 8.0    * (internal_resistance - 0.04)
    - 0.015  * overcharge_events
    - 0.010  * overdischarge_events
    - 0.005  * np.maximum(0, max_temp_seen - 45)
    - 0.003  * depth_of_discharge * cycle_count * 0.01
    - 0.002  * (avg_charge_current > 2.0).astype(int) * cycle_count * 0.01
    + 0.003  * (battery_chemistry == 'LFP').astype(int) * 10
    - 0.003  * (battery_chemistry == 'NCA').astype(int) * 10
    + np.random.normal(0, 1.5, N)
).clip(20, 100)

# ── Target 2: RUL (Remaining Useful Life — cycles) ────────────────────────────
max_cycles = {'LFP': 3000, 'NMC': 1500, 'NCA': 1200, 'LTO': 8000}
max_cyc_arr = np.array([max_cycles[c] for c in battery_chemistry])
rul = (
    max_cyc_arr - cycle_count
    - 20 * overcharge_events
    - 10 * overdischarge_events
    - 5  * np.maximum(0, max_temp_seen - 45)
    + np.random.normal(0, 50, N)
).clip(0, max_cyc_arr)

# ── Target 3: Fault Type ───────────────────────────────────────────────────────
# 0=Normal, 1=Thermal Runaway Risk, 2=Overcharge, 3=Deep Discharge, 4=Internal Short
fault = np.zeros(N, dtype=int)

thermal = (
    (temperature_c > 55) |
    (max_temp_seen > 65) |
    ((temperature_c > 45) & (avg_charge_current > 2.5))
)
overchg = (
    (overcharge_events > 5) |
    (charge_voltage > 4.22) |
    (coulombic_efficiency < 0.92)
)
deep_disc = (
    (overdischarge_events > 3) |
    (discharge_cutoff < 2.6) |
    (depth_of_discharge > 0.95)
)
short_ckt = (
    (internal_resistance < 0.02) |
    ((coulombic_efficiency < 0.90) & (temperature_c > 40)) |
    (soh < 50)
)

fault[thermal]  = 1
fault[overchg & ~thermal]   = 2
fault[deep_disc & ~thermal & ~overchg] = 3
fault[short_ckt & ~thermal & ~overchg & ~deep_disc] = 4

# Add random noise to fault (10% noise)
noise_idx = np.random.choice(N, int(N * 0.05), replace=False)
fault[noise_idx] = np.random.randint(0, 5, len(noise_idx))

fault_labels = {0: 'Normal', 1: 'Thermal_Runaway_Risk',
                2: 'Overcharge', 3: 'Deep_Discharge', 4: 'Internal_Short'}

df = pd.DataFrame({
    'cycle_count':          cycle_count,
    'nominal_capacity_ah':  nominal_capacity_ah,
    'voltage_v':            voltage_v.round(4),
    'current_a':            current_a.round(3),
    'temperature_c':        temperature_c.round(2),
    'charge_voltage':       charge_voltage.round(4),
    'discharge_cutoff':     discharge_cutoff.round(4),
    'avg_charge_current':   avg_charge_current.round(3),
    'avg_discharge_curr':   avg_discharge_curr.round(3),
    'depth_of_discharge':   depth_of_discharge.round(4),
    'charge_time_h':        charge_time_h.round(3),
    'rest_time_h':          rest_time_h.round(3),
    'coulombic_efficiency': coulombic_efficiency.round(5),
    'internal_resistance':  internal_resistance.round(5),
    'max_temp_seen':        max_temp_seen.round(2),
    'min_temp_seen':        min_temp_seen.round(2),
    'humidity_pct':         humidity_pct.round(1),
    'overcharge_events':    overcharge_events,
    'overdischarge_events': overdischarge_events,
    'battery_chemistry':    battery_chemistry,
    'battery_age_days':     battery_age_days,
    'soh_percent':          soh.round(2),
    'rul_cycles':           rul.round(0).astype(int),
    'fault_type':           fault,
    'fault_label':          [fault_labels[f] for f in fault],
})

df.to_csv('data/battery_health_dataset.csv', index=False)
print(f"Dataset created    : {len(df):,} records")
print(f"\nSOH Stats          : Mean={soh.mean():.1f}%  Min={soh.min():.1f}%  Max={soh.max():.1f}%")
print(f"RUL Stats          : Mean={rul.mean():.0f} cycles  Min={rul.min():.0f}  Max={rul.max():.0f}")
print(f"\nFault Distribution :")
for k, v in fault_labels.items():
    cnt = (fault == k).sum()
    print(f"  {v:<25}: {cnt:,} ({cnt/N*100:.1f}%)")
print(f"\nChemistry split    :\n{pd.Series(battery_chemistry).value_counts().to_string()}")
