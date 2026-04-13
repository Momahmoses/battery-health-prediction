"""
Battery Health Prediction — Professional Interactive CLI
Predicts: SOH (%), RUL (cycles), and Fault Type simultaneously.
Run: python3 predict.py
"""

import joblib
import numpy as np

soh_model   = joblib.load('models/soh_model.pkl')
rul_model   = joblib.load('models/rul_model.pkl')
fault_model = joblib.load('models/fault_model.pkl')
scaler      = joblib.load('models/scaler.pkl')
le_chem     = joblib.load('models/chemistry_encoder.pkl')
meta        = joblib.load('models/metadata.pkl')

FAULT_LABELS = {
    0: 'Normal',
    1: 'Thermal Runaway Risk',
    2: 'Overcharge',
    3: 'Deep Discharge',
    4: 'Internal Short Circuit'
}
FAULT_SEVERITY = {0: 'SAFE', 1: 'CRITICAL', 2: 'WARNING', 3: 'WARNING', 4: 'CRITICAL'}
CHEMISTRIES = ['LFP', 'NMC', 'NCA', 'LTO']

def get_input(prompt, type_fn, valid=None, min_val=None, max_val=None):
    while True:
        try:
            val = type_fn(input(prompt).strip())
            if valid and val not in valid:
                print(f"  Options: {valid}")
                continue
            if min_val is not None and val < min_val:
                print(f"  Min: {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Max: {max_val}")
                continue
            return val
        except ValueError:
            print("  Invalid input.")

def soh_grade(soh):
    if soh >= 90: return "A  — Excellent"
    elif soh >= 80: return "B  — Good"
    elif soh >= 70: return "C  — Fair (monitor closely)"
    elif soh >= 60: return "D  — Poor (plan replacement)"
    else:           return "F  — Replace immediately"

def predict():
    print("\n" + "="*65)
    print("   PROFESSIONAL BATTERY HEALTH PREDICTION SYSTEM")
    print("   Outputs: SOH (%) | RUL (cycles) | Fault Detection")
    print("="*65)
    print("\nEnter battery parameters:\n")

    cycle_count     = get_input("Cycle count: ", int, min_val=0)
    print(f"Chemistry options: {', '.join(f'{i+1}.{c}' for i,c in enumerate(CHEMISTRIES))}")
    chem_idx        = get_input("Chemistry number: ", int, min_val=1, max_val=4) - 1
    chemistry       = CHEMISTRIES[chem_idx]
    nominal_cap     = get_input("Nominal capacity (Ah): ", float, min_val=1)
    voltage         = get_input("Current voltage (V): ", float, min_val=2.0, max_val=5.0)
    current         = get_input("Current (A) [+ve=charging, -ve=discharging]: ", float)
    temperature     = get_input("Battery temperature (°C): ", float, min_val=-20, max_val=80)
    charge_voltage  = get_input("Max charge voltage (V) [e.g. 4.2]: ", float, min_val=3.5, max_val=5.0)
    disc_cutoff     = get_input("Discharge cutoff voltage (V) [e.g. 2.7]: ", float, min_val=1.5, max_val=3.5)
    avg_charge_c    = get_input("Avg charge C-rate [e.g. 0.5, 1.0]: ", float, min_val=0.1, max_val=10)
    avg_disc_c      = get_input("Avg discharge C-rate [e.g. 1.0]: ", float, min_val=0.1, max_val=10)
    dod             = get_input("Depth of discharge (0.0-1.0): ", float, min_val=0.0, max_val=1.0)
    charge_time     = get_input("Charge time per cycle (hours): ", float, min_val=0.1)
    rest_time       = get_input("Rest time between cycles (hours): ", float, min_val=0)
    coulombic_eff   = get_input("Coulombic efficiency (0.85-1.0): ", float, min_val=0.5, max_val=1.0)
    resistance      = get_input("Internal resistance (Ω): ", float, min_val=0.001)
    max_temp        = get_input("Max temperature ever recorded (°C): ", float)
    min_temp        = get_input("Min temperature ever recorded (°C): ", float)
    humidity        = get_input("Ambient humidity (%): ", float, min_val=0, max_val=100)
    overcharge_ev   = get_input("Number of overcharge events: ", int, min_val=0)
    overdischarge_ev= get_input("Number of overdischarge events: ", int, min_val=0)
    age_days        = get_input("Battery age (days): ", int, min_val=0)

    chem_enc = le_chem.transform([chemistry])[0]

    features = np.array([[
        cycle_count, nominal_cap, voltage, current, temperature,
        charge_voltage, disc_cutoff, avg_charge_c, avg_disc_c,
        dod, charge_time, rest_time, coulombic_eff, resistance,
        max_temp, min_temp, humidity, overcharge_ev, overdischarge_ev,
        age_days, chem_enc
    ]])

    features_sc = scaler.transform(features)

    # SOH
    Xte_soh = features_sc if meta['soh_scaled'] else features
    soh = float(np.clip(soh_model.predict(Xte_soh)[0], 20, 100))

    # RUL
    Xte_rul = features_sc if meta['rul_scaled'] else features
    rul = int(np.clip(rul_model.predict(Xte_rul)[0], 0, 15000))

    # Fault
    Xte_fault = features_sc if meta['fault_scaled'] else features
    fault_code  = int(fault_model.predict(Xte_fault)[0])
    fault_probs = fault_model.predict_proba(Xte_fault)[0]
    fault_label = FAULT_LABELS[fault_code]
    fault_sev   = FAULT_SEVERITY[fault_code]

    # Estimated remaining years (assuming avg 1.5 cycles/day)
    years_left  = rul / (1.5 * 365)

    print("\n" + "="*65)
    print("  BATTERY HEALTH REPORT")
    print("="*65)
    print(f"  Chemistry        : {chemistry}  |  Cycles: {cycle_count:,}")
    print(f"  Age              : {age_days} days  ({age_days//365}y {age_days%365}d)")
    print("-"*65)
    print(f"  State of Health  : {soh:.1f}%   Grade: {soh_grade(soh)}")
    print(f"  Remaining Life   : {rul:,} cycles  (~{years_left:.1f} years)")
    print(f"  Fault Status     : {fault_label}  [{fault_sev}]")
    print("-"*65)
    print(f"  Fault Probabilities:")
    for i, (code, label) in enumerate(FAULT_LABELS.items()):
        bar = '█' * int(fault_probs[i] * 20)
        print(f"    {label:<28}: {fault_probs[i]*100:5.1f}%  {bar}")
    print("="*65)

    print("\n  RECOMMENDATIONS:")
    if soh < 80:
        print(f"  - SOH at {soh:.1f}% — below 80% threshold. Plan battery replacement.")
    if rul < 200:
        print(f"  - Only {rul} cycles remaining. Schedule replacement soon.")
    if fault_code == 1:
        print("  - CRITICAL: Thermal runaway risk detected!")
        print("    Stop operation immediately, move to fireproof area, inspect cooling.")
    elif fault_code == 4:
        print("  - CRITICAL: Internal short circuit suspected!")
        print("    Isolate battery immediately and perform full inspection.")
    elif fault_code == 2:
        print("  - Overcharge detected. Check BMS voltage limits and charger.")
    elif fault_code == 3:
        print("  - Deep discharge detected. Raise discharge cutoff voltage.")
    if resistance > 0.3:
        print(f"  - High resistance ({resistance:.3f}Ω). Check connections and cell health.")
    if max_temp > 55:
        print(f"  - Battery has been exposed to high temps ({max_temp}°C). Thermal damage likely.")
    if soh >= 80 and fault_code == 0:
        print("  - Battery is in good health. Continue normal operation.")
        print("  - Schedule next check after 100 cycles or 90 days.")
    print("="*65)

    again = input("\nAnalyse another battery? (y/n): ").strip().lower()
    if again == 'y':
        predict()

if __name__ == '__main__':
    predict()
