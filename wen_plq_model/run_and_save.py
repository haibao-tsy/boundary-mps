from string_order import *
import csv
from pathlib import Path
from datetime import datetime

results = {
    "λg": np.round(mpu_lam_twisted_dom, decimals=5),
    "Δλg": np.round(
        np.abs(mpu_lam_twisted[0]) - np.abs(mpu_lam_twisted[1]), decimals=5
    ),
    "θ": np.round(
        np.degrees(np.angle(mpu_lam_twisted[0] / mpu_lam_twisted[1])), decimals=5
    ),
    "λ0": np.round(mpu_lam_untwisted_dom, decimals=5),
    "|λg/λ0|": np.round(
        np.max(np.abs(mpu_lam_twisted)) / np.max(np.abs(mpu_lam_untwisted)), decimals=5
    ),
    "phys λg": np.round(lam_twisted_dom, decimals=5),
    "phys Δλg": np.round(np.abs(lam_twisted[0]) - np.abs(lam_twisted[1]), decimals=5),
    "phys θ": np.round(
        np.degrees(np.angle(lam_twisted[0] / lam_twisted[1])), decimals=5
    ),
    "phys λ0": np.round(lam_untwisted_dom, decimals=5),
    "phys |λg/λ0|": np.round(
        np.max(np.abs(lam_twisted)) / np.max(np.abs(lam_untwisted)), decimals=5
    ),
    "⟨ff|ff⟩": np.round(ffCorrelator, decimals=5),
    "⟨GS|ee⟩": np.round(eCorrelator, decimals=5),
    "⟨GS|mm⟩": np.round(mCorrelator, decimals=5),
    "⟨mm|mm⟩": np.round(mmCorrelator, decimals=5),
    "⟨ee|ee⟩": np.round(eeCorrelator, decimals=5),
    "⟨f'f'|f'f'⟩": np.round(fbarfbarCorrelator, decimals=5),
    "⟨GS|e'e'⟩": np.round(ebarCorrelator, decimals=5),
    "⟨GS|m'm'⟩": np.round(mbarCorrelator, decimals=5),
    "⟨m'm'|m'm'⟩": np.round(mbarmbarCorrelator, decimals=5),
    "⟨e'e'|e'e'⟩": np.round(ebarebarCorrelator, decimals=5),
    "⟨GS|e₁e₁⟩": np.round(e1Correlator, decimals=5),
    "⟨e₁e₁|e₁e₁⟩": np.round(e1e1Correlator, decimals=5),
    "⟨GS|e₂e₂⟩": np.round(e2Correlator, decimals=5),
    "⟨e₂e₂|e₂e₂⟩": np.round(e2e2Correlator, decimals=5),
}
print("\n Correlator computations")
for key, item in results.items():
    print(f"{key}={item}")

print("e = e₂m₁, m = m₂, e' = e₁m₂, m' = m₁")
if_save = True

if if_save:
    sub_folder_name = "single_site_unit_cell(2)"

    output_path = (
        Path(__file__).resolve().parent / sub_folder_name / f"{sub_folder_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results.keys()))

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(results)

    npz_filename = f"peps_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
    npz_output_path = Path(__file__).resolve().parent / sub_folder_name / npz_filename
    npz_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(npz_output_path, pepsA=pepsA, pepsB=pepsB)

    info_output_path = Path(__file__).resolve().parent / sub_folder_name / "info.text"

    with open(info_output_path, "w", encoding="utf-8") as file:
        file.write(f"MPS bond dimension: {chiMPS} \n")
        file.write(f"VUMPS tolerance: {tol}")
