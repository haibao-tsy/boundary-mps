from string_order import *
import csv
from pathlib import Path
from datetime import datetime

results = {
    "λg": np.round(mpu_lam_twisted_dom, decimals=3),
    "Δλg": np.round(
        np.abs(mpu_lam_twisted[0]) - np.abs(mpu_lam_twisted[1]), decimals=3
    ),
    "θ": np.round(
        np.degrees(np.angle(mpu_lam_twisted[0] / mpu_lam_twisted[1])), decimals=3
    ),
    "λ0": np.round(mpu_lam_untwisted_dom, decimals=3),
    "|λg/λ0|": np.round(
        np.abs(mpu_lam_twisted_dom[0] / mpu_lam_untwisted_dom[0]), decimals=3
    ),
    "phys λg": np.round(lam_twisted_dom, decimals=3),
    "phys Δλg": np.round(np.abs(lam_twisted[0]) - np.abs(lam_twisted[1]), decimals=3),
    "phys θ": np.round(
        np.degrees(np.angle(lam_twisted[0] / lam_twisted[1])), decimals=3
    ),
    "phys λ0": np.round(lam_untwisted_dom, decimals=3),
    "phys |λg/λ0|": np.round(
        np.abs(lam_twisted_dom[0] / lam_untwisted_dom[0]), decimals=3
    ),
    "⟨ff|ff⟩": np.round(ffCorrelator, decimals=3),
    "⟨GS|ee⟩": np.round(eCorrelator, decimals=3),
    "⟨GS|mm⟩": np.round(mCorrelator, decimals=3),
    "⟨mm|mm⟩": np.round(mmCorrelator, decimals=3),
    "⟨ee|ee⟩": np.round(eeCorrelator, decimals=3),
    "⟨f'f'|f'f'⟩": np.round(fbarfbarCorrelator, decimals=3),
    "⟨GS|e'e'⟩": np.round(ebarCorrelator, decimals=3),
    "⟨GS|m'm'⟩": np.round(mbarCorrelator, decimals=3),
    "⟨m'm'|m'm'⟩": np.round(mbarmbarCorrelator, decimals=3),
    "⟨e'e'|e'e'⟩": np.round(ebarebarCorrelator, decimals=3),
}
sub_folder_name = "2st_no_exgg_no_pert"

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
