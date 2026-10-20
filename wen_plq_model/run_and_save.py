from string_order import *
import csv
from pathlib import Path
from datetime import datetime

results = {
    "λg": np.round( lam_twisted_dom, decimals=3 ),
    "Δλg": np.round( np.abs( lam_twisted[0]) - np.abs( lam_twisted[1] ), decimals=3 ),
    "θ": np.round( np.degrees( np.angle(lam_twisted[0] / lam_twisted[1]) ), decimals=3 ),
    "λ0": np.round( lam_untwisted_dom, decimals=3 ),
    "|λg/λ0|": np.round( np.abs( lam_twisted_dom[0] / lam_untwisted_dom[0] ), decimals=3 ),
    "⟨ff|ff⟩": np.round( ffCorrelator, decimals=3),
    "⟨GS|ee⟩": np.round( eCorrelator, decimals=3),
    "⟨GS|mm⟩": np.round( mCorrelator, decimals=3),
    "⟨mm|mm⟩": np.round( mmCorrelator, decimals=3),
    "⟨ee|ee⟩": np.round( eeCorrelator, decimals=3),
    "⟨f'f'|f'f'⟩": np.round( fbarfbarCorrelator, decimals=3),
    "⟨GS|e'e'⟩": np.round( ebarCorrelator, decimals=3),
    "⟨GS|m'm'⟩": np.round( mbarCorrelator, decimals=3),
    "⟨m'm'|m'm'⟩": np.round( mbarmbarCorrelator, decimals=3),
    "⟨e'e'|e'e'⟩": np.round( ebarebarCorrelator, decimals=3),
}

output_path = Path(__file__).resolve().parent / "2st|exgg|pert" / "2st|exgg|pert#1.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results.keys()))

    if file.tell() == 0: writer.writeheader()

    writer.writerow(results)

npz_filename = f"peps_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
npz_output_path = Path(__file__).resolve().parent / "2st|exgg|pert" / npz_filename
npz_output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    npz_output_path,
    pepsA=pepsA,
    pepsB=pepsB
)