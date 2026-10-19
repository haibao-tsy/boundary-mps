from string_order import *
import csv
from pathlib import Path

results = {
    "λg": np.round( lam_twisted_dom, decimals=3 ),
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

output_path = Path(__file__).resolve().with_name("result_test.csv")

with open(output_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results.keys()))

    if file.tell() == 0: writer.writeheader()

    writer.writerow(results)