"""
Script (attempts) to help variable selection
"""

import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    p = Path.home / "projects/physiology-gradient-marker"

    lst = []
    # read file
    with open(p / "references/interested_assessment.txt", "r") as f:
        for fn in f:
            fn = fn.split(",")[-1].split("\n")[0]
            file_path = list(p.glob(f"data/sourcedata/AssessmentData/*/{fn}.csv"))[0]
            print(pd.read_csv(file_path).T)
            print(fn)
            var = input("Enter the selected variable(s); separate by commas:\n")
            lst.append(var)
            with open(p / "references/interested_var.txt", "a") as vw:
                vw.write(fn + "," + var + "\n")