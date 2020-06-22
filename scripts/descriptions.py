"""
make a data base for all selected variable descriptions
"""

from pathlib import Path
import json


p = Path.home() / "projects/gradient-physiology"
ref = p / "references/interested_var.txt"

dict_var = {"age": {"min": 17,
                    "max": 86},
            "assessments": {}
           }

def var_descriptions(var_name):
    print(var_name)
    descriptons = input("Enter variable description\n") or "TODO"
    dtype = input("categorical? (yes/no; default: no)\n") or "no"
    if dtype == "yes":
        dtype = "categorical"
    else:
        dtype = "continuous"
    
    stuff = {v: {"descriptions": descriptons,
                 "type": dtype,}}

    if dtype == "categorical":
        complete = "no"
        cat_level = {}
        while complete is "no":
            cur_level = input("Enter key and description separate by ;:\n") or "none;none"
            l, d = cur_level.split(";")
            cat_level[l] = d
            complete = input("Press ENTER to add another key; type 'yes' to end this entry \n") or "no"
        stuff[v].update({"level":cat_level})
    
    return stuff

if __name__ == "__main__":
    with open(ref, "r") as f:
        for line in f.readlines():
            line = line.split("\n")[0]
            name = line.split(",")[0]
            var = line.split(",")[1:]
            if (len(var) == 1) and (var[0] is ""):
                pass
            else:
                stuff = {}
                for v in var:
                    stuff.update(var_descriptions(v))
                dict_var["assessments"][name] = stuff

    with open(p / "data/variable_selection_test.json", 'w') as f:
        json.dump(dict_var, f, indent=2)
