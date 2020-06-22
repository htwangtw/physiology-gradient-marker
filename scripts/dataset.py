"""
Selected assessment data to use in the analysis
plot qc stuff
"""

from pathlib import Path
import json

import NKIhelper as nh


if __name__ == "__main__":

    p = Path.home() / "projects/gradient-physiology"
    ref = p / "data/variable_selection.json"
    raw_data = p / "data/sourcedata/AssessmentData"

    # load selected variables
    with open(ref, "r") as f:
        dict_var = json.load(f)

    # filter assessment data by age
    path = list(raw_data.glob(f"*/*Age*.csv"))[0]
    df_age = nh.select_by_age(path, dict_var["age"])
    subject_list = df_age.index.tolist()

    # clean data by missing situation
    df_phenotype, missing_percentage = nh.clean_variablewise(dict_var["assessments"],
                                                             subject_list,
                                                             raw_data, threshold=0.3)
    fig = nh.plot_missing_variablewise(missing_percentage, threshold=0.3)
    fig.savefig(p / "results/figures/data_quality/missing_variablewise.png")

    df_phenotype, subj_missing_portion = nh.clean_subjectwise(df_master, threshold=0.2)
    fig = nh.plot_missing_subjectwise(subj_missing_portion, threshold=0.2)
    fig.savefig(p / "results/figures/data_quality/missing_subjectwise.png")

    # save the selected assessments
    df_age.to_csv(p / "data/derivatives/subset_age.tsv", sep="\t")
    df_phenotype.to_csv(p / "data/derivatives/subset_phenotype.tsv", sep="\t")
