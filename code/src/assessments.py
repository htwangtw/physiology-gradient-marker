"""
Selected assessment data to use in the analysis
plot qc stuff
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ids = ["Anonymized ID", "Visit"]
error_entries = {r"!WithErrors!": "",
                 r"~<userSkipped>~": None,
                 r"REFUSED": None,
                 r"PARTICIPANT DID NOT *": None,
                 r"UNKNOWN": None}


def select_by_age(path, age):
    """
    Filter NKI sample subjects by age
    """
    # select by age range
    df_age = pd.read_csv(path, header=1).sort_values("AGE_04")
    age_range = df_age['AGE_04'].between(age["min"],
                                         age["max"],
                                         inclusive=True)
    df_age = df_age[age_range].reset_index()

    # drop duplicates; we only need the id so order is not important
    df_age_dropdupe = df_age.drop_duplicates(subset='Anonymized ID', keep='first')
    df_age = df_age.set_index('Anonymized ID')
    df_age_dropdupe = df_age_dropdupe.set_index('Anonymized ID')
    df_age = df_age_dropdupe["AGE_04"]
    return df_age


def get_assessment(path, items):
    """
    select a subset of variables from any NKI dataset assessments
    """
    df = pd.read_csv(path, header=1)[ids + list(items.keys())]

    # sort by rough visit order
    df = df.sort_values("Visit")
    df = df.set_index("Anonymized ID")
    return df

def drop_duplicates(df, subject_list, items):
    # drop duplicates; keep first visit
    sl = [i for i in df.index if i in subject_list]
    try:
        df = df.loc[sl, :]
        drop_dup = df.index.duplicated("first")
        df = df[~drop_dup][items]
        return df
    except KeyError:
        print(f"{key}: exclude - variables not collected for target age group")
        return None

def remove_errors(df, error_entries):
    """
    remove error entries
    """
    # handle some error entries
    for y in df.columns:
        if (df[y].dtype == np.object): # text
            s = df[y].replace(error_entries, regex=True)
        else: # numerical
            s = df[y].replace({9999 : np.nan})
        df[y] = s.astype(np.float32)
    return df

def nan_var(df, max_n):
    """
    see how many nan are in the measures
    """
    n_valid = df.dropna(axis=0).shape[0]
    missing_data = (1 - n_valid / max_n)
    return missing_data

def rid_cat(df):
    """
    Remove categorical variables and skewed variables in the final dataset
    """
    var_names = df.columns
    var_keep = []
    for n in var_names:
        if "ACDS_42" == n or "CASI" in n or "YGTSS" in n:
            pass
        else:
            var_keep.append(n)
    return df[var_keep]

def plot_missing_variablewise(missing_percentage, threshold=0.3):
    # plot missing variable-wise
    missing_percentage = pd.DataFrame(missing_percentage, index=["% missing"]).T
    excludes = np.where(missing_percentage > threshold)[0]

    fig = plt.figure(figsize=(8, 5))
    plt.barh(range(missing_percentage.shape[0]),
             missing_percentage.values.squeeze())

    plt.yticks(excludes, missing_percentage.index[excludes])
    plt.vlines(threshold, -1, 41, linestyles=":")
    plt.tight_layout()
    return fig

def plot_missing_subjectwise(subj_missing_portion, threshold=0.2):
    fig = plt.figure(figsize=(8, 5))
    plt.barh(range(len(subj_missing_portion)),
             subj_missing_portion)
    plt.vlines(threshold, -1, len(subj_missing_portion), linestyles=":")
    plt.tight_layout()
    return fig

def clean_variablewise(dict_assessments, subject_list, base_dir, threshold=0.3):
    # Filter by missing data (exclude variables with 30% missing)
    # Investgate missing age range in variables
    # Some behavioural measures were separated for adult and children.
    # Even if the missing data in the full dataset is under 30%,
    # it would be probelmatic to impute those variables if all missing are from the same age group,
    # as there's no data points representing that population.
    def assessment_path(base_dir, key):
        """
        full path of the assessment data
        """
        return list(base_dir.glob(f"*/{key}.csv"))[0]
        
    max_n = len(subject_list)

    concat_df = []
    missing_percentage = {}
    for key, items in dict_assessments.items():
        path = assessment_path(base_dir, key)
        df = get_assessment(path, items)

        df = drop_duplicates(df, subject_list, items)
        if df is None:
            continue

        df = remove_errors(df, error_entries)
        df = rid_cat(df)

        missing_data = nan_var(df, max_n)
        missing_percentage.update({key: missing_data})

        if  missing_data > threshold:  # more than 30% of the data is missing
            print(f"{key}: exclude - more than 30% of the data is not present in the selected sample")
        else:
            concat_df.append(df)

    # Save all variables of interests that can be used in the analysis
    df_master = pd.concat(concat_df, axis=1)
    return df_master, missing_percentage

def clean_subjectwise(df, threshold=0.2):
    # plot missing subject-wise
    # Drop subject with more than 20% of the measures missing
    subj_missing_n = df.isna().sum(axis=1)
    subj_missing_portion = subj_missing_n / df.shape[1]
    n_drop_subj = sum(subj_missing_portion > threshold)
    perc_drop_subj = n_drop_subj / df.shape[0]
    print(f"print dropping  {n_drop_subj} / {100 * perc_drop_subj}% of the current sample")

    drop_idx = subj_missing_portion > threshold
    df_no_cat = df.loc[~drop_idx, :]
    return df_no_cat, subj_missing_portion
