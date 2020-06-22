from pathlib import Path
import json

import pandas as pd
import s3fs

import boto3

from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing

# Constants
SESSIONS = ['BAS1']
SCANS = ['anat', 'func', 'dwi', 'fmap']
# Mapping colloquial names for the series to BIDS names.
SERIES_MAP = {
'CHECKERBOARD1400':'task-CHECKERBOARD_acq-1400',
'CHECKERBOARD645':'task-CHECKERBOARD_acq-645',
'RESTCAP':'task-rest_acq-CAP',
'REST1400':'task-rest_acq-1400',
'BREATHHOLD1400':'task-BREATHHOLD_acq-1400',
'REST645':'task-rest_acq-645',
'RESTPCASL':'task-rest_pcasl',
'DMNTRACKINGTEST':'task-DMNTRACKINGTEST',
'DMNTRACKINGTRAIN':'task-DMNTRACKINGTRAIN',
'MASK':'mask',
'MSIT':'task-MSIT',
'PEER1':'task-PEER1',
'PEER2':'task-PEER2',
'MORALDILEMMA':'task-MORALDILEMMA'
}

s3_bucket_name = 'fcp-indi'
cpac_prefix = 'data/Projects/RocklandSample/Outputs/C-PAC/'
bids_prefix = 'data/Projects/RocklandSample/RawDataBIDS/'

s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
s3_bucket = s3.Bucket(s3_bucket_name)
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# generate a list of participants from the phenotype data
p = Path.home() / "projects/gradient-physiology"
participants = pd.read_csv(p / "data/derivatives/phenotype/subset_age.tsv",
                           delimiter='\t', index_col=0).index.tolist()

# generate a list of subjects with preporcessed data
participants_flt = {}
for subj in participants:
    scans = []
    s3_keys = s3_bucket.objects.filter(
        Prefix=cpac_prefix + f"sub-{subj}/output/pipeline_analysis_nuisance")
    s3_keylist = [key.key for key in s3_keys]
    for i in s3_keylist:
        if "BAS1/functional_to_standard/_scan_rest_acq-CAP/" in i:
            scans.append(i)
        elif "BAS1/functional_brain_mask_to_standard/_scan_rest_acq-CAP/" in i:
            scans.append(i)

    # physiology data - the latest release has no labels....
    # contacting CMI
    s3_keys = s3_bucket.objects.filter(Prefix=bids_prefix
                                            + f"sub-{subj}/")
    s3_keylist = [key.key for key in s3_keys]
    for i in s3_keylist:
        if "_ses-DS2_task-rest_acq-CAP_physio.json" in i:
            ch = pd.read_json(f"s3://{s3_bucket_name}/{i}")
            ch = ch["Columns"].tolist()
            if "cardiac" in ch:
                scans.append(i)

        elif "_ses-DS2_task-rest_acq-CAP_physio.tsv.gz" in i:
            scans.append(i)

    if len(scans) != 5:
        print(f"{subj}: incomplete collection {len(scans)} / 5")
    else:
        print(f"{subj}: keep")
        participants_flt[subj] = scans

with open(p / "data/neuroimaging_path_CAP.json", 'w') as f:
    json.dump(participants_flt, f, indent=2)
