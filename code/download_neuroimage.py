import os
from pathlib import Path
import json

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.handlers import disable_signing

s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
s3_bucket = s3.Bucket(s3_bucket_name)
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

p = Path.home() / "projects/physiology-gradient-marker"

# download data
with open(p / "data/neuroimaging_path_645.json", 'r') as f:
    participants = json.load(f)

for subject in list(participants.keys()):
    fmri_path = p / "data/derivatives" / "C-PAC" / f"sub-{subject}"
    physio_path = p / "data/derivatives" / "physiology" / f"sub-{subject}"
    for path in [fmri_path, physio_path]:
        if not path.exists():
            os.makedirs(path)

    items = participants[subject]
    for s3_path in items:
        filename = s3_path.split("/")
        if filename[-1] == "residuals_antswarp.nii.gz":
            filename = f"sub-{subject}{filename[-3]}{filename[-2]}.nii.gz"
        else:
            filename = filename[-1]
        
        if "nii.gz" in filename:
            download_file = fmri_path / filename
            with open(download_file, 'wb') as f:
                s3_client.download_fileobj(s3_bucket_name, s3_path, f)

        else:
            download_file = physio_path / filename
            with open(download_file, 'wb') as f:
                s3_client.download_fileobj(s3_bucket_name, s3_path, f)