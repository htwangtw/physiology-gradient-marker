from setuptools import setup

setup(
   name='physiology_gradient_marker',
   version='0.0.1',
   description='Utilities of this project',
   author='Hao-Ting Wang',
   author_email='htwangtw@gmail.com',
   packages=['physiology_gradient_marker'],  #same as name
   install_requires=['numpy>=1.18', 'scipy>=1.4', 'pandas>=1.0', 
                     'matplotlib>=3.2', 'seaborn>=0.10',
                     'nibabel>=3.0', 'nilearn>=0.6',
                     'boto3', 'botocore', 
                     'systole>=0.1', 'detecta'], 
#    scripts=[
#             'scripts/dataset.py',
#            ]
)

