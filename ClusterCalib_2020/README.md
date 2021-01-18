Pre-processing and post-processing scripts used for L1 HGCAL clusters studies done in 2020.

preprocessing directory includes -
1. preprocessing_vUpdated.py - It is similar to the script from 2019 but it stores all types of events including events with L1 3D clusters matched inside the matching radius cone and outside the cone also.
2. preprocessingValidation.py - same preprocessing script but also including reconstruction (matching) efficiency study for clusters.
3. Validation.py - General script looking at validation plots for the pre-processed events.

postprocessing_BDTCalibration directory includes scripts used to perform BDT calibration studies for L1 HGCAL clusters.
It has two directories - 
1. photons -
   a. BDTCalibration_photonsfinal.py - Script used to do the final photon BDT calibration studies (presented in the TDAQ meeting).
   b. BDTCalibration_photons_dev.py - Updated version of BDTCalibration_photonsfinal.py script but with an added feature for the study of custom learning rates (evolving) using the XGBoost library.

2. electrons - 
   a. BDTCalibration_testalle.py - Script used to do the electron BDT calibration studies. This study has not been finalised. There were problems seen with electron calibration results and needs to be understood.


