

import os
import numpy as np
import shutil
import torch


class Configuration():
    def __init__(self):
        self.dataset = "NTUST" # "ieee-phm-2012_Learning_set_acc" # "auo_vibration_of_bearings" # "ieee-phm-2012_Learning_set_temp" # "auo_vibration_of_bearings_features" # "ieee-phm-2012" # "auo_vibration_of_bearings_tb213" # "t0_robot" # 
        self.scenario = "cvae 20240313" # "full_comparison" # 
        # self.model = "cvae" # "vqvae" # "cae_m" # "fae" # "ae" # "daae" # 
        self.health_indicators = {
            "PC1": "PC₁",

            "IC_MD": "ICA Mahalanobis Distance",
            "IC_HT2": "ICA Hotelling T²",

            "CVAE_REC_ERR": "CVAE Reconstruction Error",
            "CVAE_REC_MD": "CVAE Reconstruction Mahalanobis Distance",
            "CVAE_REC_HT2": "CVAE Reconstruction Hotelling T²",

            "CVAE_LS_MD": "CVAE Latent Space Mahalanobis Distance",
            "CVAE_LS_HT2": "CVAE Latent Space Hotelling T²",

            "CVAE_REC_LS_MD": "CVAE Hybrid Mahalanobis Distance",
            "CVAE_REC_LS_HT2": "CVAE Hybrid Hotelling T²",

            "CAEM_REC_MD": "CAE-M Reconstruction Mahalanobis Distance",
            "CAEM_REC_ERR": "CAE-M Reconstruction Error",
            "CAEM_LS_MD": "CAE-M Latent Space Mahalanobis Distance",
            "CAEM_LS_ERR": "CAE-M Latent Space Reconstruction Error",
            "CAEM_ANOM": "CAE-M Anomaly Score",

            "VQVAE_REC_MD": "VQ-VAE Reconstruction Mahalanobis Distance", 
            "VQVAE_REC_ERR": "VQ-VAE Reconstruction Error",
            "VQVAE_REC_HT2": "VQ-VAE Reconstruction Hotelling's T²",
            "VQVAE_QUANT_MD": "VQ-VAE Quantized Latent Representation",
            "VQVAE_QUANT_ERR": "VQ-VAE Quantized Latent Representation Error",
            }
        self.deterioration_method = "EWCD" # "EWVOI" # "EWVOI+EWCD" # "EW" # 

        # Parameters
        self.test_ratio = 0.5
        # self.window_size = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42

        # Deterioration modeling parameters
        self.n_paths = 10000
        self.n_paths_shown = 10

        # Paths
        root = os.path.abspath("..")+os.sep
        self.data_dir = root+"data"+os.sep+self.dataset+os.sep
        
        self.result_dir = root+os.sep+"result"+os.sep+self.dataset+os.sep+self.scenario+os.sep
        self.original_signal_dir = root+os.sep+"result"+os.sep+self.dataset+os.sep+"original signal"+os.sep
        self.training_dir = self.result_dir+"training"+os.sep
        self.reconstruction_dir = self.result_dir+"reconstruction"+os.sep
        self.health_indicator_dir = self.result_dir+"health indicator"+os.sep
        # self.health_indicator_dir = dict((
        #     (hi, self.result_dir+"health indicator"+os.sep+hi+os.sep) for hi in self.health_indicators
        #     ))
        
        self.deterioration_modeling_dir = self.result_dir+"deterioration modeling"+os.sep+self.deterioration_method+os.sep

        self._make_directories([
            self.original_signal_dir, 
            self.training_dir, 
            self.reconstruction_dir, 
            self.health_indicator_dir, 
            # *self.health_indicator_dir.values(), 
            self.deterioration_modeling_dir, 
            ])
        
        self.backup()
        self._set()
        self._seed_everything()

    def _clear_directories(self, dirs):
        for dir in dirs:
            os.rmdir(dir)
        
    def _make_directories(self, dirs, clear=False):
        if clear:
            # Clear directories
            self._clear_directories(dirs)
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def _seed_everything(self):    
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

    def _set(self):
        self.input_sequence=[0, 1, 2, 3]
        assert 0 in self.input_sequence, "0 must be in input_sequence"
        self.output_sequence=[0]
        assert 0 in self.output_sequence, "0 must be in output_sequence"
        assert (s in self.input_sequence for s in self.output_sequence), "output_sequence must be a subset of input_sequence"

        # Dataset paramters
        if self.dataset=="ieee-phm-2012":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 0.01 * 1000 # 1 # 25.6 * 1000 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 0.01 # 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 0.001 # 0.01 # 0.04 # 
            self.alarm_point = -1000 # 4000 # 8000 # 
            self.failure_point = -1 # 5400 # 10000 # 
            self.descrete_window_size = 30 # 256 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 5.5, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 200000, # One time corrective cost
                "I_penalty": 2, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 10+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

        elif self.dataset=="ieee-phm-2012_Learning_set_acc":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 25.6 * 1000 # 1 # 25.6 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 0.01 # 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 0 # 0.01 # 0.05 # 0.001 # 0.04 # 
            self.alarm_point = -1000 # 4000 # 8000 # 
            self.failure_point = -1 # 5400 # 10000 # 
            self.descrete_window_size = 30 # 256 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 200, # 550, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 300000, # 200000, # One time corrective cost
                "I_penalty": 2, # 200, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 30+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

        elif self.dataset=="ieee-phm-2012_Learning_set_temp":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 0.01 * 1000 # 1 # 25.6 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 0.01 # 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 550 # 0.001 # 0.01 # 0.04 # 
            self.alarm_point = -300 # 4000 # 8000 # 
            self.failure_point = -1 # 5400 # 10000 # 
            self.descrete_window_size = 30 # 256 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 5.5, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 200000, # One time corrective cost
                "I_penalty": 2, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 10+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

        elif self.dataset=="auo_vibration_of_bearings":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 25.6 * 1000 # 2092 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 0.01 # 0.04 # 
            self.alarm_point = -1000 # 
            self.failure_point = -1 # 10000 # 
            self.descrete_window_size = 30 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 5.5, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 200000, # One time corrective cost
                "I_penalty": 2, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 30+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

        elif self.dataset=="auo_vibration_of_bearings_features":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 25.6 * 1000 # 2092 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 0.01 # 0.04 # 
            self.alarm_point = -1000 # 
            self.failure_point = -1 # 10000 # 
            self.descrete_window_size = 30 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 5.5, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 200000, # One time corrective cost
                "I_penalty": 2, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 30+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

        elif self.dataset=="NTUST":
            # Simulation (Note that this in the sense of the data, not sensor time.)
            self.frequency = 25.6 # 2092 # (float): sampling rate, sampling frequency, observations per time period (Hz).
            # self.dt = 1/self.frequency # (float): time step per observation for simulation.
            self.epsilon = 0.01 # 0.04 # 
            self.alarm_point = -1000 # 
            self.failure_point = -1 # 10000 # 
            self.descrete_window_size = 30 # 5 # 

            # Real options
            self.option_valuation = {
                "R_prix": 55, # Hourly revenue of extending operation
                "C_dm": 2000, # Preventive cost per deterioration level
                "C_rep": 200000, # One time corrective cost
                "I_penalty": 2, # Penalties per hour
                "delta_i": 30, # Opportunities maintenance interval
                "I": np.arange(1, 30+1) # Candidate ith time interval to maintain after the alarm (for real options analysis)
                }

    def backup(self):
        # Create backup directory
        backup_dir = os.path.join(self.result_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Track how many files we copy
        copied_count = 0
        
        # Walk through all directories under current working directory
        for root, dirs, files in os.walk(os.getcwd()):
            # For each Python file
            for file in files:
                if file.endswith(".py"):
                    # Get the full path of the source file
                    source_file = os.path.join(root, file)
                    
                    # Create relative path to preserve directory structure
                    rel_path = os.path.relpath(root, os.getcwd())
                    
                    # Create target directory with same structure
                    if rel_path != ".":
                        target_dir = os.path.join(backup_dir, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        # Destination includes the directory structure
                        destination = os.path.join(target_dir, file)
                    else:
                        # Files in the root directory
                        destination = os.path.join(backup_dir, file)
                    
                    # Copy the file
                    shutil.copy2(source_file, destination)
                    copied_count += 1
        
        print(f"Backed up {copied_count} Python files to {backup_dir}")
