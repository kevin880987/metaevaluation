import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA, FastICA
from tsquared import HotellingT2
from sklearn.preprocessing import MinMaxScaler

from data_helpers import stack_sequence
from preprocessor import XPreprocessor, YPreprocessor, EPreprocessor, select_feature
from mahalanobis_distance import mahalanobis_distance
from visualization import plot_reconstruction, plot_health_indicator
from model_manager import TorchModelManager, SklearnModelManager


def preprocess(config, holder, checkpoint=True):
    print("=" * 20)
    print("DATA PREPROCESSING")
    print("=" * 20)
    print(f"Experiments: {holder.experiments}")
    print(f"Checkpoint mode: {'Enabled' if checkpoint else 'Disabled (will recompute)'}")
    print("-" * 20)
    
    preprocessor = {}

    # Preprocess data for feature selection
    print("Step 1: Initial data preprocessing for feature selection")

    X_out_ = []
    Y_ = []
    E_ = []
    
    for idx, e in enumerate(holder.experiments, 1):
        print(f"  [{idx:3d}/{len(holder.experiments):3d}] Processing experiment: {e} | Output: {holder.get('X', dataset='train', experiment=e, output_sequence=config.output_sequence).shape} | Target: {holder.get('Y', dataset='train', experiment=e).shape} | Experiment: {holder.get('E', dataset='train', experiment=e).shape}", end="\r")
        
        # X_out
        # X_out = holder.get("X", dataset="train", experiment=e, 
        #                    output_sequence=config.output_sequence)
        X_out = holder.get("X", dataset="train", experiment=e)
        x_out_preprocessor = XPreprocessor()
        X_out = x_out_preprocessor.fit_transform(X_out)
        X_out_.append(X_out)

        # Y
        Y = holder.get("Y", dataset="train", experiment=e)
        y_preprocessor = YPreprocessor()
        Y = y_preprocessor.fit_transform(Y)
        Y_.append(Y)

        # E
        E = holder.get("E", dataset="train", experiment=e)
        e_preprocessor = EPreprocessor()
        E = e_preprocessor.fit_transform(E)
        E_.append(E)

        preprocessor.setdefault("Y", {})[e] = y_preprocessor
        preprocessor.setdefault("E", {})[e] = e_preprocessor
    
    print()  # New line after progress
    
    # # Preprocess time invariant data

    # x_in_preprocessor = XPreprocessor()
    # X_in = holder.get("X", dataset="train", experiment="all", format_2d=True)
    # X_in = x_in_preprocessor.fit_transform(X_in)

    # x_out_preprocessor = XPreprocessor()
    # X_out = holder.get("X", dataset="train", experiment="all", format_2d=True)
    # X_out = x_out_preprocessor.fit_transform(X_out)

    # y_preprocessor = YPreprocessor()
    # Y = holder.get("Y", dataset="train", experiment="all", format_2d=True)
    # Y = y_preprocessor.fit_transform(Y)

    # Select output features
    print("Step 2: Output feature selection")
    if not checkpoint:
        print("  Computing feature selection...")
        selected_out = select_feature(X_out, Y, dir=config.training_dir)
        pd.Series(selected_out, name="Selected Output Features").to_frame()\
            .to_csv(config.training_dir+"selected_output_features.csv", index=False)
        print(f"  Selected {len(selected_out)} output features")
    else:
        print("  Loading existing feature selection...")

    selected_out = pd.read_csv(config.training_dir+"selected_output_features.csv")\
        .values.flatten().tolist()
    selected_out = np.array(selected_out, dtype=str).tolist()
    preprocessor["selected_out"] = selected_out
    print(f"  Output features loaded: {len(selected_out)}/{X_out.shape[1]} features")

    # Preprocess input X based on selected output features for input feature selection
    print("Step 3: Input data preprocessing based on selected output features")
    X_in_ = []
    for idx, e in enumerate(holder.experiments, 1):
        # X_in = holder.get("X", dataset="train", experiment=e, 
        #                   input_sequence=config.input_sequence, selected=selected_out)
        X_in = holder.get("X", dataset="train", experiment=e, selected=selected_out)
        print(f"  [{idx:3d}/{len(holder.experiments):3d}] Processing input data for experiment: {e} | Shape: {X_in.shape}", end="\r")
        # X_in
        x_in_preprocessor = XPreprocessor()
        X_in = x_in_preprocessor.fit_transform(X_in)
        X_in_.append(X_in)

        preprocessor.setdefault("X_in", {})[e] = x_in_preprocessor

    print()  # New line after progress

    # Select input features
    print("Step 4: Input feature selection")
    if not checkpoint:
        print("  Computing input feature selection...")
        selected_in = select_feature(X_in, Y, dir=config.training_dir)
        selected_in += list(set(selected_out)-set(selected_in)) # add output features for reconstruction
        pd.Series(selected_in, name="Selected Input Features").to_frame()\
            .to_csv(config.training_dir+"selected_input_features.csv", index=False)
        print(f"  Selected {len(selected_in)} input features (including output features for reconstruction)")
    else:
        print("  Loading existing input feature selection...")

    selected_in = pd.read_csv(config.training_dir+"selected_input_features.csv")\
        .values.flatten().tolist()
    selected_in = np.array(selected_in, dtype=str).tolist()
    preprocessor["selected_in"] = selected_in
    print(f"  Input features loaded: {len(selected_in)}/{X_in.shape[1]} features")

    # Refit preprocessor using selected features
    print("Step 5: Final preprocessing with selected features")

    # X_in = X_in[selected_in]
    # x_in_preprocessor = XPreprocessor()
    # X_in = x_in_preprocessor.fit_transform(X_in)
    # preprocessor["X_in"] = x_in_preprocessor

    # X_out = X_out[selected_out]
    # x_out_preprocessor = XPreprocessor()
    # X_out = x_out_preprocessor.fit_transform(X_out)
    # preprocessor["X_out"] = x_out_preprocessor

    # preprocessor["Y"] = y_preprocessor

    X_in_ = []
    X_out_ = []
    for idx, e in enumerate(holder.experiments, 1):
        # X_in
        # X_in = holder.get("X", dataset="train", experiment=e, 
        #                   input_sequence=config.input_sequence)[selected_in]
        X_in = holder.get("X", dataset="train", experiment=e)[selected_in]
        x_in_preprocessor = XPreprocessor()
        X_in = x_in_preprocessor.fit_transform(X_in)
        X_in_.append(X_in)

        # X_out
        # X_out = holder.get("X", dataset="train", experiment=e, 
        #                    output_sequence=config.output_sequence)[selected_out]
        X_out = holder.get("X", dataset="train", experiment=e)[selected_out]
        x_out_preprocessor = XPreprocessor()
        X_out = x_out_preprocessor.fit_transform(X_out)
        X_out_.append(X_out)
        
        print(f"  [{idx:3d}/{len(holder.experiments):3d}] Final preprocessing for experiment: {e} | Input: {X_in.shape} | Output: {X_out.shape}", end="\r")

        preprocessor.setdefault("X_in", {})[e] = x_in_preprocessor
        preprocessor.setdefault("X_out", {})[e] = x_out_preprocessor

    print()  # New line after progress

    # X_in_ and X_out_ are dictionaries {experiment: selected and transformed dataframe}
    X_in = holder._agg(X_in_)[selected_in]
    X_out = holder._agg(X_out_)[selected_out]
    Y = holder._agg(Y_)
    E = holder.get("E", dataset="train")

    assert (X_in.index == X_out.index).all(), "Input and output must have same samples"

    print("PREPROCESSING COMPLETED")
    print(f"  Aggregated input data: {X_in.shape}")
    print(f"  Aggregated output data: {X_out.shape}")
    print(f"  Target data: {Y.shape}")
    print("=" * 20)

    return preprocessor, X_in, X_out, Y


def train_models(config, X_in, X_out, checkpoint=True):
    """Train or load all required models"""
    input_sequence = config.input_sequence
    output_sequence = config.output_sequence

    # # Fit/load sklearn models (2D format)
    # X_in_2d, X_out_2d = prepare_data(
    #     X_in, X_out, 
    #     input_sequence, output_sequence, 
    #     format_2d=True, 
    #     )
    # sklearn_model_manager = SklearnModelManager(config)
    # sklearn_models = sklearn_model_manager.fit_all_models(
    #     X_in_2d, feature_names=X_in_2d.columns.tolist(),
    #     force_refit=not checkpoint
    # )
    
    # Train/load PyTorch models (3D format)
    X_in_3d, X_out_3d = prepare_data(
        X_in, X_out, 
        input_sequence, output_sequence, 
        format_2d=False, 
        )
    torch_model_manager = TorchModelManager(config)
    torch_models = torch_model_manager.train_all_models(
        X_in=X_in_3d, X_out=X_out_3d, force_retrain=not checkpoint
    )
    
    return torch_models


def get_health_indicator(config, holder, preprocessor, torch_models):
    """Extract health indicators using trained models"""
    selected_in = preprocessor["selected_in"]
    selected_out = preprocessor["selected_out"]
    input_sequence = config.input_sequence
    output_sequence = config.output_sequence

    print("=" * 20)
    print("EXTRACTING HEALTH INDICATORS")
    print("=" * 20)
    print(f"Experiments to process: {holder.experiments}")
    print(f"Health indicators to extract: {list(config.health_indicators.keys())}")
    print(f"Selected input features: {len(selected_in)}")
    print(f"Selected output features: {len(selected_out)}")
    print("-" * 20)
    
    # Extract health indicator
    hi_dict = {}
    total_experiments = len(holder.experiments)
    
    for experiment_idx, e in enumerate(holder.experiments, 1):
        print(f"PROCESSING EXPERIMENT [{experiment_idx}/{total_experiments}]: {e}")
        print("-" * 60)
        
        hi_dir = config.health_indicator_dir + e + os.sep
        os.makedirs(hi_dir, exist_ok=True)

        # Prepare preprocessor for input and output
        x_in_preprocessor = preprocessor["X_in"][e]
        x_out_preprocessor = preprocessor["X_out"][e]

        # Prepare input and output data
        X_e_in_raw = holder.get("X", experiment=e, selected=selected_in, format_2d=True)
        X_e_in = x_in_preprocessor.transform(X_e_in_raw)
        X_e_out_raw = holder.get("X", experiment=e, selected=selected_out, format_2d=True)
        X_e_out = x_out_preprocessor.transform(X_e_out_raw)

        # Prepare data for model input
        X_e_in_2d, X_e_out_2d = prepare_data(
            X_e_in, X_e_out, 
            input_sequence, output_sequence, 
            format_2d=True, 
            )
        X_e_in_3d, X_e_out_3d = prepare_data(
            X_e_in, X_e_out, 
            input_sequence, output_sequence, 
            format_2d=False, 
            )
        
        # Store mutual index
        idx = X_e_out_2d.index

        # Process each torch model for reconstruction
        print(f"  Generating reconstructions using {len(torch_models)} models...")
        reconstructions = {}
        
        for model_idx, (model_name, model) in enumerate(torch_models.items(), 1):
            recon_dir = config.reconstruction_dir + e + os.sep + model_name + os.sep
            os.makedirs(recon_dir, exist_ok=True)

            # Get reconstructions using correct predict method
            print(f"    [{model_idx:2d}/{len(torch_models):2d}] Generating reconstruction with {model_name.upper()}", end="\r")
            
            # Use predict method which expects 3D numpy arrays
            X_out_hat_3d = model.predict(X_e_in_3d)

            # Align 2D output
            X_out_hat_2d = X_out_hat_3d.reshape(X_out_hat_3d.shape[0], -1)
            X_out_hat_2d = pd.DataFrame(X_out_hat_2d, columns=X_e_out_2d.columns, index=idx)

            # Get reconstruction df
            recon_seq_idx = output_sequence.index(0)
            X_recon_2d = X_out_hat_3d[:, :, recon_seq_idx].squeeze()
            X_recon_2d = pd.DataFrame(X_recon_2d, columns=selected_out, index=idx)

            # Get raw reconstruction df
            X_recon_2d_raw = pd.DataFrame(
                x_out_preprocessor.inverse_transform(X_recon_2d), 
                columns=selected_out,
                index=idx
            )

            # Replace some input with reconstruction
            out_seq_idx = [input_sequence.index(f) for f in output_sequence]
            out_feat_idx = [selected_in.index(f) for f in selected_out]
            X_in_hat_3d = X_e_in_3d.copy()
            X_in_hat_3d[:, out_feat_idx, out_seq_idx] = X_out_hat_3d.squeeze() # squeeze in case some dimensions are 1

            # Store for health indicator calculation
            reconstructions[model_name] = {
                'X_recon_2d_raw': X_recon_2d_raw,
                'X_out_hat_2d': X_out_hat_2d,
                'X_in_hat_3d': X_in_hat_3d,
            }
            
            # Save reconstruction plot
            plot_reconstruction(X_e_out_raw.loc[idx], X_recon_2d_raw, recon_dir, f"{e}, {model_name.upper()}")
                
        print()  # New line after reconstruction progress

        # Train PCA and ICA models
        pca_model = PCA(n_components=None)
        pca_model.fit(X_e_in_2d)
        # sklearn_models["pca"] = pca_model

        ica_model = FastICA(max_iter=10000, n_components=None)
        ica_model.fit(X_e_out_2d)
        # sklearn_models["ica"] = ica_model

        # Calculate health indicators for each model
        print(f"  Calculating {len(config.health_indicators)} health indicators...")
        experiment_hi_results = {}
        
        for hi_idx, hi in enumerate(config.health_indicators, 1):
            health_indicator = None
            
            print(f"    [{hi_idx:2d}/{len(config.health_indicators):2d}] Computing {hi}", end="")
            
            if hi == "PC1":
                print(" | Using PCA (first principal component)", end="\r")
                # pca_model = sklearn_models["pca"]
                # Use 2D data for sklearn models
                health_indicator = pca_model.transform(X_e_in_2d)[:, 0].ravel()

            elif hi == "IC_MD":
                print(" | Using ICA + Mahalanobis Distance with CVAE reconstruction", end="\r")
                # ica_model = sklearn_models["ica"]
                if "cvae" in reconstructions:
                    X_out_hat_2d = reconstructions["cvae"]["X_out_hat_2d"].loc[X_e_out_2d.index]
                    health_indicator = mahalanobis_distance(
                        ica_model.transform(X_e_out_2d),
                        ica_model.transform(X_out_hat_2d)
                    )
                
            elif hi == "IC_HT2":
                print(" | Using ICA + Hotelling's T² with CVAE reconstruction", end="\r")
                # ica_model = sklearn_models["ica"]
                if "cvae" in reconstructions:
                    ica_features = ica_model.transform(X_e_out_2d)
                    hotellings_t2_hi = HotellingT2()
                    hotellings_t2_hi.fit(ica_features)

                    X_out_hat_2d = reconstructions["cvae"]["X_out_hat_2d"].loc[X_e_out_2d.index]
                    ica_recon = ica_model.transform(X_out_hat_2d)
                    health_indicator = hotellings_t2_hi.score_samples(ica_features - ica_recon)

            # CVAE-specific health indicators
            elif hi.startswith("CVAE_") and "cvae" in reconstructions:
                cvae_model = torch_models["cvae"]
                X_out_hat_2d = reconstructions["cvae"]["X_out_hat_2d"]

                if "CVAE_REC_ERR" in hi:
                    print(" | Using CVAE Reconstruction Error", end="\r")
                    health_indicator = np.mean((X_e_out_2d.values - X_out_hat_2d.values) ** 2, axis=1)

                elif "CVAE_REC_MD" in hi:
                    print(" | Using CVAE Reconstruction Mahalanobis Distance", end="\r")
                    health_indicator = mahalanobis_distance(X_e_out_2d.values,  X_out_hat_2d.values)
                    
                elif "CVAE_REC_HT2" in hi:
                    print(" | Using CVAE Reconstruction Hotelling's T²", end="\r")
                    hotellings_t2_hi = HotellingT2()
                    hotellings_t2_hi.fit(X_e_out_2d)
                    health_indicator = hotellings_t2_hi.score_samples(X_e_out_2d - X_out_hat_2d)

                if "_LS_" in hi:
                    latent_space = cvae_model.encode(X_e_in_3d)
                    # Create modified input for latent space comparison
                    X_in_hat_3d = reconstructions["cvae"]["X_in_hat_3d"]
                    latent_space_hat = cvae_model.encode(X_in_hat_3d)

                    if "CVAE_LS_ERR" in hi:
                        print(" | Using CVAE Latent Space Reconstruction Error", end="\r")
                        health_indicator = np.mean((latent_space - latent_space_hat) ** 2, axis=1)

                    elif "CVAE_LS_MD" in hi:
                        print(" | Using CVAE Latent Space Mahalanobis Distance", end="\r")
                        health_indicator = mahalanobis_distance(latent_space, latent_space_hat)

                    elif "CVAE_LS_HT2" in hi:
                        print(" | Using CVAE Latent Space Hotelling's T²", end="\r")
                        hotellings_t2_ls = HotellingT2()
                        hotellings_t2_ls.fit(latent_space)
                        health_indicator = hotellings_t2_ls.score_samples(latent_space - latent_space_hat)

                if "_REC_LS_" in hi:
                    if latent_space.shape[1]<=X_out_hat_2d.shape[1]:
                        pca = PCA(n_components=latent_space.shape[1])
                        pca.fit(X_out_hat_2d)
                        rec = pca.transform(X_e_out_2d)-pca.transform(X_out_hat_2d)
                        ls = latent_space-latent_space_hat
                        scalar = MinMaxScaler().fit(ls)
                        rec = scalar.transform(rec)
                        
                        hotellings_t2_hi = HotellingT2()
                        hotellings_t2_hi.fit(latent_space)

                    else:
                        pca = PCA(n_components=X_e_out_2d.shape[1])
                        pca.fit(latent_space)
                        rec = np.array(X_e_out_2d)-np.array(X_out_hat_2d)
                        ls = pca.transform(latent_space)-pca.transform(latent_space_hat)

                        scalar = MinMaxScaler().fit(rec)
                        ls = scalar.transform(ls)

                        hotellings_t2_hi = HotellingT2()
                        hotellings_t2_hi.fit(X_e_out_2d)
                    
                    # Mahalanobis distance of reconstruction and latent space
                    if hi=="CVAE_REC_LS_MD":
                        health_indicator = mahalanobis_distance(rec, ls)

                    # Hotelling"s T-square of reconstruction and latent space
                    # https://github.com/cetic/tsquared
                    elif hi=="CVAE_REC_LS_HT2":
                        health_indicator = hotellings_t2_hi.score_samples(rec-ls)

            # CAE-M specific health indicators
            elif hi.startswith("CAEM_") and "cae_m" in reconstructions:
                caem_model = torch_models["cae_m"]
                X_out_hat_2d = reconstructions["cae_m"]["X_out_hat_2d"]

                if hi == "CAEM_REC_ERR":
                    print(" | Using CAE-M Reconstruction Error", end="\r")
                    health_indicator = np.mean((X_e_out_2d.values - X_out_hat_2d.values) ** 2, axis=1)
                    
                elif hi == "CAEM_REC_MD":
                    print(" | Using CAE-M Reconstruction Mahalanobis Distance", end="\r")
                    health_indicator = mahalanobis_distance(X_e_out_2d.values, X_out_hat_2d.values)

                elif "CAEM_REC_HT2" in hi:
                    print(" | Using CAEM Reconstruction Hotelling's T²", end="\r")
                    hotellings_t2 = HotellingT2()
                    hotellings_t2.fit(X_e_out_2d)
                    health_indicator = hotellings_t2.score_samples(X_e_out_2d - X_out_hat_2d)

                if "_LS_" in hi:
                    latent_space = cvae_model.encode(X_e_in_3d)
                    # Create modified input for latent space comparison
                    X_in_hat_3d = reconstructions["cvae"]["X_in_hat_3d"]
                    latent_space_hat = cvae_model.encode(X_in_hat_3d)

                    if "CAEM_LS_ERR" in hi:
                        print(" | Using CAEM Latent Space Reconstruction Error", end="\r")
                        health_indicator = np.mean((latent_space - latent_space_hat) ** 2, axis=1)

                    elif "CAEM_LS_MD" in hi:
                        print(" | Using CAEM Latent Space Mahalanobis Distance", end="\r")
                        health_indicator = mahalanobis_distance(latent_space, latent_space_hat)

                    elif "CAEM_LS_HT2" in hi:
                        print(" | Using CAEM Latent Space Hotelling's T²", end="\r")
                        hotellings_t2_ls = HotellingT2()
                        hotellings_t2_ls.fit(latent_space)
                        health_indicator = hotellings_t2_ls.score_samples(latent_space - latent_space_hat)

                elif hi == "CAEM_ANOM":
                    print(" | Using CAE-M Anomaly Score", end="\r")
                    mse = np.mean((X_e_out_2d.values - X_out_hat_2d.values) ** 2, axis=1)
                    min_mse = np.min(mse)
                    max_mse = np.max(mse)
                    health_indicator = (mse - min_mse) / (max_mse - min_mse + 1e-8)
                        
            # VQ-VAE specific health indicators  
            elif hi.startswith("VQVAE_") and "vq_vae" in torch_models:
                vqvae_model = torch_models["vq_vae"]
                X_out_hat_2d = reconstructions["vq_vae"]["X_out_hat_2d"]

                if hi == "VQVAE_REC_ERR" and "vq_vae" in reconstructions:
                    print(" | Using VQ-VAE Reconstruction Error", end="\r")
                    health_indicator = np.mean((X_e_out_2d.values - X_out_hat_2d.values) ** 2, axis=1)

                if hi == "VQVAE_REC_MD" and "vq_vae" in reconstructions:
                    print(" | Using VQ-VAE Reconstruction Mahalanobis Distance", end="\r")
                    health_indicator = mahalanobis_distance(X_e_out_2d.values, X_out_hat_2d.values)

                if hi == "VQVAE_REC_HT2" and "vq_vae" in reconstructions:
                    print(" | Using VQ-VAE Reconstruction Hotelling's T²", end="\r")
                    hotellings_t2_vq = HotellingT2()
                    hotellings_t2_vq.fit(X_e_out_2d)
                    health_indicator = hotellings_t2_vq.score_samples(X_e_out_2d - X_out_hat_2d)

                if "_QUANT_" in hi:
                    latent_space = vqvae_model.encode(X_e_in_3d)
                    quantization = vqvae_model.encode_latent(X_e_in_3d)
                    n_samples = latent_space.shape[0]
                    latent_space = latent_space.reshape(n_samples, -1)
                    quantization = quantization.reshape(n_samples, -1)

                    if hi == "VQVAE_QUANT_ERR":
                        print(" | Using VQ-VAE Quantization Error", end="\r")
                        health_indicator = np.mean((latent_space - quantization) ** 2, axis=1)

                    elif hi == "VQVAE_QUANT_MD":
                        print(" | Using VQ-VAE Quantization Mahalanobis Distance", end="\r")
                        health_indicator = mahalanobis_distance(latent_space, quantization)

                    elif "VQVAE_QUANT_HT2" in hi:
                        print(" | Using VQ-VAE Quantization Hotelling's T²", end="\r")
                        hotellings_t2_vq = HotellingT2()
                        hotellings_t2_vq.fit(latent_space)
                        health_indicator = hotellings_t2_vq.score_samples(latent_space - quantization)

            # Save and plot health indicator if calculated
            if health_indicator is not None:
                # Ensure proper indexing
                if len(health_indicator) == len(idx):
                    hi_index = idx
                else:
                    # Truncate or pad as needed
                    min_len = min(len(health_indicator), len(X_e_out_2d))
                    health_indicator = health_indicator[:min_len]
                    hi_index = X_e_out_2d.index[:min_len]
                
                health_indicator_df = pd.DataFrame(
                    health_indicator.reshape(-1, 1) if health_indicator.ndim == 1 else health_indicator, 
                    index=hi_index, 
                    columns=[hi]
                )
                experiment_hi_results[hi] = health_indicator_df
                plot_health_indicator(health_indicator_df, hi_dir, e, hi)
                # print(f" | SUCCESS: {len(health_indicator)} samples")
            
            else:
                raise ValueError(f"Unknow health indicator {hi}.")
                    
        print()

        # Save experiment results
        if experiment_hi_results:
            # Combine all health indicators for this experiment
            experiment_df = pd.concat([hi_df for hi_df in experiment_hi_results.values()], axis=1)
            hi_dict[e] = experiment_df
            experiment_df.to_csv(hi_dir + f"{e}.csv")
            print(f"  Experiment {e} results saved: {len(experiment_hi_results)} health indicators")
        else:
            print(f"  No health indicators calculated for experiment {e}")

    print("HEALTH INDICATOR EXTRACTION COMPLETED")
    print(f"  Processed {len(holder.experiments)} experiments")
    print(f"  Attempted {len(config.health_indicators)} health indicators per experiment")
    print(f"  Results saved to: {config.health_indicator_dir}")
    print("=" * 20)

    return hi_dict
    

def prepare_data(X_in, X_out, input_sequence, output_sequence, format_2d=False):
    if max(output_sequence) > 0:
        idx = X_out.index[max(input_sequence):-max(output_sequence)]
    else:
        idx = X_out.index[max(input_sequence):]

    if format_2d:
        # Prepare 2D format for Sklearn models
        X_in_2d = stack_sequence(X_in, input_sequence, [], idx, format_2d=True)
        X_out_2d = stack_sequence(X_out, [], output_sequence, idx, format_2d=True)
        return X_in_2d, X_out_2d
    else:
        # Prepare 3D format for PyTorch models
        X_in_3d = stack_sequence(X_in, input_sequence, [], idx, format_2d=False)
        X_out_3d = stack_sequence(X_out, [], output_sequence, idx, format_2d=False)
        return X_in_3d, X_out_3d


def extract(config, holder, checkpoint=True):
    """Main extraction function"""
    # Preprocess
    preprocessor, X_in, X_out, Y = preprocess(
        config, holder, checkpoint, 
        )

    # Train/load models
    # checkpoint=False
    torch_models = train_models(
        config, X_in, X_out, checkpoint, 
        )

    # Get health indicators
    hi_dict = get_health_indicator(
        config, holder, preprocessor, 
        torch_models, 
        )

    return hi_dict

