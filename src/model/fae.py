

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
# import plotly
# import plotly.tools as tls
# import warnings
from collections import defaultdict


# class ForecastAutoencoder(nn.Module):
#     """
#     Class for the densely connected hidden cells version of the model
#     """
#     def __init__(self, structure):
#         """
#         Constructor
#         :param input_dim: Dimension of the inputs
#         :param hidden_dim: Number of hidden units
#         :param output_dim: Dimension of the outputs
#         :param in_seq: Length of the input sequence
#         :param out_seq_length: Length of the output sequence
#         """
#         super(ForecastAutoencoder, self).__init__()

#         input_dim = structure['input_dim']
#         n_layers = structure['n_layers']
#         hidden_dim = structure['hidden_dim']
#         output_dim = structure['output_dim']
#         in_seq = structure['in_seq']
#         out_seq_length = structure['out_seq_length']
#         # device = structure['device']
        
#         # Input dimension of componed inputs and sequences
#         input_dim_comb = input_dim * in_seq

#         # Initialise layers
#         input_layer = [nn.Linear(input_dim_comb, hidden_dim)]
#         for _ in range(out_seq_length-1):
#             input_layer.append(nn.Linear(input_dim_comb + hidden_dim + output_dim, hidden_dim))
#         hidden_layers = [nn.ModuleList(input_layer)]
#         for _ in range(n_layers):
#             hidden_layers.append(nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(out_seq_length)]))
#         self.hidden_layers = nn.ModuleList(hidden_layers)
#         self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

#     def forward(self, input, target=None, is_training=False):
#         """
#         Forward propagation of the dense ForecastNet model
#         :param input: Input data in the form [in_seq, batch_size, input_dim]
#         :param target: Target data in the form [out_seq_length, batch_size, output_dim]
#         :param is_training: If true, use target data for training, else use the previous output.
#         :return: outputs: Forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
#         """
#         assert target is not None or ~is_training, "Must provide y while training."

#         # Format the inputs
#         # input = format_input(input) # to 2d
#         input = input.to(self.device)

#         # Initialise outputs
#         outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
#         # First input
#         next_cell_input = input
#         for i in range(self.out_seq_length):
#             hidden = next_cell_input
#             # Propagate through cell
#             for hidden_layer in self.hidden_layers:
#                 hidden = F.relu(hidden_layer[i](hidden))
#             # Calculate the output
#             output = self.output_layer[i](hidden)
#             outputs[i,:,:] = output
#             # Prepare the next input
#             if is_training:
#                 next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1)
#             else:
#                 next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1)
        
#         return outputs


class ForecastAutoencoder(nn.Module):
    def __init__(self, structure):
        super(ForecastAutoencoder, self).__init__()
        encoder_neurons, decoder_neurons = structure
        alpha = 1.0 # exponential linear unit

        # https://zhuanlan.zhihu.com/p/64990232

        # Encoder
        encoder = []
        for n in zip(encoder_neurons[:-1], encoder_neurons[1:]):
            encoder.append(nn.Linear(*n))
            encoder.append(nn.ELU(alpha=alpha))
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        for n in zip(decoder_neurons[:-2], decoder_neurons[1:-1]):
            decoder.append(nn.Linear(*n))
            decoder.append(nn.ELU(alpha=alpha))
        decoder.append(nn.Linear(*decoder_neurons[-2:]))
        self.decoder = nn.Sequential(*decoder)

        # # Classifier
        # classifier = []
        # for n in zip(classifier_neurons[:-2], classifier_neurons[1:-1]):
        #     classifier.append(nn.Linear(*n))
        #     classifier.append(nn.ELU(alpha=alpha))
        # classifier.append(nn.Linear(*classifier_neurons[-2:]))
        # self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        theta = 1.0 # gradient reveral layer

        # Encoder
        feature = self.encoder(x)
        theta = torch.tensor(theta).detach().requires_grad_(True)
        # reversal_feature = GradientReversalLayer.apply(feature, theta)

        # Decoder
        reconstruction = self.decoder(feature)

        # # # Classifier
        # domain = self.classifier(reversal_feature)

        return reconstruction#, domain


class FAE():
    def __init__(self, config, X_in:np.array, X_out:np.array, D:np.array):
        self.config = config
        self.X_in = X_in
        self.X_out = X_out
        self.D = D

        # Define hyperparameters
        input_size = X_in.shape[1]
        output_size = X_out.shape[1]
        self.structure = config.get_structure(input_size, output_size)
        self.learning_rate = config.learning_rate
        self.n_epochs = config.n_epochs
        self.device = config.device
        # self.lambda_ = config.lambda_
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle

        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _load_batches(self, batch_size=100, shuffle=True):
        idx = np.arange(self.X_in.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        for i in range(int(np.ceil(idx.size/batch_size))):
            x_in = self.X_in[idx[i*batch_size:(i+1)*batch_size]]
            x_out = self.X_out[idx[i*batch_size:(i+1)*batch_size]]
            # d = self.D[idx[i*batch_size:(i+1)*batch_size]]
            yield x_in, x_out#, d
                  
    def _train_one_epoch(self):
        # Define the loss function and optimizer
        reconstruction_criterion = nn.MSELoss()  # Mean Squared Error loss
        # domain_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        batch_losses = []
        for x_in, x_out in self._load_batches(self.batch_size, self.shuffle):
            x_in = torch.from_numpy(x_in).type(torch.FloatTensor).to(self.device)
            x_out = torch.from_numpy(x_out).type(torch.FloatTensor).to(self.device)
            # d = torch.from_numpy(d).type(torch.FloatTensor).to(self.device)
            
            self.model.train()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            reconstruction_ouputs = self.model(x_in)#, domain_ouputs
            
            # Compute the loss
            reconstruction_loss = reconstruction_criterion(reconstruction_ouputs, x_out)
            # domain_loss = domain_criterion(domain_ouputs, d)
            loss = reconstruction_loss# + self.lambda_ * domain_loss
            batch_losses.append([
                reconstruction_loss.item(), 
                # domain_loss.item(), 
                loss.item(), 
                ])
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
        # Find average loss over sequences and batches
        epoch_loss = np.array(batch_losses).mean(axis=0)
        epoch_loss = dict(zip(['reconstruction loss', 'total loss'], epoch_loss))#, 'domain loss'

        return epoch_loss

    def train(self):
        sns.set()
        plt.rcParams['font.family'] = 'serif'
        
        # Create an instance of the Autoencoder
        self.model = ForecastAutoencoder(self.structure)

        history = pd.DataFrame()
        best = defaultdict(lambda: np.inf)
        stopping_criteria_ctr = 0
        stop = False
        start_time = time.time()
        for epoch in range(self.n_epochs):
            epoch_loss = self._train_one_epoch()
            result = {
                'epoch': epoch, 
                'elapsed time': time.time()-start_time, 
                }
            result.update(epoch_loss)

            # Stopping criteria
            if np.all([best[k]<=epoch_loss[k] for k in epoch_loss.keys()]):
                stopping_criteria_ctr += 1
            else:
                result["best"] = "True"
                best = epoch_loss
                stopping_criteria_ctr = 0
            if stopping_criteria_ctr>=self.config.stopping_count:
                stop = True

            # Save history
            history = pd.concat([history, pd.Series(result, name=epoch).to_frame().T], axis=0)

            # Intermittently 
            format_text = lambda t: f'{t:.3f}' if t<10**3 and t>=10**-3 else f'{t:.3e}'
            if epoch%50==0 or epoch==self.config.n_epochs-1 or stop:
                # Save
                history.to_csv(self.config.training_dir+'history.csv', index=False)

                # Save the trained model (fix chinese dir issue)
                with open(self.config.model_dir, mode='wb') as f:
                    torch.save(self.model.state_dict(), f)

                # Plot the training curves
                title = 'training curves'
                fig, ax1 = plt.subplots(figsize=(min(max(epoch/1000, 12), 24), 6))
                # ax2 = ax1.twinx()
                ax1.plot(
                    history['epoch'].values, 
                    history['reconstruction loss'].values, 
                    alpha=.7, 
                    color='b', 
                    label='reconstruction loss', 
                    )
                # ax2.plot(
                #     history['epoch'].values, 
                #     history['domain loss'].values, 
                #     alpha=.7, 
                #     color='r', 
                #     label='domain loss', 
                #     )
                ax1.text(
                    history['epoch'].values[-1], 
                    history['reconstruction loss'].values[-1], 
                    f'reconstruction loss: {format_text(history["reconstruction loss"].values[-1])}\
                    \ntotal loss: {format_text(history["total loss"].values[-1])}',
                    # horizontalalignment='right', 
                    verticalalignment='bottom', 
                    )
                    # \ndomain loss: {format_text(history["domain loss"].values[-1])}\
                ax1.set_ylabel('reconstruction loss')
                # ax2.set_ylabel('domain loss')
                ax1.legend(*ax1.get_legend_handles_labels(), loc='upper left')
                # ax2.legend(*ax2.get_legend_handles_labels(), loc='upper right')
                # ax2.grid(False)
                plt.title(title)
                plt.xlabel('epoch')
                plt.savefig(self.config.training_dir+title+'.png', transparent=True, 
                            bbox_inches='tight', dpi=144)
                # with warnings.catch_warnings():
                #     warnings.filterwarnings("ignore", category=UserWarning)
                #     plotly.offline.plot(tls.mpl_to_plotly(fig), 
                #                         filename=self.config.training_dir+title+'.html', 
                #                         auto_open=False)
                plt.clf()
                plt.close('all')

            # Build epoch messagees
            format_text = lambda t: f'{t:.3f}' if t<10**3 and t>=10**-3 else f'{t:.3e}'
            loss_msg = f', Loss: {", ".join([f"{k} {v}" for k, v in epoch_loss.items()])}'
            etc_msg = ', ETC: %.2f minutes (%.2f seconds)' %\
                ((self.n_epochs - epoch - 1) * (time.time() - start_time) / 60,
                (self.n_epochs - epoch - 1) * (time.time() - start_time))
            print(f'Epoch [{epoch+1}/{self.n_epochs}]{loss_msg}{etc_msg}\t\t', end='\r')

            if stop:
                break

        print()
        print()
        print(f'Elapsed time for training: {round(time.time()-start_time, 2)} seconds')
        for key, value in best.items():
            print(f'Best {key}: {value}')
        print(f'Model saved to {self.config.model_dir}')
        print('-'*80)
        print()

    def load_model(self):
        model = ForecastAutoencoder(self.structure)
        with open(self.config.model_dir, mode='rb') as f:
            model.load_state_dict(torch.load(f, map_location=self.device))

        self.model = model

    def predict(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

        # Reconstruct
        self.model.eval()
        reconstruction_ouputs = self.model(x)#, domain_ouputs
        reconstruction_ouputs = reconstruction_ouputs.detach().numpy()
        # domain_ouputs = domain_ouputs.detach().numpy()
        return reconstruction_ouputs#, domain_ouputs

    def encode(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

        # Encode
        self.model.eval()
        latent_space = self.model.encoder(x)
        latent_space = latent_space.detach().numpy()
        return latent_space





