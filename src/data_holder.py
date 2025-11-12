

import os
import pandas as pd
import numpy as np
from data_helpers import save_pkl, load_pkl, sklearn_to_pytorch_format, stack_sequence#, get_feature_names_for_sequence#, pytorch_to_sklearn_format


# self=DataHolder()
# self=holder
# dir=config.data_dir
class DataHolder():
    def __init__(self):
        self.original_features = None  # Store original feature names
        self.experiments = None
        self.item = {}
        self.idx = {}
        self.input_sequence = None
        self.output_sequence = None

    def _split(self, idx, test_ratio):
        test = int(len(idx)*test_ratio)
        train_id, test_id = idx[:-test].tolist(), idx[-test:].tolist()
        return train_id, test_id

    def _agg(self, iterable):
        if type(iterable)==dict:
            iterable = iterable.values()
        output = pd.concat(iterable)
        output = output.dropna(axis=0)
        output = output.reset_index(drop=True)
        return output

    def read_files(self, dir):
        self.item = {}
        signals = []
        for p, n in [(f.path, f.name) for f in os.scandir(dir) if f.is_dir()]:
            # if n=='Bearing1_1_acc': break
            # Load csv
            x = pd.read_csv(os.path.join(p, 'X.csv'), index_col=0).sort_index()
            y = pd.read_csv(os.path.join(p, 'Y.csv'), index_col=0).sort_index()
            e = pd.DataFrame([n]*x.shape[0], index=x.index, columns=['experiment'])
            assert np.all(x.index==y.index), f'y index incompatible in {n}'
            assert np.all(x.index==e.index), f'e index incompatible in {n}'

            signals.append(x.columns.to_list())

            # Save by experiment
            self.item.setdefault('X', {}).setdefault(n, x)
            self.item.setdefault('Y', {}).setdefault(n, y)
            self.item.setdefault('E', {}).setdefault(n, e)
        
        assert np.all([np.all(f1==f2) for f1, f2 in zip(signals[:-1], signals[1:])]), "All experiments must have the same features"
        self.original_features = signals[0]  # Store original feature names
        self.experiments = list(self.item['X'].keys())

    def prepare(self, test_ratio=.2, input_sequence=[0], output_sequence=[0]):
        """
        Prepare data with sequence information and train/test split
        
        Args:
            test_ratio: Ratio for test split
            input_sequence: List of input sequence positions (e.g., [0, 1, 2])
            output_sequence: List of output sequence positions (e.g., [0])
        """
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence
        assert max(output_sequence) <= max(input_sequence), "Output sequence cannot be longer than input sequence"

        self.idx = {}
        
        for e in self.experiments:
            # Use 2D format temporarily for splitting (maintaining compatibility)
            X = stack_sequence(self.item['X'][e], input_sequence, output_sequence, format_2d=True)

            # Split data
            train_id, test_id = self._split(X.index, test_ratio)

            # Save by experiment
            self.idx.setdefault('all', {}).setdefault(e, train_id + test_id)
            self.idx.setdefault('train', {}).setdefault(e, train_id)
            self.idx.setdefault('test', {}).setdefault(e, test_id)

    def get(self, item, dataset='all', experiment='all', 
            input_sequence=[0], output_sequence=[0], 
            selected=None, format_2d=True):
        """
        Get data in specified format
        
        Args:
            item: 'X', 'Y', 'E'
            dataset: 'all', 'train', 'test'
            experiment: 'all', or other existing experiments
            input_sequence: Input sequence positions
            output_sequence: Output sequence positions  
            selected: Selected feature names
            format_2d: If True, return 2D format for sklearn compatibility
                      If False (default), return 3D format for PyTorch models
        
        Returns:
            For 'X' with format_2d=False: 3D numpy array [samples, features, sequence_length]
            For 'X' with format_2d=True: 2D DataFrame [samples, features * sequence_length]
            For 'Y' and 'E': Always 2D (unchanged behavior)
        """
        assert max(input_sequence) <= max(self.input_sequence), f'Sequence should not be larger than {self.input_sequence}, got {input_sequence}.'
        assert max(output_sequence) <= max(self.output_sequence), f'Sequence should not be larger than {self.output_sequence}, got {output_sequence}.'
        
        if experiment == 'all':
            experiments = self.experiments
        else:
            experiments = [experiment]

        if item == 'X':
            # Handle X data with format options
            data_list = []
            for e in experiments:
                if selected is None:
                    df = self.item[item][e]
                else:
                    df = self.item[item][e][selected]
                
                # Get data in requested format
                data = stack_sequence(df, input_sequence, output_sequence, 
                                          self.idx[dataset][e], format_2d=format_2d)
                data_list.append(data)
            
            if format_2d:
                # Return 2D DataFrame for sklearn with correct feature names
                df_agg = self._agg(data_list)
                return df_agg
            else:
                # Return 3D numpy array for PyTorch
                return np.concatenate(data_list, axis=0)
                
        else:
            # Handle Y and E data (always 2D)
            dfs = []
            for e in experiments:
                if selected is None:
                    df = self.item[item][e]
                else:
                    df = self.item[item][e][np.array(selected, dtype=str)]
                dfs.append(stack_sequence(df, input_sequence, output_sequence, 
                                          self.idx[dataset][e], format_2d=True))

            if item == 'Y':
                return self._agg(dfs)
            elif item == 'E':
                return pd.get_dummies(self._agg(dfs))
    
    # def get_feature_names(self, input_sequence=None, output_sequence=None, selected=None):
    #     """
    #     Get feature names for sequence data in sklearn 2D format
        
    #     Args:
    #         input_sequence: Input sequence positions (uses self.input_sequence if None)
    #         output_sequence: Output sequence positions (uses self.output_sequence if None) 
    #         selected: Selected feature names (uses all features if None)
        
    #     Returns:
    #         List of feature names in sklearn 2D format
    #     """
    #     if input_sequence is None:
    #         input_sequence = self.input_sequence
    #     if output_sequence is None:
    #         output_sequence = self.output_sequence
    #     if selected is None:
    #         selected = self.original_features
        
    #     # Create sequence positions 
    #     seq = sorted(list(set([-s for s in input_sequence] + output_sequence)))
        
    #     return get_feature_names_for_sequence(selected, seq)
    
    # def get_original_features(self):
    #     """Get the original feature names before sequence expansion"""
    #     return self.original_features.copy() if self.original_features else []

