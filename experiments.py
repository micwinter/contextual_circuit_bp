import os
import numpy as np


class experiments():
    """Class for experiments."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def globals(self):
        """Globals."""
        return {
            'batch_size': 64,  # Train/val batch size.
            'data_augmentations': [None],  # Random_crop, etc.
            'epochs': 200,
            'shuffle': True,  # Shuffle data.
            'validation_iters': 500,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False  # Stop training if the loss stops improving.
        }

    def add_globals(self, exp):
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def one_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'one_layer_conv_mlp'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-5],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual_div_norm_no_reg'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_1'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_3'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_4'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_5'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_6'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_7'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_8'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_9'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_10'),
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def one_layer_conv_mlp_all_variants(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'one_layer_conv_mlp'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-5],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual'),
                os.path.join(model_folder, 'contextual_div_norm'),
                os.path.join(model_folder, 'contextual_div_norm_no_reg'),
                os.path.join(model_folder, 'contextual_l2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_10'),
                os.path.join(model_folder, 'contextual_frozen_eCRF_connectivity_l2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2'),
                os.path.join(model_folder, 'contextual_frozen_eCRF_connectivity_l2'),
                os.path.join(model_folder, 'contextual_rnn_1'),
                os.path.join(model_folder, 'contextual_rnn_2'),
                os.path.join(model_folder, 'contextual_rnn_3'),
                os.path.join(model_folder, 'contextual_rnn_4'),
                os.path.join(model_folder, 'contextual_rnn_5'),
                os.path.join(model_folder, 'contextual_rnn_1_l2'),
                os.path.join(model_folder, 'contextual_rnn_2_l2'),
                os.path.join(model_folder, 'contextual_rnn_3_l2'),
                os.path.join(model_folder, 'contextual_rnn_4_l2'),
                os.path.join(model_folder, 'contextual_rnn_5_l2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_1'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_3'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_4'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_5'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_6'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_7'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_8'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_9'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_10')
            ],
            'dataset': ['cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def two_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'two_layer_conv_mlp'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-5],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual'),
                os.path.join(model_folder, 'contextual_rnn'),
                os.path.join(model_folder, 'contextual_rnn_no_relu'),
                # os.path.join(model_folder, 'contextual_selu'),
                # os.path.join(model_folder, 'contextual_rnn_selu'),
            ],
            'dataset': ['cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_conv(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_conv'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],
            'regularization_strength': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_fc(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_fc'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['sgd'],
            'regularization_type': [None],
            'regularization_strength': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_res(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_res'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],
            'regularization_strength': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment