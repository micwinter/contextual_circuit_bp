"""Class to specify all DNN experiments."""
import os
import numpy as np


class experiments():
    """Class for experiments."""

    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""
        return hasattr(self, name)

    def globals(self):
        """Global variables for all experiments."""
        return {
            'batch_size': 64,  # Train/val batch size.
            'data_augmentations': [
                [
                    'random_crop',
                    'left_right'
                ]
            ],  # TODO: document all data augmentations.
            'epochs': 200,
            'shuffle': True,  # Shuffle data.
            'validation_iters': 5000,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False,  # Stop training if the loss stops improving.
            'save_weights': False,  # Save model weights on validation evals.
        }

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def perceptual_iq_hp_optimization(self):
        """Each key in experiment_dict must be manually added to the schema.

        If using grid-search -- put hps in lists.
        If using hp-optim, do not use lists except for domains.
        """
        model_folder = 'perceptual_iq_hp_optimization'
        exp = {
            'experiment_name': model_folder,
            'hp_optim': 'gpyopt',
            'hp_multiple': 10,
            'lr': 1e-4,
            'lr_domain': [1e-1, 1e-5],
            'loss_function': None,  # Leave as None to use dataset default
            'optimizer': 'adam',
            'regularization_type': None,  # [None, 'l1', 'l2'],
            'regularization_strength': 1e-5,
            'regularization_strength_domain': [1e-1, 1e-7],
            # 'timesteps': True,
            'model_struct': [
                os.path.join(model_folder, 'divisive_1l'),
                os.path.join(model_folder, 'layer_1l'),
                os.path.join(model_folder, 'divisive_2l'),
                os.path.join(model_folder, 'layer_2l'),
            ],
            'dataset': 'ChallengeDB_release'
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

    def coco_cnn(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'coco_cnn'
        exp = {
            'experiment_name': [model_folder],
            'lr': [3e-4],
            'loss_function': ['sigmoid'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'cnn'),
                os.path.join(model_folder, 'contextual_cnn')
            ],
            'dataset': ['coco_2014']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 200
        exp['batch_size'] = 16  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def challengedb_cnns(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'challengedb_cnns'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'cnn'),
                os.path.join(model_folder, 'contextual_cnn')
            ],
            'dataset': ['ChallengeDB_release']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 200
        exp['batch_size'] = 8  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def contextual_model_paper(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'contextual_model_paper'
        exp = {
            'experiment_name': [model_folder],
            'lr': [5e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'q_t': [1e-3, 1e-1],
            'p_t': [1e-2, 1e-1, 1],
            't_t': [1e-2, 1e-1, 1],
            'timesteps': [5, 10],
            'model_struct': [
                # os.path.join(model_folder, 'divisive_paper_rfs'),
                os.path.join(model_folder, 'contextual_paper_rfs'),
                # os.path.join(model_folder, 'divisive'),
                # os.path.join(model_folder, 'contextual'),
            ],
            'dataset': ['contextual_model_multi_stimuli']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [[None]]
        exp['epochs'] = 1000
        exp['save_weights'] = True
        return exp

    def ALLEN_selected_cells_1(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_selected_cells_1'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'adj_norm_conv2d'),
                os.path.join(model_folder, 'scalar_norm_conv2d'),
                os.path.join(model_folder, 'vector_norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_selected_cells_1']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def ALLEN_selected_cells_103(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_selected_cells_103'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'adj_norm_conv2d'),
                os.path.join(model_folder, 'scalar_norm_conv2d'),
                os.path.join(model_folder, 'vector_norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_selected_cells_103']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def ALLEN_random_cells_103(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_random_cells_103'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'adj_norm_conv2d'),
                os.path.join(model_folder, 'scalar_norm_conv2d'),
                os.path.join(model_folder, 'vector_norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_random_cells_103']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def ALLEN_all_cells(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_all_cells_103'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'adj_norm_conv2d'),
                os.path.join(model_folder, 'scalar_norm_conv2d'),
                os.path.join(model_folder, 'vector_norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_all_cells']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_rRFchirKwMOmMInq(self):
        """MULTIALLEN_rRFchirKwMOmMInq multi-experiment creation."""
        model_folder = 'MULTIALLEN_rRFchirKwMOmMInq'
        exp = {
            'experiment_name': ['MULTIALLEN_rRFchirKwMOmMInq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rRFchirKwMOmMInq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PCBpaaTtzeMgCpMT(self):
        """MULTIALLEN_PCBpaaTtzeMgCpMT multi-experiment creation."""
        model_folder = 'MULTIALLEN_PCBpaaTtzeMgCpMT'
        exp = {
            'experiment_name': ['MULTIALLEN_PCBpaaTtzeMgCpMT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PCBpaaTtzeMgCpMT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ALClEIVFcnMyqOqt(self):
        """MULTIALLEN_ALClEIVFcnMyqOqt multi-experiment creation."""
        model_folder = 'MULTIALLEN_ALClEIVFcnMyqOqt'
        exp = {
            'experiment_name': ['MULTIALLEN_ALClEIVFcnMyqOqt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ALClEIVFcnMyqOqt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JPmWGlPCZBhCrqdJ(self):
        """MULTIALLEN_JPmWGlPCZBhCrqdJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_JPmWGlPCZBhCrqdJ'
        exp = {
            'experiment_name': ['MULTIALLEN_JPmWGlPCZBhCrqdJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JPmWGlPCZBhCrqdJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MLoWfNxqYQofyloC(self):
        """MULTIALLEN_MLoWfNxqYQofyloC multi-experiment creation."""
        model_folder = 'MULTIALLEN_MLoWfNxqYQofyloC'
        exp = {
            'experiment_name': ['MULTIALLEN_MLoWfNxqYQofyloC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MLoWfNxqYQofyloC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZlUmWRajCBaTYrfg(self):
        """MULTIALLEN_ZlUmWRajCBaTYrfg multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZlUmWRajCBaTYrfg'
        exp = {
            'experiment_name': ['MULTIALLEN_ZlUmWRajCBaTYrfg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZlUmWRajCBaTYrfg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zSDRsILwTFKbmqQr(self):
        """MULTIALLEN_zSDRsILwTFKbmqQr multi-experiment creation."""
        model_folder = 'MULTIALLEN_zSDRsILwTFKbmqQr'
        exp = {
            'experiment_name': ['MULTIALLEN_zSDRsILwTFKbmqQr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zSDRsILwTFKbmqQr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yaAghpRiBsjDPALI(self):
        """MULTIALLEN_yaAghpRiBsjDPALI multi-experiment creation."""
        model_folder = 'MULTIALLEN_yaAghpRiBsjDPALI'
        exp = {
            'experiment_name': ['MULTIALLEN_yaAghpRiBsjDPALI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yaAghpRiBsjDPALI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tzQrxrFEiZzJLIZQ(self):
        """MULTIALLEN_tzQrxrFEiZzJLIZQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_tzQrxrFEiZzJLIZQ'
        exp = {
            'experiment_name': ['MULTIALLEN_tzQrxrFEiZzJLIZQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tzQrxrFEiZzJLIZQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xHCRdAtJuEmkUuPk(self):
        """MULTIALLEN_xHCRdAtJuEmkUuPk multi-experiment creation."""
        model_folder = 'MULTIALLEN_xHCRdAtJuEmkUuPk'
        exp = {
            'experiment_name': ['MULTIALLEN_xHCRdAtJuEmkUuPk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xHCRdAtJuEmkUuPk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vZPPisLFHWfeefvD(self):
        """MULTIALLEN_vZPPisLFHWfeefvD multi-experiment creation."""
        model_folder = 'MULTIALLEN_vZPPisLFHWfeefvD'
        exp = {
            'experiment_name': ['MULTIALLEN_vZPPisLFHWfeefvD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vZPPisLFHWfeefvD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KooYzvDNZYdzFIHE(self):
        """MULTIALLEN_KooYzvDNZYdzFIHE multi-experiment creation."""
        model_folder = 'MULTIALLEN_KooYzvDNZYdzFIHE'
        exp = {
            'experiment_name': ['MULTIALLEN_KooYzvDNZYdzFIHE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KooYzvDNZYdzFIHE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OnsOdWRjUZhtRjKH(self):
        """MULTIALLEN_OnsOdWRjUZhtRjKH multi-experiment creation."""
        model_folder = 'MULTIALLEN_OnsOdWRjUZhtRjKH'
        exp = {
            'experiment_name': ['MULTIALLEN_OnsOdWRjUZhtRjKH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OnsOdWRjUZhtRjKH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hYFuuQeVTAmrVHyL(self):
        """MULTIALLEN_hYFuuQeVTAmrVHyL multi-experiment creation."""
        model_folder = 'MULTIALLEN_hYFuuQeVTAmrVHyL'
        exp = {
            'experiment_name': ['MULTIALLEN_hYFuuQeVTAmrVHyL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hYFuuQeVTAmrVHyL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EBzpTBAEfZEdBpga(self):
        """MULTIALLEN_EBzpTBAEfZEdBpga multi-experiment creation."""
        model_folder = 'MULTIALLEN_EBzpTBAEfZEdBpga'
        exp = {
            'experiment_name': ['MULTIALLEN_EBzpTBAEfZEdBpga'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EBzpTBAEfZEdBpga']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dUDIAhSQaStzPuoN(self):
        """MULTIALLEN_dUDIAhSQaStzPuoN multi-experiment creation."""
        model_folder = 'MULTIALLEN_dUDIAhSQaStzPuoN'
        exp = {
            'experiment_name': ['MULTIALLEN_dUDIAhSQaStzPuoN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dUDIAhSQaStzPuoN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OZPvTVlsZGFgzdOw(self):
        """MULTIALLEN_OZPvTVlsZGFgzdOw multi-experiment creation."""
        model_folder = 'MULTIALLEN_OZPvTVlsZGFgzdOw'
        exp = {
            'experiment_name': ['MULTIALLEN_OZPvTVlsZGFgzdOw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OZPvTVlsZGFgzdOw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iOIASPHWlajSRHik(self):
        """MULTIALLEN_iOIASPHWlajSRHik multi-experiment creation."""
        model_folder = 'MULTIALLEN_iOIASPHWlajSRHik'
        exp = {
            'experiment_name': ['MULTIALLEN_iOIASPHWlajSRHik'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iOIASPHWlajSRHik']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nsiaiuIIddCWwtWt(self):
        """MULTIALLEN_nsiaiuIIddCWwtWt multi-experiment creation."""
        model_folder = 'MULTIALLEN_nsiaiuIIddCWwtWt'
        exp = {
            'experiment_name': ['MULTIALLEN_nsiaiuIIddCWwtWt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nsiaiuIIddCWwtWt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OBXQZbJOrWTtyjMh(self):
        """MULTIALLEN_OBXQZbJOrWTtyjMh multi-experiment creation."""
        model_folder = 'MULTIALLEN_OBXQZbJOrWTtyjMh'
        exp = {
            'experiment_name': ['MULTIALLEN_OBXQZbJOrWTtyjMh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OBXQZbJOrWTtyjMh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rNgkcBipUyFcSpBL(self):
        """MULTIALLEN_rNgkcBipUyFcSpBL multi-experiment creation."""
        model_folder = 'MULTIALLEN_rNgkcBipUyFcSpBL'
        exp = {
            'experiment_name': ['MULTIALLEN_rNgkcBipUyFcSpBL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rNgkcBipUyFcSpBL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DCGmWrpGJSTyBEKs(self):
        """MULTIALLEN_DCGmWrpGJSTyBEKs multi-experiment creation."""
        model_folder = 'MULTIALLEN_DCGmWrpGJSTyBEKs'
        exp = {
            'experiment_name': ['MULTIALLEN_DCGmWrpGJSTyBEKs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DCGmWrpGJSTyBEKs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tjFAxHUzzjmgQCph(self):
        """MULTIALLEN_tjFAxHUzzjmgQCph multi-experiment creation."""
        model_folder = 'MULTIALLEN_tjFAxHUzzjmgQCph'
        exp = {
            'experiment_name': ['MULTIALLEN_tjFAxHUzzjmgQCph'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tjFAxHUzzjmgQCph']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sZyFOJzxkDMDvoXa(self):
        """MULTIALLEN_sZyFOJzxkDMDvoXa multi-experiment creation."""
        model_folder = 'MULTIALLEN_sZyFOJzxkDMDvoXa'
        exp = {
            'experiment_name': ['MULTIALLEN_sZyFOJzxkDMDvoXa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sZyFOJzxkDMDvoXa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_podBULUcWbAVtArz(self):
        """MULTIALLEN_podBULUcWbAVtArz multi-experiment creation."""
        model_folder = 'MULTIALLEN_podBULUcWbAVtArz'
        exp = {
            'experiment_name': ['MULTIALLEN_podBULUcWbAVtArz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_podBULUcWbAVtArz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KbSriMayJEtuEIIf(self):
        """MULTIALLEN_KbSriMayJEtuEIIf multi-experiment creation."""
        model_folder = 'MULTIALLEN_KbSriMayJEtuEIIf'
        exp = {
            'experiment_name': ['MULTIALLEN_KbSriMayJEtuEIIf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KbSriMayJEtuEIIf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SnMOSZitLemhUxxt(self):
        """MULTIALLEN_SnMOSZitLemhUxxt multi-experiment creation."""
        model_folder = 'MULTIALLEN_SnMOSZitLemhUxxt'
        exp = {
            'experiment_name': ['MULTIALLEN_SnMOSZitLemhUxxt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SnMOSZitLemhUxxt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VCmMTXlSwmlYYFhM(self):
        """MULTIALLEN_VCmMTXlSwmlYYFhM multi-experiment creation."""
        model_folder = 'MULTIALLEN_VCmMTXlSwmlYYFhM'
        exp = {
            'experiment_name': ['MULTIALLEN_VCmMTXlSwmlYYFhM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VCmMTXlSwmlYYFhM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WDauIRPYqwMlQgDi(self):
        """MULTIALLEN_WDauIRPYqwMlQgDi multi-experiment creation."""
        model_folder = 'MULTIALLEN_WDauIRPYqwMlQgDi'
        exp = {
            'experiment_name': ['MULTIALLEN_WDauIRPYqwMlQgDi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WDauIRPYqwMlQgDi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aKliRutAooAXhpxW(self):
        """MULTIALLEN_aKliRutAooAXhpxW multi-experiment creation."""
        model_folder = 'MULTIALLEN_aKliRutAooAXhpxW'
        exp = {
            'experiment_name': ['MULTIALLEN_aKliRutAooAXhpxW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aKliRutAooAXhpxW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vtXcQdetyJdKLEyh(self):
        """MULTIALLEN_vtXcQdetyJdKLEyh multi-experiment creation."""
        model_folder = 'MULTIALLEN_vtXcQdetyJdKLEyh'
        exp = {
            'experiment_name': ['MULTIALLEN_vtXcQdetyJdKLEyh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vtXcQdetyJdKLEyh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kEuXDXlbvBgFrmkx(self):
        """MULTIALLEN_kEuXDXlbvBgFrmkx multi-experiment creation."""
        model_folder = 'MULTIALLEN_kEuXDXlbvBgFrmkx'
        exp = {
            'experiment_name': ['MULTIALLEN_kEuXDXlbvBgFrmkx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kEuXDXlbvBgFrmkx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DEnWOcQMsvIyjVht(self):
        """MULTIALLEN_DEnWOcQMsvIyjVht multi-experiment creation."""
        model_folder = 'MULTIALLEN_DEnWOcQMsvIyjVht'
        exp = {
            'experiment_name': ['MULTIALLEN_DEnWOcQMsvIyjVht'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DEnWOcQMsvIyjVht']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_blnWQTqmBtXbngzx(self):
        """MULTIALLEN_blnWQTqmBtXbngzx multi-experiment creation."""
        model_folder = 'MULTIALLEN_blnWQTqmBtXbngzx'
        exp = {
            'experiment_name': ['MULTIALLEN_blnWQTqmBtXbngzx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_blnWQTqmBtXbngzx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zZDEOKvgZeaKoRQm(self):
        """MULTIALLEN_zZDEOKvgZeaKoRQm multi-experiment creation."""
        model_folder = 'MULTIALLEN_zZDEOKvgZeaKoRQm'
        exp = {
            'experiment_name': ['MULTIALLEN_zZDEOKvgZeaKoRQm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zZDEOKvgZeaKoRQm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yJUzqyJuTNiLqnyY(self):
        """MULTIALLEN_yJUzqyJuTNiLqnyY multi-experiment creation."""
        model_folder = 'MULTIALLEN_yJUzqyJuTNiLqnyY'
        exp = {
            'experiment_name': ['MULTIALLEN_yJUzqyJuTNiLqnyY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yJUzqyJuTNiLqnyY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gBPVivuVlNtGZdsx(self):
        """MULTIALLEN_gBPVivuVlNtGZdsx multi-experiment creation."""
        model_folder = 'MULTIALLEN_gBPVivuVlNtGZdsx'
        exp = {
            'experiment_name': ['MULTIALLEN_gBPVivuVlNtGZdsx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gBPVivuVlNtGZdsx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PlVdmcrWkOhSbhwF(self):
        """MULTIALLEN_PlVdmcrWkOhSbhwF multi-experiment creation."""
        model_folder = 'MULTIALLEN_PlVdmcrWkOhSbhwF'
        exp = {
            'experiment_name': ['MULTIALLEN_PlVdmcrWkOhSbhwF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PlVdmcrWkOhSbhwF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IPXTvWRRwpeNlaQV(self):
        """MULTIALLEN_IPXTvWRRwpeNlaQV multi-experiment creation."""
        model_folder = 'MULTIALLEN_IPXTvWRRwpeNlaQV'
        exp = {
            'experiment_name': ['MULTIALLEN_IPXTvWRRwpeNlaQV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IPXTvWRRwpeNlaQV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZavunreXfYDCocrF(self):
        """MULTIALLEN_ZavunreXfYDCocrF multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZavunreXfYDCocrF'
        exp = {
            'experiment_name': ['MULTIALLEN_ZavunreXfYDCocrF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZavunreXfYDCocrF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pJnKIHdgYwOrtCBN(self):
        """MULTIALLEN_pJnKIHdgYwOrtCBN multi-experiment creation."""
        model_folder = 'MULTIALLEN_pJnKIHdgYwOrtCBN'
        exp = {
            'experiment_name': ['MULTIALLEN_pJnKIHdgYwOrtCBN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pJnKIHdgYwOrtCBN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XwuPEPEgkccKLNPO(self):
        """MULTIALLEN_XwuPEPEgkccKLNPO multi-experiment creation."""
        model_folder = 'MULTIALLEN_XwuPEPEgkccKLNPO'
        exp = {
            'experiment_name': ['MULTIALLEN_XwuPEPEgkccKLNPO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XwuPEPEgkccKLNPO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fAljtkUfoEnRKeak(self):
        """MULTIALLEN_fAljtkUfoEnRKeak multi-experiment creation."""
        model_folder = 'MULTIALLEN_fAljtkUfoEnRKeak'
        exp = {
            'experiment_name': ['MULTIALLEN_fAljtkUfoEnRKeak'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fAljtkUfoEnRKeak']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QrFMFGZKXRokRLdY(self):
        """MULTIALLEN_QrFMFGZKXRokRLdY multi-experiment creation."""
        model_folder = 'MULTIALLEN_QrFMFGZKXRokRLdY'
        exp = {
            'experiment_name': ['MULTIALLEN_QrFMFGZKXRokRLdY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QrFMFGZKXRokRLdY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KqcqbUEitRjNJTbT(self):
        """MULTIALLEN_KqcqbUEitRjNJTbT multi-experiment creation."""
        model_folder = 'MULTIALLEN_KqcqbUEitRjNJTbT'
        exp = {
            'experiment_name': ['MULTIALLEN_KqcqbUEitRjNJTbT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KqcqbUEitRjNJTbT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FaoAvZEPpdYqydVG(self):
        """MULTIALLEN_FaoAvZEPpdYqydVG multi-experiment creation."""
        model_folder = 'MULTIALLEN_FaoAvZEPpdYqydVG'
        exp = {
            'experiment_name': ['MULTIALLEN_FaoAvZEPpdYqydVG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FaoAvZEPpdYqydVG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ytnpqYTFBgSpDYNr(self):
        """MULTIALLEN_ytnpqYTFBgSpDYNr multi-experiment creation."""
        model_folder = 'MULTIALLEN_ytnpqYTFBgSpDYNr'
        exp = {
            'experiment_name': ['MULTIALLEN_ytnpqYTFBgSpDYNr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ytnpqYTFBgSpDYNr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zGvDagIPrcpSfTSZ(self):
        """MULTIALLEN_zGvDagIPrcpSfTSZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_zGvDagIPrcpSfTSZ'
        exp = {
            'experiment_name': ['MULTIALLEN_zGvDagIPrcpSfTSZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zGvDagIPrcpSfTSZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OSCBozjoxmgZNlUQ(self):
        """MULTIALLEN_OSCBozjoxmgZNlUQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_OSCBozjoxmgZNlUQ'
        exp = {
            'experiment_name': ['MULTIALLEN_OSCBozjoxmgZNlUQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OSCBozjoxmgZNlUQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TQOcYzJVOPVIJQAV(self):
        """MULTIALLEN_TQOcYzJVOPVIJQAV multi-experiment creation."""
        model_folder = 'MULTIALLEN_TQOcYzJVOPVIJQAV'
        exp = {
            'experiment_name': ['MULTIALLEN_TQOcYzJVOPVIJQAV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TQOcYzJVOPVIJQAV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UTCHKgquEaERhaJr(self):
        """MULTIALLEN_UTCHKgquEaERhaJr multi-experiment creation."""
        model_folder = 'MULTIALLEN_UTCHKgquEaERhaJr'
        exp = {
            'experiment_name': ['MULTIALLEN_UTCHKgquEaERhaJr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UTCHKgquEaERhaJr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GCAJEUGdgZhrpRXy(self):
        """MULTIALLEN_GCAJEUGdgZhrpRXy multi-experiment creation."""
        model_folder = 'MULTIALLEN_GCAJEUGdgZhrpRXy'
        exp = {
            'experiment_name': ['MULTIALLEN_GCAJEUGdgZhrpRXy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GCAJEUGdgZhrpRXy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KAHMOrFsDhhbvWrg(self):
        """MULTIALLEN_KAHMOrFsDhhbvWrg multi-experiment creation."""
        model_folder = 'MULTIALLEN_KAHMOrFsDhhbvWrg'
        exp = {
            'experiment_name': ['MULTIALLEN_KAHMOrFsDhhbvWrg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KAHMOrFsDhhbvWrg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YRyFOiEErOLhRwXA(self):
        """MULTIALLEN_YRyFOiEErOLhRwXA multi-experiment creation."""
        model_folder = 'MULTIALLEN_YRyFOiEErOLhRwXA'
        exp = {
            'experiment_name': ['MULTIALLEN_YRyFOiEErOLhRwXA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YRyFOiEErOLhRwXA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mSBLOgzXEulDctBK(self):
        """MULTIALLEN_mSBLOgzXEulDctBK multi-experiment creation."""
        model_folder = 'MULTIALLEN_mSBLOgzXEulDctBK'
        exp = {
            'experiment_name': ['MULTIALLEN_mSBLOgzXEulDctBK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mSBLOgzXEulDctBK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PKAcKEcWUrAZlFCA(self):
        """MULTIALLEN_PKAcKEcWUrAZlFCA multi-experiment creation."""
        model_folder = 'MULTIALLEN_PKAcKEcWUrAZlFCA'
        exp = {
            'experiment_name': ['MULTIALLEN_PKAcKEcWUrAZlFCA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PKAcKEcWUrAZlFCA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OdQQNRWJtdTMzIcP(self):
        """MULTIALLEN_OdQQNRWJtdTMzIcP multi-experiment creation."""
        model_folder = 'MULTIALLEN_OdQQNRWJtdTMzIcP'
        exp = {
            'experiment_name': ['MULTIALLEN_OdQQNRWJtdTMzIcP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OdQQNRWJtdTMzIcP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gTaKdjvQfXgpnSDv(self):
        """MULTIALLEN_gTaKdjvQfXgpnSDv multi-experiment creation."""
        model_folder = 'MULTIALLEN_gTaKdjvQfXgpnSDv'
        exp = {
            'experiment_name': ['MULTIALLEN_gTaKdjvQfXgpnSDv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gTaKdjvQfXgpnSDv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XqTYOLEMKeBcJCyT(self):
        """MULTIALLEN_XqTYOLEMKeBcJCyT multi-experiment creation."""
        model_folder = 'MULTIALLEN_XqTYOLEMKeBcJCyT'
        exp = {
            'experiment_name': ['MULTIALLEN_XqTYOLEMKeBcJCyT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XqTYOLEMKeBcJCyT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aqCbXWYLHyFmOSFm(self):
        """MULTIALLEN_aqCbXWYLHyFmOSFm multi-experiment creation."""
        model_folder = 'MULTIALLEN_aqCbXWYLHyFmOSFm'
        exp = {
            'experiment_name': ['MULTIALLEN_aqCbXWYLHyFmOSFm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aqCbXWYLHyFmOSFm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sfqYjlZKPkgMcfuV(self):
        """MULTIALLEN_sfqYjlZKPkgMcfuV multi-experiment creation."""
        model_folder = 'MULTIALLEN_sfqYjlZKPkgMcfuV'
        exp = {
            'experiment_name': ['MULTIALLEN_sfqYjlZKPkgMcfuV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sfqYjlZKPkgMcfuV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zxVrfqsDKiEINGcD(self):
        """MULTIALLEN_zxVrfqsDKiEINGcD multi-experiment creation."""
        model_folder = 'MULTIALLEN_zxVrfqsDKiEINGcD'
        exp = {
            'experiment_name': ['MULTIALLEN_zxVrfqsDKiEINGcD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zxVrfqsDKiEINGcD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zBWGomaqZSmdjQxd(self):
        """MULTIALLEN_zBWGomaqZSmdjQxd multi-experiment creation."""
        model_folder = 'MULTIALLEN_zBWGomaqZSmdjQxd'
        exp = {
            'experiment_name': ['MULTIALLEN_zBWGomaqZSmdjQxd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zBWGomaqZSmdjQxd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VJnEJNQwahpLYCfY(self):
        """MULTIALLEN_VJnEJNQwahpLYCfY multi-experiment creation."""
        model_folder = 'MULTIALLEN_VJnEJNQwahpLYCfY'
        exp = {
            'experiment_name': ['MULTIALLEN_VJnEJNQwahpLYCfY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VJnEJNQwahpLYCfY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YdyVjNVKFvDXGGpO(self):
        """MULTIALLEN_YdyVjNVKFvDXGGpO multi-experiment creation."""
        model_folder = 'MULTIALLEN_YdyVjNVKFvDXGGpO'
        exp = {
            'experiment_name': ['MULTIALLEN_YdyVjNVKFvDXGGpO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YdyVjNVKFvDXGGpO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QFjvbNEHQFwZdAPY(self):
        """MULTIALLEN_QFjvbNEHQFwZdAPY multi-experiment creation."""
        model_folder = 'MULTIALLEN_QFjvbNEHQFwZdAPY'
        exp = {
            'experiment_name': ['MULTIALLEN_QFjvbNEHQFwZdAPY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QFjvbNEHQFwZdAPY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vDUhwhYKFiZPBQbB(self):
        """MULTIALLEN_vDUhwhYKFiZPBQbB multi-experiment creation."""
        model_folder = 'MULTIALLEN_vDUhwhYKFiZPBQbB'
        exp = {
            'experiment_name': ['MULTIALLEN_vDUhwhYKFiZPBQbB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vDUhwhYKFiZPBQbB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GybmbmvNAQIfbnWn(self):
        """MULTIALLEN_GybmbmvNAQIfbnWn multi-experiment creation."""
        model_folder = 'MULTIALLEN_GybmbmvNAQIfbnWn'
        exp = {
            'experiment_name': ['MULTIALLEN_GybmbmvNAQIfbnWn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GybmbmvNAQIfbnWn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mZspTYAInqWrTQNx(self):
        """MULTIALLEN_mZspTYAInqWrTQNx multi-experiment creation."""
        model_folder = 'MULTIALLEN_mZspTYAInqWrTQNx'
        exp = {
            'experiment_name': ['MULTIALLEN_mZspTYAInqWrTQNx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mZspTYAInqWrTQNx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HqMglARgzlMrAgmK(self):
        """MULTIALLEN_HqMglARgzlMrAgmK multi-experiment creation."""
        model_folder = 'MULTIALLEN_HqMglARgzlMrAgmK'
        exp = {
            'experiment_name': ['MULTIALLEN_HqMglARgzlMrAgmK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HqMglARgzlMrAgmK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xmXyDWYRzKXdtWVV(self):
        """MULTIALLEN_xmXyDWYRzKXdtWVV multi-experiment creation."""
        model_folder = 'MULTIALLEN_xmXyDWYRzKXdtWVV'
        exp = {
            'experiment_name': ['MULTIALLEN_xmXyDWYRzKXdtWVV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xmXyDWYRzKXdtWVV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QBlHQwGMLIeJmytm(self):
        """MULTIALLEN_QBlHQwGMLIeJmytm multi-experiment creation."""
        model_folder = 'MULTIALLEN_QBlHQwGMLIeJmytm'
        exp = {
            'experiment_name': ['MULTIALLEN_QBlHQwGMLIeJmytm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QBlHQwGMLIeJmytm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dGGjHBrPoLEvFjzs(self):
        """MULTIALLEN_dGGjHBrPoLEvFjzs multi-experiment creation."""
        model_folder = 'MULTIALLEN_dGGjHBrPoLEvFjzs'
        exp = {
            'experiment_name': ['MULTIALLEN_dGGjHBrPoLEvFjzs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dGGjHBrPoLEvFjzs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iTTxNzkApBKCewuC(self):
        """MULTIALLEN_iTTxNzkApBKCewuC multi-experiment creation."""
        model_folder = 'MULTIALLEN_iTTxNzkApBKCewuC'
        exp = {
            'experiment_name': ['MULTIALLEN_iTTxNzkApBKCewuC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iTTxNzkApBKCewuC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AcTAUJcgRHqFCgOm(self):
        """MULTIALLEN_AcTAUJcgRHqFCgOm multi-experiment creation."""
        model_folder = 'MULTIALLEN_AcTAUJcgRHqFCgOm'
        exp = {
            'experiment_name': ['MULTIALLEN_AcTAUJcgRHqFCgOm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AcTAUJcgRHqFCgOm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xMLuALBgdFdtvTAn(self):
        """MULTIALLEN_xMLuALBgdFdtvTAn multi-experiment creation."""
        model_folder = 'MULTIALLEN_xMLuALBgdFdtvTAn'
        exp = {
            'experiment_name': ['MULTIALLEN_xMLuALBgdFdtvTAn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xMLuALBgdFdtvTAn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KCTloMIxIdIekpvL(self):
        """MULTIALLEN_KCTloMIxIdIekpvL multi-experiment creation."""
        model_folder = 'MULTIALLEN_KCTloMIxIdIekpvL'
        exp = {
            'experiment_name': ['MULTIALLEN_KCTloMIxIdIekpvL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KCTloMIxIdIekpvL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PWemoFcDRacyXxfy(self):
        """MULTIALLEN_PWemoFcDRacyXxfy multi-experiment creation."""
        model_folder = 'MULTIALLEN_PWemoFcDRacyXxfy'
        exp = {
            'experiment_name': ['MULTIALLEN_PWemoFcDRacyXxfy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PWemoFcDRacyXxfy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xnBxDajAmLBYQUHy(self):
        """MULTIALLEN_xnBxDajAmLBYQUHy multi-experiment creation."""
        model_folder = 'MULTIALLEN_xnBxDajAmLBYQUHy'
        exp = {
            'experiment_name': ['MULTIALLEN_xnBxDajAmLBYQUHy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xnBxDajAmLBYQUHy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PtnOZERbuGZgkQwe(self):
        """MULTIALLEN_PtnOZERbuGZgkQwe multi-experiment creation."""
        model_folder = 'MULTIALLEN_PtnOZERbuGZgkQwe'
        exp = {
            'experiment_name': ['MULTIALLEN_PtnOZERbuGZgkQwe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PtnOZERbuGZgkQwe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jQeADtqWduETmHnD(self):
        """MULTIALLEN_jQeADtqWduETmHnD multi-experiment creation."""
        model_folder = 'MULTIALLEN_jQeADtqWduETmHnD'
        exp = {
            'experiment_name': ['MULTIALLEN_jQeADtqWduETmHnD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jQeADtqWduETmHnD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PYzMJvgwKXHBmxkd(self):
        """MULTIALLEN_PYzMJvgwKXHBmxkd multi-experiment creation."""
        model_folder = 'MULTIALLEN_PYzMJvgwKXHBmxkd'
        exp = {
            'experiment_name': ['MULTIALLEN_PYzMJvgwKXHBmxkd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PYzMJvgwKXHBmxkd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UqUEGdcuOSpDolMp(self):
        """MULTIALLEN_UqUEGdcuOSpDolMp multi-experiment creation."""
        model_folder = 'MULTIALLEN_UqUEGdcuOSpDolMp'
        exp = {
            'experiment_name': ['MULTIALLEN_UqUEGdcuOSpDolMp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UqUEGdcuOSpDolMp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VKzjCYLxkAmTiNPi(self):
        """MULTIALLEN_VKzjCYLxkAmTiNPi multi-experiment creation."""
        model_folder = 'MULTIALLEN_VKzjCYLxkAmTiNPi'
        exp = {
            'experiment_name': ['MULTIALLEN_VKzjCYLxkAmTiNPi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VKzjCYLxkAmTiNPi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cDSCFLyBJesYTNTk(self):
        """MULTIALLEN_cDSCFLyBJesYTNTk multi-experiment creation."""
        model_folder = 'MULTIALLEN_cDSCFLyBJesYTNTk'
        exp = {
            'experiment_name': ['MULTIALLEN_cDSCFLyBJesYTNTk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cDSCFLyBJesYTNTk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nyuyRiUDsvBOOPOd(self):
        """MULTIALLEN_nyuyRiUDsvBOOPOd multi-experiment creation."""
        model_folder = 'MULTIALLEN_nyuyRiUDsvBOOPOd'
        exp = {
            'experiment_name': ['MULTIALLEN_nyuyRiUDsvBOOPOd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nyuyRiUDsvBOOPOd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XVUGRwYNhGSNgIvM(self):
        """MULTIALLEN_XVUGRwYNhGSNgIvM multi-experiment creation."""
        model_folder = 'MULTIALLEN_XVUGRwYNhGSNgIvM'
        exp = {
            'experiment_name': ['MULTIALLEN_XVUGRwYNhGSNgIvM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XVUGRwYNhGSNgIvM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GQIfnaxOidvPYICv(self):
        """MULTIALLEN_GQIfnaxOidvPYICv multi-experiment creation."""
        model_folder = 'MULTIALLEN_GQIfnaxOidvPYICv'
        exp = {
            'experiment_name': ['MULTIALLEN_GQIfnaxOidvPYICv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GQIfnaxOidvPYICv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VfwXWyMGZWAWXMCy(self):
        """MULTIALLEN_VfwXWyMGZWAWXMCy multi-experiment creation."""
        model_folder = 'MULTIALLEN_VfwXWyMGZWAWXMCy'
        exp = {
            'experiment_name': ['MULTIALLEN_VfwXWyMGZWAWXMCy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VfwXWyMGZWAWXMCy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EgUklWiOazfUVYoR(self):
        """MULTIALLEN_EgUklWiOazfUVYoR multi-experiment creation."""
        model_folder = 'MULTIALLEN_EgUklWiOazfUVYoR'
        exp = {
            'experiment_name': ['MULTIALLEN_EgUklWiOazfUVYoR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EgUklWiOazfUVYoR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ggtbobGvJCkJCzlx(self):
        """MULTIALLEN_ggtbobGvJCkJCzlx multi-experiment creation."""
        model_folder = 'MULTIALLEN_ggtbobGvJCkJCzlx'
        exp = {
            'experiment_name': ['MULTIALLEN_ggtbobGvJCkJCzlx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ggtbobGvJCkJCzlx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ExmOZafRlqTsMseN(self):
        """MULTIALLEN_ExmOZafRlqTsMseN multi-experiment creation."""
        model_folder = 'MULTIALLEN_ExmOZafRlqTsMseN'
        exp = {
            'experiment_name': ['MULTIALLEN_ExmOZafRlqTsMseN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ExmOZafRlqTsMseN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_coYGcEUgCDGijCOE(self):
        """MULTIALLEN_coYGcEUgCDGijCOE multi-experiment creation."""
        model_folder = 'MULTIALLEN_coYGcEUgCDGijCOE'
        exp = {
            'experiment_name': ['MULTIALLEN_coYGcEUgCDGijCOE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_coYGcEUgCDGijCOE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dbKNVgfikXvkUvXo(self):
        """MULTIALLEN_dbKNVgfikXvkUvXo multi-experiment creation."""
        model_folder = 'MULTIALLEN_dbKNVgfikXvkUvXo'
        exp = {
            'experiment_name': ['MULTIALLEN_dbKNVgfikXvkUvXo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dbKNVgfikXvkUvXo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YEWYKlerfENAfFOg(self):
        """MULTIALLEN_YEWYKlerfENAfFOg multi-experiment creation."""
        model_folder = 'MULTIALLEN_YEWYKlerfENAfFOg'
        exp = {
            'experiment_name': ['MULTIALLEN_YEWYKlerfENAfFOg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YEWYKlerfENAfFOg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IQpuOEBZkTDEMdnu(self):
        """MULTIALLEN_IQpuOEBZkTDEMdnu multi-experiment creation."""
        model_folder = 'MULTIALLEN_IQpuOEBZkTDEMdnu'
        exp = {
            'experiment_name': ['MULTIALLEN_IQpuOEBZkTDEMdnu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IQpuOEBZkTDEMdnu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aQaVOlWBIzXOhjpg(self):
        """MULTIALLEN_aQaVOlWBIzXOhjpg multi-experiment creation."""
        model_folder = 'MULTIALLEN_aQaVOlWBIzXOhjpg'
        exp = {
            'experiment_name': ['MULTIALLEN_aQaVOlWBIzXOhjpg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aQaVOlWBIzXOhjpg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UQNxhOVhiuBNyGwj(self):
        """MULTIALLEN_UQNxhOVhiuBNyGwj multi-experiment creation."""
        model_folder = 'MULTIALLEN_UQNxhOVhiuBNyGwj'
        exp = {
            'experiment_name': ['MULTIALLEN_UQNxhOVhiuBNyGwj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UQNxhOVhiuBNyGwj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AlEMSvQdNFBfhxED(self):
        """MULTIALLEN_AlEMSvQdNFBfhxED multi-experiment creation."""
        model_folder = 'MULTIALLEN_AlEMSvQdNFBfhxED'
        exp = {
            'experiment_name': ['MULTIALLEN_AlEMSvQdNFBfhxED'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AlEMSvQdNFBfhxED']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nnfbXnOedwOYqBqo(self):
        """MULTIALLEN_nnfbXnOedwOYqBqo multi-experiment creation."""
        model_folder = 'MULTIALLEN_nnfbXnOedwOYqBqo'
        exp = {
            'experiment_name': ['MULTIALLEN_nnfbXnOedwOYqBqo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nnfbXnOedwOYqBqo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aoLkNtaeoeylzRuM(self):
        """MULTIALLEN_aoLkNtaeoeylzRuM multi-experiment creation."""
        model_folder = 'MULTIALLEN_aoLkNtaeoeylzRuM'
        exp = {
            'experiment_name': ['MULTIALLEN_aoLkNtaeoeylzRuM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aoLkNtaeoeylzRuM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qsJioiaRYQItenlZ(self):
        """MULTIALLEN_qsJioiaRYQItenlZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_qsJioiaRYQItenlZ'
        exp = {
            'experiment_name': ['MULTIALLEN_qsJioiaRYQItenlZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qsJioiaRYQItenlZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zeFsrbrPqqFRsPjk(self):
        """MULTIALLEN_zeFsrbrPqqFRsPjk multi-experiment creation."""
        model_folder = 'MULTIALLEN_zeFsrbrPqqFRsPjk'
        exp = {
            'experiment_name': ['MULTIALLEN_zeFsrbrPqqFRsPjk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zeFsrbrPqqFRsPjk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_luCNwOWZhMPGoITw(self):
        """MULTIALLEN_luCNwOWZhMPGoITw multi-experiment creation."""
        model_folder = 'MULTIALLEN_luCNwOWZhMPGoITw'
        exp = {
            'experiment_name': ['MULTIALLEN_luCNwOWZhMPGoITw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_luCNwOWZhMPGoITw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_orpXYdaizWpxXQin(self):
        """MULTIALLEN_orpXYdaizWpxXQin multi-experiment creation."""
        model_folder = 'MULTIALLEN_orpXYdaizWpxXQin'
        exp = {
            'experiment_name': ['MULTIALLEN_orpXYdaizWpxXQin'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_orpXYdaizWpxXQin']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FejHKWASFPlCUiYp(self):
        """MULTIALLEN_FejHKWASFPlCUiYp multi-experiment creation."""
        model_folder = 'MULTIALLEN_FejHKWASFPlCUiYp'
        exp = {
            'experiment_name': ['MULTIALLEN_FejHKWASFPlCUiYp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FejHKWASFPlCUiYp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uNyZLAiyGBqxnoyy(self):
        """MULTIALLEN_uNyZLAiyGBqxnoyy multi-experiment creation."""
        model_folder = 'MULTIALLEN_uNyZLAiyGBqxnoyy'
        exp = {
            'experiment_name': ['MULTIALLEN_uNyZLAiyGBqxnoyy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uNyZLAiyGBqxnoyy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yCEkHHPBhUjVQLDO(self):
        """MULTIALLEN_yCEkHHPBhUjVQLDO multi-experiment creation."""
        model_folder = 'MULTIALLEN_yCEkHHPBhUjVQLDO'
        exp = {
            'experiment_name': ['MULTIALLEN_yCEkHHPBhUjVQLDO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yCEkHHPBhUjVQLDO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lczPjpCauAotYJRx(self):
        """MULTIALLEN_lczPjpCauAotYJRx multi-experiment creation."""
        model_folder = 'MULTIALLEN_lczPjpCauAotYJRx'
        exp = {
            'experiment_name': ['MULTIALLEN_lczPjpCauAotYJRx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lczPjpCauAotYJRx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dXLfJJyMuzEIwlDv(self):
        """MULTIALLEN_dXLfJJyMuzEIwlDv multi-experiment creation."""
        model_folder = 'MULTIALLEN_dXLfJJyMuzEIwlDv'
        exp = {
            'experiment_name': ['MULTIALLEN_dXLfJJyMuzEIwlDv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dXLfJJyMuzEIwlDv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_StmWDUUuINACuBPy(self):
        """MULTIALLEN_StmWDUUuINACuBPy multi-experiment creation."""
        model_folder = 'MULTIALLEN_StmWDUUuINACuBPy'
        exp = {
            'experiment_name': ['MULTIALLEN_StmWDUUuINACuBPy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_StmWDUUuINACuBPy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CNeBKfwdeJGDXPdo(self):
        """MULTIALLEN_CNeBKfwdeJGDXPdo multi-experiment creation."""
        model_folder = 'MULTIALLEN_CNeBKfwdeJGDXPdo'
        exp = {
            'experiment_name': ['MULTIALLEN_CNeBKfwdeJGDXPdo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CNeBKfwdeJGDXPdo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ltvFFcCVrHkHFRnp(self):
        """MULTIALLEN_ltvFFcCVrHkHFRnp multi-experiment creation."""
        model_folder = 'MULTIALLEN_ltvFFcCVrHkHFRnp'
        exp = {
            'experiment_name': ['MULTIALLEN_ltvFFcCVrHkHFRnp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ltvFFcCVrHkHFRnp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MoMhvKpbhLDyJDOg(self):
        """MULTIALLEN_MoMhvKpbhLDyJDOg multi-experiment creation."""
        model_folder = 'MULTIALLEN_MoMhvKpbhLDyJDOg'
        exp = {
            'experiment_name': ['MULTIALLEN_MoMhvKpbhLDyJDOg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MoMhvKpbhLDyJDOg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PuVNDmnUiWgIecTy(self):
        """MULTIALLEN_PuVNDmnUiWgIecTy multi-experiment creation."""
        model_folder = 'MULTIALLEN_PuVNDmnUiWgIecTy'
        exp = {
            'experiment_name': ['MULTIALLEN_PuVNDmnUiWgIecTy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PuVNDmnUiWgIecTy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rsJTsHIXjyvSpYjt(self):
        """MULTIALLEN_rsJTsHIXjyvSpYjt multi-experiment creation."""
        model_folder = 'MULTIALLEN_rsJTsHIXjyvSpYjt'
        exp = {
            'experiment_name': ['MULTIALLEN_rsJTsHIXjyvSpYjt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rsJTsHIXjyvSpYjt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kexDataYblSKarbW(self):
        """MULTIALLEN_kexDataYblSKarbW multi-experiment creation."""
        model_folder = 'MULTIALLEN_kexDataYblSKarbW'
        exp = {
            'experiment_name': ['MULTIALLEN_kexDataYblSKarbW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kexDataYblSKarbW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LGtjjgxlWOyerYID(self):
        """MULTIALLEN_LGtjjgxlWOyerYID multi-experiment creation."""
        model_folder = 'MULTIALLEN_LGtjjgxlWOyerYID'
        exp = {
            'experiment_name': ['MULTIALLEN_LGtjjgxlWOyerYID'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LGtjjgxlWOyerYID']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VJImBrRywZitQfze(self):
        """MULTIALLEN_VJImBrRywZitQfze multi-experiment creation."""
        model_folder = 'MULTIALLEN_VJImBrRywZitQfze'
        exp = {
            'experiment_name': ['MULTIALLEN_VJImBrRywZitQfze'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VJImBrRywZitQfze']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MKQgRcdEvFpdcaLg(self):
        """MULTIALLEN_MKQgRcdEvFpdcaLg multi-experiment creation."""
        model_folder = 'MULTIALLEN_MKQgRcdEvFpdcaLg'
        exp = {
            'experiment_name': ['MULTIALLEN_MKQgRcdEvFpdcaLg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MKQgRcdEvFpdcaLg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IFsjYhJgrEueyKoS(self):
        """MULTIALLEN_IFsjYhJgrEueyKoS multi-experiment creation."""
        model_folder = 'MULTIALLEN_IFsjYhJgrEueyKoS'
        exp = {
            'experiment_name': ['MULTIALLEN_IFsjYhJgrEueyKoS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IFsjYhJgrEueyKoS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mQodyCblHnquSOhh(self):
        """MULTIALLEN_mQodyCblHnquSOhh multi-experiment creation."""
        model_folder = 'MULTIALLEN_mQodyCblHnquSOhh'
        exp = {
            'experiment_name': ['MULTIALLEN_mQodyCblHnquSOhh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mQodyCblHnquSOhh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HOwCcecHOcphWjnY(self):
        """MULTIALLEN_HOwCcecHOcphWjnY multi-experiment creation."""
        model_folder = 'MULTIALLEN_HOwCcecHOcphWjnY'
        exp = {
            'experiment_name': ['MULTIALLEN_HOwCcecHOcphWjnY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HOwCcecHOcphWjnY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jgCgGiKPnwAVuWFL(self):
        """MULTIALLEN_jgCgGiKPnwAVuWFL multi-experiment creation."""
        model_folder = 'MULTIALLEN_jgCgGiKPnwAVuWFL'
        exp = {
            'experiment_name': ['MULTIALLEN_jgCgGiKPnwAVuWFL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jgCgGiKPnwAVuWFL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uMOhYJWIFXHPLDsu(self):
        """MULTIALLEN_uMOhYJWIFXHPLDsu multi-experiment creation."""
        model_folder = 'MULTIALLEN_uMOhYJWIFXHPLDsu'
        exp = {
            'experiment_name': ['MULTIALLEN_uMOhYJWIFXHPLDsu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uMOhYJWIFXHPLDsu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SDiRfgziheFQxwvg(self):
        """MULTIALLEN_SDiRfgziheFQxwvg multi-experiment creation."""
        model_folder = 'MULTIALLEN_SDiRfgziheFQxwvg'
        exp = {
            'experiment_name': ['MULTIALLEN_SDiRfgziheFQxwvg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SDiRfgziheFQxwvg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cITWsCByqaMNnKEm(self):
        """MULTIALLEN_cITWsCByqaMNnKEm multi-experiment creation."""
        model_folder = 'MULTIALLEN_cITWsCByqaMNnKEm'
        exp = {
            'experiment_name': ['MULTIALLEN_cITWsCByqaMNnKEm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cITWsCByqaMNnKEm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SvNXNlhMDvDFiSCr(self):
        """MULTIALLEN_SvNXNlhMDvDFiSCr multi-experiment creation."""
        model_folder = 'MULTIALLEN_SvNXNlhMDvDFiSCr'
        exp = {
            'experiment_name': ['MULTIALLEN_SvNXNlhMDvDFiSCr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SvNXNlhMDvDFiSCr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jvAaYRKBRrYuzdyw(self):
        """MULTIALLEN_jvAaYRKBRrYuzdyw multi-experiment creation."""
        model_folder = 'MULTIALLEN_jvAaYRKBRrYuzdyw'
        exp = {
            'experiment_name': ['MULTIALLEN_jvAaYRKBRrYuzdyw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jvAaYRKBRrYuzdyw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tFVJQXfaDNoYgUug(self):
        """MULTIALLEN_tFVJQXfaDNoYgUug multi-experiment creation."""
        model_folder = 'MULTIALLEN_tFVJQXfaDNoYgUug'
        exp = {
            'experiment_name': ['MULTIALLEN_tFVJQXfaDNoYgUug'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tFVJQXfaDNoYgUug']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YIkgExdjmucjCNBt(self):
        """MULTIALLEN_YIkgExdjmucjCNBt multi-experiment creation."""
        model_folder = 'MULTIALLEN_YIkgExdjmucjCNBt'
        exp = {
            'experiment_name': ['MULTIALLEN_YIkgExdjmucjCNBt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YIkgExdjmucjCNBt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aNiANOpdkEzZywMx(self):
        """MULTIALLEN_aNiANOpdkEzZywMx multi-experiment creation."""
        model_folder = 'MULTIALLEN_aNiANOpdkEzZywMx'
        exp = {
            'experiment_name': ['MULTIALLEN_aNiANOpdkEzZywMx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aNiANOpdkEzZywMx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GdQyaxvDFAUbDTXG(self):
        """MULTIALLEN_GdQyaxvDFAUbDTXG multi-experiment creation."""
        model_folder = 'MULTIALLEN_GdQyaxvDFAUbDTXG'
        exp = {
            'experiment_name': ['MULTIALLEN_GdQyaxvDFAUbDTXG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GdQyaxvDFAUbDTXG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VPQaiVFVWVttdolh(self):
        """MULTIALLEN_VPQaiVFVWVttdolh multi-experiment creation."""
        model_folder = 'MULTIALLEN_VPQaiVFVWVttdolh'
        exp = {
            'experiment_name': ['MULTIALLEN_VPQaiVFVWVttdolh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VPQaiVFVWVttdolh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iRQyYjYrFnhlGAdk(self):
        """MULTIALLEN_iRQyYjYrFnhlGAdk multi-experiment creation."""
        model_folder = 'MULTIALLEN_iRQyYjYrFnhlGAdk'
        exp = {
            'experiment_name': ['MULTIALLEN_iRQyYjYrFnhlGAdk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iRQyYjYrFnhlGAdk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ajVmlLznuwjNPTbq(self):
        """MULTIALLEN_ajVmlLznuwjNPTbq multi-experiment creation."""
        model_folder = 'MULTIALLEN_ajVmlLznuwjNPTbq'
        exp = {
            'experiment_name': ['MULTIALLEN_ajVmlLznuwjNPTbq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ajVmlLznuwjNPTbq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oPJyCiydGSYNbceM(self):
        """MULTIALLEN_oPJyCiydGSYNbceM multi-experiment creation."""
        model_folder = 'MULTIALLEN_oPJyCiydGSYNbceM'
        exp = {
            'experiment_name': ['MULTIALLEN_oPJyCiydGSYNbceM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oPJyCiydGSYNbceM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fLfrBGQjQCKwHvGs(self):
        """MULTIALLEN_fLfrBGQjQCKwHvGs multi-experiment creation."""
        model_folder = 'MULTIALLEN_fLfrBGQjQCKwHvGs'
        exp = {
            'experiment_name': ['MULTIALLEN_fLfrBGQjQCKwHvGs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fLfrBGQjQCKwHvGs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wwrfOsFPhHcFAqVC(self):
        """MULTIALLEN_wwrfOsFPhHcFAqVC multi-experiment creation."""
        model_folder = 'MULTIALLEN_wwrfOsFPhHcFAqVC'
        exp = {
            'experiment_name': ['MULTIALLEN_wwrfOsFPhHcFAqVC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wwrfOsFPhHcFAqVC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rSMmJhcdFOkXYjYY(self):
        """MULTIALLEN_rSMmJhcdFOkXYjYY multi-experiment creation."""
        model_folder = 'MULTIALLEN_rSMmJhcdFOkXYjYY'
        exp = {
            'experiment_name': ['MULTIALLEN_rSMmJhcdFOkXYjYY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rSMmJhcdFOkXYjYY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PzRsqvjIhdtENRLW(self):
        """MULTIALLEN_PzRsqvjIhdtENRLW multi-experiment creation."""
        model_folder = 'MULTIALLEN_PzRsqvjIhdtENRLW'
        exp = {
            'experiment_name': ['MULTIALLEN_PzRsqvjIhdtENRLW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PzRsqvjIhdtENRLW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QkSwySzGXISfUUuq(self):
        """MULTIALLEN_QkSwySzGXISfUUuq multi-experiment creation."""
        model_folder = 'MULTIALLEN_QkSwySzGXISfUUuq'
        exp = {
            'experiment_name': ['MULTIALLEN_QkSwySzGXISfUUuq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QkSwySzGXISfUUuq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FBOmJFAcOVQSaKZN(self):
        """MULTIALLEN_FBOmJFAcOVQSaKZN multi-experiment creation."""
        model_folder = 'MULTIALLEN_FBOmJFAcOVQSaKZN'
        exp = {
            'experiment_name': ['MULTIALLEN_FBOmJFAcOVQSaKZN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FBOmJFAcOVQSaKZN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hjqxcDIeOKxoVaGk(self):
        """MULTIALLEN_hjqxcDIeOKxoVaGk multi-experiment creation."""
        model_folder = 'MULTIALLEN_hjqxcDIeOKxoVaGk'
        exp = {
            'experiment_name': ['MULTIALLEN_hjqxcDIeOKxoVaGk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hjqxcDIeOKxoVaGk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dDrxtIIkISTkrmLZ(self):
        """MULTIALLEN_dDrxtIIkISTkrmLZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_dDrxtIIkISTkrmLZ'
        exp = {
            'experiment_name': ['MULTIALLEN_dDrxtIIkISTkrmLZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dDrxtIIkISTkrmLZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QRbFhDNDkVAhCnQV(self):
        """MULTIALLEN_QRbFhDNDkVAhCnQV multi-experiment creation."""
        model_folder = 'MULTIALLEN_QRbFhDNDkVAhCnQV'
        exp = {
            'experiment_name': ['MULTIALLEN_QRbFhDNDkVAhCnQV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QRbFhDNDkVAhCnQV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fkMiwCYmaDCgZeSC(self):
        """MULTIALLEN_fkMiwCYmaDCgZeSC multi-experiment creation."""
        model_folder = 'MULTIALLEN_fkMiwCYmaDCgZeSC'
        exp = {
            'experiment_name': ['MULTIALLEN_fkMiwCYmaDCgZeSC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fkMiwCYmaDCgZeSC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_czrNmljWtYNMBmlj(self):
        """MULTIALLEN_czrNmljWtYNMBmlj multi-experiment creation."""
        model_folder = 'MULTIALLEN_czrNmljWtYNMBmlj'
        exp = {
            'experiment_name': ['MULTIALLEN_czrNmljWtYNMBmlj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_czrNmljWtYNMBmlj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fHGHaYTlcXQAzBHI(self):
        """MULTIALLEN_fHGHaYTlcXQAzBHI multi-experiment creation."""
        model_folder = 'MULTIALLEN_fHGHaYTlcXQAzBHI'
        exp = {
            'experiment_name': ['MULTIALLEN_fHGHaYTlcXQAzBHI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fHGHaYTlcXQAzBHI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PjddoTmEutdJfxWD(self):
        """MULTIALLEN_PjddoTmEutdJfxWD multi-experiment creation."""
        model_folder = 'MULTIALLEN_PjddoTmEutdJfxWD'
        exp = {
            'experiment_name': ['MULTIALLEN_PjddoTmEutdJfxWD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PjddoTmEutdJfxWD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qIObmTebXVxTvstO(self):
        """MULTIALLEN_qIObmTebXVxTvstO multi-experiment creation."""
        model_folder = 'MULTIALLEN_qIObmTebXVxTvstO'
        exp = {
            'experiment_name': ['MULTIALLEN_qIObmTebXVxTvstO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qIObmTebXVxTvstO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gUtaLvvhHoulgEUL(self):
        """MULTIALLEN_gUtaLvvhHoulgEUL multi-experiment creation."""
        model_folder = 'MULTIALLEN_gUtaLvvhHoulgEUL'
        exp = {
            'experiment_name': ['MULTIALLEN_gUtaLvvhHoulgEUL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gUtaLvvhHoulgEUL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FUnTyCFsLWEjYPfl(self):
        """MULTIALLEN_FUnTyCFsLWEjYPfl multi-experiment creation."""
        model_folder = 'MULTIALLEN_FUnTyCFsLWEjYPfl'
        exp = {
            'experiment_name': ['MULTIALLEN_FUnTyCFsLWEjYPfl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FUnTyCFsLWEjYPfl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HFiYZtCNYmywlEQG(self):
        """MULTIALLEN_HFiYZtCNYmywlEQG multi-experiment creation."""
        model_folder = 'MULTIALLEN_HFiYZtCNYmywlEQG'
        exp = {
            'experiment_name': ['MULTIALLEN_HFiYZtCNYmywlEQG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HFiYZtCNYmywlEQG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sJnbNSCRchFDUfFM(self):
        """MULTIALLEN_sJnbNSCRchFDUfFM multi-experiment creation."""
        model_folder = 'MULTIALLEN_sJnbNSCRchFDUfFM'
        exp = {
            'experiment_name': ['MULTIALLEN_sJnbNSCRchFDUfFM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sJnbNSCRchFDUfFM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ERzacwpjYxSvrUON(self):
        """MULTIALLEN_ERzacwpjYxSvrUON multi-experiment creation."""
        model_folder = 'MULTIALLEN_ERzacwpjYxSvrUON'
        exp = {
            'experiment_name': ['MULTIALLEN_ERzacwpjYxSvrUON'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ERzacwpjYxSvrUON']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tgeiaObCFOCLtAlk(self):
        """MULTIALLEN_tgeiaObCFOCLtAlk multi-experiment creation."""
        model_folder = 'MULTIALLEN_tgeiaObCFOCLtAlk'
        exp = {
            'experiment_name': ['MULTIALLEN_tgeiaObCFOCLtAlk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tgeiaObCFOCLtAlk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TXSUwqsXmbUqEjUt(self):
        """MULTIALLEN_TXSUwqsXmbUqEjUt multi-experiment creation."""
        model_folder = 'MULTIALLEN_TXSUwqsXmbUqEjUt'
        exp = {
            'experiment_name': ['MULTIALLEN_TXSUwqsXmbUqEjUt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TXSUwqsXmbUqEjUt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KfohqAzQWKTDwslr(self):
        """MULTIALLEN_KfohqAzQWKTDwslr multi-experiment creation."""
        model_folder = 'MULTIALLEN_KfohqAzQWKTDwslr'
        exp = {
            'experiment_name': ['MULTIALLEN_KfohqAzQWKTDwslr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KfohqAzQWKTDwslr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WHyvCQRRGfiIanGp(self):
        """MULTIALLEN_WHyvCQRRGfiIanGp multi-experiment creation."""
        model_folder = 'MULTIALLEN_WHyvCQRRGfiIanGp'
        exp = {
            'experiment_name': ['MULTIALLEN_WHyvCQRRGfiIanGp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WHyvCQRRGfiIanGp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rgCIDWlVQZrBNigM(self):
        """MULTIALLEN_rgCIDWlVQZrBNigM multi-experiment creation."""
        model_folder = 'MULTIALLEN_rgCIDWlVQZrBNigM'
        exp = {
            'experiment_name': ['MULTIALLEN_rgCIDWlVQZrBNigM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rgCIDWlVQZrBNigM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_riIjILKuyHaLyTHy(self):
        """MULTIALLEN_riIjILKuyHaLyTHy multi-experiment creation."""
        model_folder = 'MULTIALLEN_riIjILKuyHaLyTHy'
        exp = {
            'experiment_name': ['MULTIALLEN_riIjILKuyHaLyTHy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_riIjILKuyHaLyTHy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JhJVoeoyRQWLrmwg(self):
        """MULTIALLEN_JhJVoeoyRQWLrmwg multi-experiment creation."""
        model_folder = 'MULTIALLEN_JhJVoeoyRQWLrmwg'
        exp = {
            'experiment_name': ['MULTIALLEN_JhJVoeoyRQWLrmwg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JhJVoeoyRQWLrmwg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TlxedlLlSoyLseXe(self):
        """MULTIALLEN_TlxedlLlSoyLseXe multi-experiment creation."""
        model_folder = 'MULTIALLEN_TlxedlLlSoyLseXe'
        exp = {
            'experiment_name': ['MULTIALLEN_TlxedlLlSoyLseXe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TlxedlLlSoyLseXe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RyGOuFWnrxhadQZT(self):
        """MULTIALLEN_RyGOuFWnrxhadQZT multi-experiment creation."""
        model_folder = 'MULTIALLEN_RyGOuFWnrxhadQZT'
        exp = {
            'experiment_name': ['MULTIALLEN_RyGOuFWnrxhadQZT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RyGOuFWnrxhadQZT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vdAxdAsyBVMuXDDn(self):
        """MULTIALLEN_vdAxdAsyBVMuXDDn multi-experiment creation."""
        model_folder = 'MULTIALLEN_vdAxdAsyBVMuXDDn'
        exp = {
            'experiment_name': ['MULTIALLEN_vdAxdAsyBVMuXDDn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vdAxdAsyBVMuXDDn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WaAvCSJFuxBPCxyF(self):
        """MULTIALLEN_WaAvCSJFuxBPCxyF multi-experiment creation."""
        model_folder = 'MULTIALLEN_WaAvCSJFuxBPCxyF'
        exp = {
            'experiment_name': ['MULTIALLEN_WaAvCSJFuxBPCxyF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WaAvCSJFuxBPCxyF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pHEiJnxYOKNhREfl(self):
        """MULTIALLEN_pHEiJnxYOKNhREfl multi-experiment creation."""
        model_folder = 'MULTIALLEN_pHEiJnxYOKNhREfl'
        exp = {
            'experiment_name': ['MULTIALLEN_pHEiJnxYOKNhREfl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pHEiJnxYOKNhREfl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PYsgIBJjRbOsEqPF(self):
        """MULTIALLEN_PYsgIBJjRbOsEqPF multi-experiment creation."""
        model_folder = 'MULTIALLEN_PYsgIBJjRbOsEqPF'
        exp = {
            'experiment_name': ['MULTIALLEN_PYsgIBJjRbOsEqPF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PYsgIBJjRbOsEqPF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HjElPRDmhuDhpNEV(self):
        """MULTIALLEN_HjElPRDmhuDhpNEV multi-experiment creation."""
        model_folder = 'MULTIALLEN_HjElPRDmhuDhpNEV'
        exp = {
            'experiment_name': ['MULTIALLEN_HjElPRDmhuDhpNEV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HjElPRDmhuDhpNEV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZxlArAKvxeMFukmh(self):
        """MULTIALLEN_ZxlArAKvxeMFukmh multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZxlArAKvxeMFukmh'
        exp = {
            'experiment_name': ['MULTIALLEN_ZxlArAKvxeMFukmh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZxlArAKvxeMFukmh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VDKmygONybYMLAHV(self):
        """MULTIALLEN_VDKmygONybYMLAHV multi-experiment creation."""
        model_folder = 'MULTIALLEN_VDKmygONybYMLAHV'
        exp = {
            'experiment_name': ['MULTIALLEN_VDKmygONybYMLAHV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VDKmygONybYMLAHV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dpteXXHnCkoEyGVg(self):
        """MULTIALLEN_dpteXXHnCkoEyGVg multi-experiment creation."""
        model_folder = 'MULTIALLEN_dpteXXHnCkoEyGVg'
        exp = {
            'experiment_name': ['MULTIALLEN_dpteXXHnCkoEyGVg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dpteXXHnCkoEyGVg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hPMyGZLDjTVSdOcq(self):
        """MULTIALLEN_hPMyGZLDjTVSdOcq multi-experiment creation."""
        model_folder = 'MULTIALLEN_hPMyGZLDjTVSdOcq'
        exp = {
            'experiment_name': ['MULTIALLEN_hPMyGZLDjTVSdOcq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hPMyGZLDjTVSdOcq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nydRSyZGvwXQtqQn(self):
        """MULTIALLEN_nydRSyZGvwXQtqQn multi-experiment creation."""
        model_folder = 'MULTIALLEN_nydRSyZGvwXQtqQn'
        exp = {
            'experiment_name': ['MULTIALLEN_nydRSyZGvwXQtqQn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nydRSyZGvwXQtqQn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qxghmMLZgKliycZV(self):
        """MULTIALLEN_qxghmMLZgKliycZV multi-experiment creation."""
        model_folder = 'MULTIALLEN_qxghmMLZgKliycZV'
        exp = {
            'experiment_name': ['MULTIALLEN_qxghmMLZgKliycZV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qxghmMLZgKliycZV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zTVofwdyHhomExKT(self):
        """MULTIALLEN_zTVofwdyHhomExKT multi-experiment creation."""
        model_folder = 'MULTIALLEN_zTVofwdyHhomExKT'
        exp = {
            'experiment_name': ['MULTIALLEN_zTVofwdyHhomExKT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zTVofwdyHhomExKT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ktrsBcZZoAsMJoCC(self):
        """MULTIALLEN_ktrsBcZZoAsMJoCC multi-experiment creation."""
        model_folder = 'MULTIALLEN_ktrsBcZZoAsMJoCC'
        exp = {
            'experiment_name': ['MULTIALLEN_ktrsBcZZoAsMJoCC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ktrsBcZZoAsMJoCC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZTvXwlHkEPXPidqY(self):
        """MULTIALLEN_ZTvXwlHkEPXPidqY multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZTvXwlHkEPXPidqY'
        exp = {
            'experiment_name': ['MULTIALLEN_ZTvXwlHkEPXPidqY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZTvXwlHkEPXPidqY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZuDIIfcihIFvsQgN(self):
        """MULTIALLEN_ZuDIIfcihIFvsQgN multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZuDIIfcihIFvsQgN'
        exp = {
            'experiment_name': ['MULTIALLEN_ZuDIIfcihIFvsQgN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZuDIIfcihIFvsQgN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZyJEFaJdnKDwasUS(self):
        """MULTIALLEN_ZyJEFaJdnKDwasUS multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZyJEFaJdnKDwasUS'
        exp = {
            'experiment_name': ['MULTIALLEN_ZyJEFaJdnKDwasUS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZyJEFaJdnKDwasUS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SQBYzbrWqKeTyZfm(self):
        """MULTIALLEN_SQBYzbrWqKeTyZfm multi-experiment creation."""
        model_folder = 'MULTIALLEN_SQBYzbrWqKeTyZfm'
        exp = {
            'experiment_name': ['MULTIALLEN_SQBYzbrWqKeTyZfm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SQBYzbrWqKeTyZfm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tUyUtChqXsiebwgW(self):
        """MULTIALLEN_tUyUtChqXsiebwgW multi-experiment creation."""
        model_folder = 'MULTIALLEN_tUyUtChqXsiebwgW'
        exp = {
            'experiment_name': ['MULTIALLEN_tUyUtChqXsiebwgW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tUyUtChqXsiebwgW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bLxdQeJGMHNAEKfi(self):
        """MULTIALLEN_bLxdQeJGMHNAEKfi multi-experiment creation."""
        model_folder = 'MULTIALLEN_bLxdQeJGMHNAEKfi'
        exp = {
            'experiment_name': ['MULTIALLEN_bLxdQeJGMHNAEKfi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bLxdQeJGMHNAEKfi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DbDAhOjFuPOBgDHx(self):
        """MULTIALLEN_DbDAhOjFuPOBgDHx multi-experiment creation."""
        model_folder = 'MULTIALLEN_DbDAhOjFuPOBgDHx'
        exp = {
            'experiment_name': ['MULTIALLEN_DbDAhOjFuPOBgDHx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DbDAhOjFuPOBgDHx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MPcYscZikGJgOjrz(self):
        """MULTIALLEN_MPcYscZikGJgOjrz multi-experiment creation."""
        model_folder = 'MULTIALLEN_MPcYscZikGJgOjrz'
        exp = {
            'experiment_name': ['MULTIALLEN_MPcYscZikGJgOjrz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MPcYscZikGJgOjrz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bSufxrDbQEYtMiPK(self):
        """MULTIALLEN_bSufxrDbQEYtMiPK multi-experiment creation."""
        model_folder = 'MULTIALLEN_bSufxrDbQEYtMiPK'
        exp = {
            'experiment_name': ['MULTIALLEN_bSufxrDbQEYtMiPK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bSufxrDbQEYtMiPK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CZpyJQdUnCzxkMya(self):
        """MULTIALLEN_CZpyJQdUnCzxkMya multi-experiment creation."""
        model_folder = 'MULTIALLEN_CZpyJQdUnCzxkMya'
        exp = {
            'experiment_name': ['MULTIALLEN_CZpyJQdUnCzxkMya'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CZpyJQdUnCzxkMya']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eIHlRYZCMfTautUx(self):
        """MULTIALLEN_eIHlRYZCMfTautUx multi-experiment creation."""
        model_folder = 'MULTIALLEN_eIHlRYZCMfTautUx'
        exp = {
            'experiment_name': ['MULTIALLEN_eIHlRYZCMfTautUx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eIHlRYZCMfTautUx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aeFZqCplwOhftAmM(self):
        """MULTIALLEN_aeFZqCplwOhftAmM multi-experiment creation."""
        model_folder = 'MULTIALLEN_aeFZqCplwOhftAmM'
        exp = {
            'experiment_name': ['MULTIALLEN_aeFZqCplwOhftAmM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aeFZqCplwOhftAmM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xycclVHnWdMAqyIT(self):
        """MULTIALLEN_xycclVHnWdMAqyIT multi-experiment creation."""
        model_folder = 'MULTIALLEN_xycclVHnWdMAqyIT'
        exp = {
            'experiment_name': ['MULTIALLEN_xycclVHnWdMAqyIT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xycclVHnWdMAqyIT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ktGhhMUxbeMUcrRu(self):
        """MULTIALLEN_ktGhhMUxbeMUcrRu multi-experiment creation."""
        model_folder = 'MULTIALLEN_ktGhhMUxbeMUcrRu'
        exp = {
            'experiment_name': ['MULTIALLEN_ktGhhMUxbeMUcrRu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ktGhhMUxbeMUcrRu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HJnNGJqRRWnaJHMy(self):
        """MULTIALLEN_HJnNGJqRRWnaJHMy multi-experiment creation."""
        model_folder = 'MULTIALLEN_HJnNGJqRRWnaJHMy'
        exp = {
            'experiment_name': ['MULTIALLEN_HJnNGJqRRWnaJHMy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HJnNGJqRRWnaJHMy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cWRcuACoAtPvOIpH(self):
        """MULTIALLEN_cWRcuACoAtPvOIpH multi-experiment creation."""
        model_folder = 'MULTIALLEN_cWRcuACoAtPvOIpH'
        exp = {
            'experiment_name': ['MULTIALLEN_cWRcuACoAtPvOIpH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cWRcuACoAtPvOIpH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OybYXrRQiBRROSpF(self):
        """MULTIALLEN_OybYXrRQiBRROSpF multi-experiment creation."""
        model_folder = 'MULTIALLEN_OybYXrRQiBRROSpF'
        exp = {
            'experiment_name': ['MULTIALLEN_OybYXrRQiBRROSpF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OybYXrRQiBRROSpF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uLRIhdRDhAlOoEiD(self):
        """MULTIALLEN_uLRIhdRDhAlOoEiD multi-experiment creation."""
        model_folder = 'MULTIALLEN_uLRIhdRDhAlOoEiD'
        exp = {
            'experiment_name': ['MULTIALLEN_uLRIhdRDhAlOoEiD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uLRIhdRDhAlOoEiD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XbgzvIMktzpOkCtH(self):
        """MULTIALLEN_XbgzvIMktzpOkCtH multi-experiment creation."""
        model_folder = 'MULTIALLEN_XbgzvIMktzpOkCtH'
        exp = {
            'experiment_name': ['MULTIALLEN_XbgzvIMktzpOkCtH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XbgzvIMktzpOkCtH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WeRdfRsGHjusiaHW(self):
        """MULTIALLEN_WeRdfRsGHjusiaHW multi-experiment creation."""
        model_folder = 'MULTIALLEN_WeRdfRsGHjusiaHW'
        exp = {
            'experiment_name': ['MULTIALLEN_WeRdfRsGHjusiaHW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WeRdfRsGHjusiaHW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xbiVnvedgHqdQBtp(self):
        """MULTIALLEN_xbiVnvedgHqdQBtp multi-experiment creation."""
        model_folder = 'MULTIALLEN_xbiVnvedgHqdQBtp'
        exp = {
            'experiment_name': ['MULTIALLEN_xbiVnvedgHqdQBtp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xbiVnvedgHqdQBtp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HlvITHrAIUJrgOkD(self):
        """MULTIALLEN_HlvITHrAIUJrgOkD multi-experiment creation."""
        model_folder = 'MULTIALLEN_HlvITHrAIUJrgOkD'
        exp = {
            'experiment_name': ['MULTIALLEN_HlvITHrAIUJrgOkD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HlvITHrAIUJrgOkD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uOslgkSQuMBOAmkv(self):
        """MULTIALLEN_uOslgkSQuMBOAmkv multi-experiment creation."""
        model_folder = 'MULTIALLEN_uOslgkSQuMBOAmkv'
        exp = {
            'experiment_name': ['MULTIALLEN_uOslgkSQuMBOAmkv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uOslgkSQuMBOAmkv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yUBoGdRHoIeSSBUi(self):
        """MULTIALLEN_yUBoGdRHoIeSSBUi multi-experiment creation."""
        model_folder = 'MULTIALLEN_yUBoGdRHoIeSSBUi'
        exp = {
            'experiment_name': ['MULTIALLEN_yUBoGdRHoIeSSBUi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yUBoGdRHoIeSSBUi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DgXFnPMBIepcTSbn(self):
        """MULTIALLEN_DgXFnPMBIepcTSbn multi-experiment creation."""
        model_folder = 'MULTIALLEN_DgXFnPMBIepcTSbn'
        exp = {
            'experiment_name': ['MULTIALLEN_DgXFnPMBIepcTSbn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DgXFnPMBIepcTSbn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DUmFirhfqdbttUsV(self):
        """MULTIALLEN_DUmFirhfqdbttUsV multi-experiment creation."""
        model_folder = 'MULTIALLEN_DUmFirhfqdbttUsV'
        exp = {
            'experiment_name': ['MULTIALLEN_DUmFirhfqdbttUsV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DUmFirhfqdbttUsV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EmhmjvZQwsLysemM(self):
        """MULTIALLEN_EmhmjvZQwsLysemM multi-experiment creation."""
        model_folder = 'MULTIALLEN_EmhmjvZQwsLysemM'
        exp = {
            'experiment_name': ['MULTIALLEN_EmhmjvZQwsLysemM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EmhmjvZQwsLysemM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mocmAtxbpGzzGdtM(self):
        """MULTIALLEN_mocmAtxbpGzzGdtM multi-experiment creation."""
        model_folder = 'MULTIALLEN_mocmAtxbpGzzGdtM'
        exp = {
            'experiment_name': ['MULTIALLEN_mocmAtxbpGzzGdtM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mocmAtxbpGzzGdtM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pqabYktzayeZFGCs(self):
        """MULTIALLEN_pqabYktzayeZFGCs multi-experiment creation."""
        model_folder = 'MULTIALLEN_pqabYktzayeZFGCs'
        exp = {
            'experiment_name': ['MULTIALLEN_pqabYktzayeZFGCs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pqabYktzayeZFGCs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DsogsNLyzeBcTPzE(self):
        """MULTIALLEN_DsogsNLyzeBcTPzE multi-experiment creation."""
        model_folder = 'MULTIALLEN_DsogsNLyzeBcTPzE'
        exp = {
            'experiment_name': ['MULTIALLEN_DsogsNLyzeBcTPzE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DsogsNLyzeBcTPzE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FEprpuPYhnNmKYAp(self):
        """MULTIALLEN_FEprpuPYhnNmKYAp multi-experiment creation."""
        model_folder = 'MULTIALLEN_FEprpuPYhnNmKYAp'
        exp = {
            'experiment_name': ['MULTIALLEN_FEprpuPYhnNmKYAp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FEprpuPYhnNmKYAp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rSOZUjGlKAnLYlQj(self):
        """MULTIALLEN_rSOZUjGlKAnLYlQj multi-experiment creation."""
        model_folder = 'MULTIALLEN_rSOZUjGlKAnLYlQj'
        exp = {
            'experiment_name': ['MULTIALLEN_rSOZUjGlKAnLYlQj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rSOZUjGlKAnLYlQj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VSBLYerMixLalVAo(self):
        """MULTIALLEN_VSBLYerMixLalVAo multi-experiment creation."""
        model_folder = 'MULTIALLEN_VSBLYerMixLalVAo'
        exp = {
            'experiment_name': ['MULTIALLEN_VSBLYerMixLalVAo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VSBLYerMixLalVAo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_StHVUFPmOTIdxJDT(self):
        """MULTIALLEN_StHVUFPmOTIdxJDT multi-experiment creation."""
        model_folder = 'MULTIALLEN_StHVUFPmOTIdxJDT'
        exp = {
            'experiment_name': ['MULTIALLEN_StHVUFPmOTIdxJDT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_StHVUFPmOTIdxJDT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BeKzPCOaRJpmxcJX(self):
        """MULTIALLEN_BeKzPCOaRJpmxcJX multi-experiment creation."""
        model_folder = 'MULTIALLEN_BeKzPCOaRJpmxcJX'
        exp = {
            'experiment_name': ['MULTIALLEN_BeKzPCOaRJpmxcJX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BeKzPCOaRJpmxcJX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PmKobqtGVsmxFjbC(self):
        """MULTIALLEN_PmKobqtGVsmxFjbC multi-experiment creation."""
        model_folder = 'MULTIALLEN_PmKobqtGVsmxFjbC'
        exp = {
            'experiment_name': ['MULTIALLEN_PmKobqtGVsmxFjbC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PmKobqtGVsmxFjbC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qUAqepfucYfZAFwR(self):
        """MULTIALLEN_qUAqepfucYfZAFwR multi-experiment creation."""
        model_folder = 'MULTIALLEN_qUAqepfucYfZAFwR'
        exp = {
            'experiment_name': ['MULTIALLEN_qUAqepfucYfZAFwR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qUAqepfucYfZAFwR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aBLkzqalAwuCuSKX(self):
        """MULTIALLEN_aBLkzqalAwuCuSKX multi-experiment creation."""
        model_folder = 'MULTIALLEN_aBLkzqalAwuCuSKX'
        exp = {
            'experiment_name': ['MULTIALLEN_aBLkzqalAwuCuSKX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aBLkzqalAwuCuSKX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tZtzKaOnmCsOhQDF(self):
        """MULTIALLEN_tZtzKaOnmCsOhQDF multi-experiment creation."""
        model_folder = 'MULTIALLEN_tZtzKaOnmCsOhQDF'
        exp = {
            'experiment_name': ['MULTIALLEN_tZtzKaOnmCsOhQDF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tZtzKaOnmCsOhQDF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DQUdtekoXffZylgb(self):
        """MULTIALLEN_DQUdtekoXffZylgb multi-experiment creation."""
        model_folder = 'MULTIALLEN_DQUdtekoXffZylgb'
        exp = {
            'experiment_name': ['MULTIALLEN_DQUdtekoXffZylgb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DQUdtekoXffZylgb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lewcUvSyMAqqJrGC(self):
        """MULTIALLEN_lewcUvSyMAqqJrGC multi-experiment creation."""
        model_folder = 'MULTIALLEN_lewcUvSyMAqqJrGC'
        exp = {
            'experiment_name': ['MULTIALLEN_lewcUvSyMAqqJrGC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lewcUvSyMAqqJrGC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fQqDEUDWURNWaJYm(self):
        """MULTIALLEN_fQqDEUDWURNWaJYm multi-experiment creation."""
        model_folder = 'MULTIALLEN_fQqDEUDWURNWaJYm'
        exp = {
            'experiment_name': ['MULTIALLEN_fQqDEUDWURNWaJYm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fQqDEUDWURNWaJYm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LVIJLVJcsWSJGPGh(self):
        """MULTIALLEN_LVIJLVJcsWSJGPGh multi-experiment creation."""
        model_folder = 'MULTIALLEN_LVIJLVJcsWSJGPGh'
        exp = {
            'experiment_name': ['MULTIALLEN_LVIJLVJcsWSJGPGh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LVIJLVJcsWSJGPGh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RZFOVSzplOuvbjrB(self):
        """MULTIALLEN_RZFOVSzplOuvbjrB multi-experiment creation."""
        model_folder = 'MULTIALLEN_RZFOVSzplOuvbjrB'
        exp = {
            'experiment_name': ['MULTIALLEN_RZFOVSzplOuvbjrB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RZFOVSzplOuvbjrB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fJNLMJyUFbgiJAOo(self):
        """MULTIALLEN_fJNLMJyUFbgiJAOo multi-experiment creation."""
        model_folder = 'MULTIALLEN_fJNLMJyUFbgiJAOo'
        exp = {
            'experiment_name': ['MULTIALLEN_fJNLMJyUFbgiJAOo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fJNLMJyUFbgiJAOo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eHsIWpdjbEHKLAsm(self):
        """MULTIALLEN_eHsIWpdjbEHKLAsm multi-experiment creation."""
        model_folder = 'MULTIALLEN_eHsIWpdjbEHKLAsm'
        exp = {
            'experiment_name': ['MULTIALLEN_eHsIWpdjbEHKLAsm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eHsIWpdjbEHKLAsm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SgTAnBGAJxQlKyXs(self):
        """MULTIALLEN_SgTAnBGAJxQlKyXs multi-experiment creation."""
        model_folder = 'MULTIALLEN_SgTAnBGAJxQlKyXs'
        exp = {
            'experiment_name': ['MULTIALLEN_SgTAnBGAJxQlKyXs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SgTAnBGAJxQlKyXs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XiPeqVdYYkEMLOcG(self):
        """MULTIALLEN_XiPeqVdYYkEMLOcG multi-experiment creation."""
        model_folder = 'MULTIALLEN_XiPeqVdYYkEMLOcG'
        exp = {
            'experiment_name': ['MULTIALLEN_XiPeqVdYYkEMLOcG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XiPeqVdYYkEMLOcG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hzoLAyqfjMQAJnDF(self):
        """MULTIALLEN_hzoLAyqfjMQAJnDF multi-experiment creation."""
        model_folder = 'MULTIALLEN_hzoLAyqfjMQAJnDF'
        exp = {
            'experiment_name': ['MULTIALLEN_hzoLAyqfjMQAJnDF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hzoLAyqfjMQAJnDF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kISwmNhhkyaHQSyU(self):
        """MULTIALLEN_kISwmNhhkyaHQSyU multi-experiment creation."""
        model_folder = 'MULTIALLEN_kISwmNhhkyaHQSyU'
        exp = {
            'experiment_name': ['MULTIALLEN_kISwmNhhkyaHQSyU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kISwmNhhkyaHQSyU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SsxrovRpkwVrAMfV(self):
        """MULTIALLEN_SsxrovRpkwVrAMfV multi-experiment creation."""
        model_folder = 'MULTIALLEN_SsxrovRpkwVrAMfV'
        exp = {
            'experiment_name': ['MULTIALLEN_SsxrovRpkwVrAMfV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SsxrovRpkwVrAMfV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_puuhsyqDCvvcirrj(self):
        """MULTIALLEN_puuhsyqDCvvcirrj multi-experiment creation."""
        model_folder = 'MULTIALLEN_puuhsyqDCvvcirrj'
        exp = {
            'experiment_name': ['MULTIALLEN_puuhsyqDCvvcirrj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_puuhsyqDCvvcirrj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ppuBPDvmKQUMjnuk(self):
        """MULTIALLEN_ppuBPDvmKQUMjnuk multi-experiment creation."""
        model_folder = 'MULTIALLEN_ppuBPDvmKQUMjnuk'
        exp = {
            'experiment_name': ['MULTIALLEN_ppuBPDvmKQUMjnuk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ppuBPDvmKQUMjnuk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ErDvdaPdbVQNZCTB(self):
        """MULTIALLEN_ErDvdaPdbVQNZCTB multi-experiment creation."""
        model_folder = 'MULTIALLEN_ErDvdaPdbVQNZCTB'
        exp = {
            'experiment_name': ['MULTIALLEN_ErDvdaPdbVQNZCTB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ErDvdaPdbVQNZCTB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CKodVZnQuDyrcJKb(self):
        """MULTIALLEN_CKodVZnQuDyrcJKb multi-experiment creation."""
        model_folder = 'MULTIALLEN_CKodVZnQuDyrcJKb'
        exp = {
            'experiment_name': ['MULTIALLEN_CKodVZnQuDyrcJKb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CKodVZnQuDyrcJKb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hntqGJjdxSijfHvp(self):
        """MULTIALLEN_hntqGJjdxSijfHvp multi-experiment creation."""
        model_folder = 'MULTIALLEN_hntqGJjdxSijfHvp'
        exp = {
            'experiment_name': ['MULTIALLEN_hntqGJjdxSijfHvp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hntqGJjdxSijfHvp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hVwOshuUXTWxLyKf(self):
        """MULTIALLEN_hVwOshuUXTWxLyKf multi-experiment creation."""
        model_folder = 'MULTIALLEN_hVwOshuUXTWxLyKf'
        exp = {
            'experiment_name': ['MULTIALLEN_hVwOshuUXTWxLyKf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hVwOshuUXTWxLyKf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PRNtyuMJIcQMPnXe(self):
        """MULTIALLEN_PRNtyuMJIcQMPnXe multi-experiment creation."""
        model_folder = 'MULTIALLEN_PRNtyuMJIcQMPnXe'
        exp = {
            'experiment_name': ['MULTIALLEN_PRNtyuMJIcQMPnXe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PRNtyuMJIcQMPnXe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YLmWEancvDNIacQn(self):
        """MULTIALLEN_YLmWEancvDNIacQn multi-experiment creation."""
        model_folder = 'MULTIALLEN_YLmWEancvDNIacQn'
        exp = {
            'experiment_name': ['MULTIALLEN_YLmWEancvDNIacQn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YLmWEancvDNIacQn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qUdFwFSdVBpPRquW(self):
        """MULTIALLEN_qUdFwFSdVBpPRquW multi-experiment creation."""
        model_folder = 'MULTIALLEN_qUdFwFSdVBpPRquW'
        exp = {
            'experiment_name': ['MULTIALLEN_qUdFwFSdVBpPRquW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qUdFwFSdVBpPRquW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qKEPmdRNTiyEOVfp(self):
        """MULTIALLEN_qKEPmdRNTiyEOVfp multi-experiment creation."""
        model_folder = 'MULTIALLEN_qKEPmdRNTiyEOVfp'
        exp = {
            'experiment_name': ['MULTIALLEN_qKEPmdRNTiyEOVfp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qKEPmdRNTiyEOVfp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TngexMaSKZcNsAYj(self):
        """MULTIALLEN_TngexMaSKZcNsAYj multi-experiment creation."""
        model_folder = 'MULTIALLEN_TngexMaSKZcNsAYj'
        exp = {
            'experiment_name': ['MULTIALLEN_TngexMaSKZcNsAYj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TngexMaSKZcNsAYj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NAlWBcbYbKSGywKU(self):
        """MULTIALLEN_NAlWBcbYbKSGywKU multi-experiment creation."""
        model_folder = 'MULTIALLEN_NAlWBcbYbKSGywKU'
        exp = {
            'experiment_name': ['MULTIALLEN_NAlWBcbYbKSGywKU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NAlWBcbYbKSGywKU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eqnEveyYDHRNTCSV(self):
        """MULTIALLEN_eqnEveyYDHRNTCSV multi-experiment creation."""
        model_folder = 'MULTIALLEN_eqnEveyYDHRNTCSV'
        exp = {
            'experiment_name': ['MULTIALLEN_eqnEveyYDHRNTCSV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eqnEveyYDHRNTCSV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JVLrPDghpzIakfiR(self):
        """MULTIALLEN_JVLrPDghpzIakfiR multi-experiment creation."""
        model_folder = 'MULTIALLEN_JVLrPDghpzIakfiR'
        exp = {
            'experiment_name': ['MULTIALLEN_JVLrPDghpzIakfiR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JVLrPDghpzIakfiR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WOjmtWKZcmYuWmGI(self):
        """MULTIALLEN_WOjmtWKZcmYuWmGI multi-experiment creation."""
        model_folder = 'MULTIALLEN_WOjmtWKZcmYuWmGI'
        exp = {
            'experiment_name': ['MULTIALLEN_WOjmtWKZcmYuWmGI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WOjmtWKZcmYuWmGI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SoLMPDdGsaXpTTWH(self):
        """MULTIALLEN_SoLMPDdGsaXpTTWH multi-experiment creation."""
        model_folder = 'MULTIALLEN_SoLMPDdGsaXpTTWH'
        exp = {
            'experiment_name': ['MULTIALLEN_SoLMPDdGsaXpTTWH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SoLMPDdGsaXpTTWH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PMVwEvTXcfdmyBAn(self):
        """MULTIALLEN_PMVwEvTXcfdmyBAn multi-experiment creation."""
        model_folder = 'MULTIALLEN_PMVwEvTXcfdmyBAn'
        exp = {
            'experiment_name': ['MULTIALLEN_PMVwEvTXcfdmyBAn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PMVwEvTXcfdmyBAn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fGXfgNdqfOFWfVVI(self):
        """MULTIALLEN_fGXfgNdqfOFWfVVI multi-experiment creation."""
        model_folder = 'MULTIALLEN_fGXfgNdqfOFWfVVI'
        exp = {
            'experiment_name': ['MULTIALLEN_fGXfgNdqfOFWfVVI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fGXfgNdqfOFWfVVI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ifveDTpXVmMUJwuj(self):
        """MULTIALLEN_ifveDTpXVmMUJwuj multi-experiment creation."""
        model_folder = 'MULTIALLEN_ifveDTpXVmMUJwuj'
        exp = {
            'experiment_name': ['MULTIALLEN_ifveDTpXVmMUJwuj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ifveDTpXVmMUJwuj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nkfYLOcuaHCDNTNq(self):
        """MULTIALLEN_nkfYLOcuaHCDNTNq multi-experiment creation."""
        model_folder = 'MULTIALLEN_nkfYLOcuaHCDNTNq'
        exp = {
            'experiment_name': ['MULTIALLEN_nkfYLOcuaHCDNTNq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nkfYLOcuaHCDNTNq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TbFEQBnIYeuZoiZR(self):
        """MULTIALLEN_TbFEQBnIYeuZoiZR multi-experiment creation."""
        model_folder = 'MULTIALLEN_TbFEQBnIYeuZoiZR'
        exp = {
            'experiment_name': ['MULTIALLEN_TbFEQBnIYeuZoiZR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TbFEQBnIYeuZoiZR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gVfVAublkVJFPksA(self):
        """MULTIALLEN_gVfVAublkVJFPksA multi-experiment creation."""
        model_folder = 'MULTIALLEN_gVfVAublkVJFPksA'
        exp = {
            'experiment_name': ['MULTIALLEN_gVfVAublkVJFPksA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gVfVAublkVJFPksA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uKCOCLBuzIDBmVbx(self):
        """MULTIALLEN_uKCOCLBuzIDBmVbx multi-experiment creation."""
        model_folder = 'MULTIALLEN_uKCOCLBuzIDBmVbx'
        exp = {
            'experiment_name': ['MULTIALLEN_uKCOCLBuzIDBmVbx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uKCOCLBuzIDBmVbx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fhZTQWTqwmebIGar(self):
        """MULTIALLEN_fhZTQWTqwmebIGar multi-experiment creation."""
        model_folder = 'MULTIALLEN_fhZTQWTqwmebIGar'
        exp = {
            'experiment_name': ['MULTIALLEN_fhZTQWTqwmebIGar'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fhZTQWTqwmebIGar']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IcJAYkTHLdCSgpGn(self):
        """MULTIALLEN_IcJAYkTHLdCSgpGn multi-experiment creation."""
        model_folder = 'MULTIALLEN_IcJAYkTHLdCSgpGn'
        exp = {
            'experiment_name': ['MULTIALLEN_IcJAYkTHLdCSgpGn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IcJAYkTHLdCSgpGn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kGCFOUVpwvnCrWkP(self):
        """MULTIALLEN_kGCFOUVpwvnCrWkP multi-experiment creation."""
        model_folder = 'MULTIALLEN_kGCFOUVpwvnCrWkP'
        exp = {
            'experiment_name': ['MULTIALLEN_kGCFOUVpwvnCrWkP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kGCFOUVpwvnCrWkP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DdSkQXAHvmDsWiGa(self):
        """MULTIALLEN_DdSkQXAHvmDsWiGa multi-experiment creation."""
        model_folder = 'MULTIALLEN_DdSkQXAHvmDsWiGa'
        exp = {
            'experiment_name': ['MULTIALLEN_DdSkQXAHvmDsWiGa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DdSkQXAHvmDsWiGa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pMQZLlAHsANdWEye(self):
        """MULTIALLEN_pMQZLlAHsANdWEye multi-experiment creation."""
        model_folder = 'MULTIALLEN_pMQZLlAHsANdWEye'
        exp = {
            'experiment_name': ['MULTIALLEN_pMQZLlAHsANdWEye'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pMQZLlAHsANdWEye']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aTGOwUqRCyPQahQD(self):
        """MULTIALLEN_aTGOwUqRCyPQahQD multi-experiment creation."""
        model_folder = 'MULTIALLEN_aTGOwUqRCyPQahQD'
        exp = {
            'experiment_name': ['MULTIALLEN_aTGOwUqRCyPQahQD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aTGOwUqRCyPQahQD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JBtFmHqdbDAkTEFH(self):
        """MULTIALLEN_JBtFmHqdbDAkTEFH multi-experiment creation."""
        model_folder = 'MULTIALLEN_JBtFmHqdbDAkTEFH'
        exp = {
            'experiment_name': ['MULTIALLEN_JBtFmHqdbDAkTEFH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JBtFmHqdbDAkTEFH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dgKOiLUPXcrYnMWL(self):
        """MULTIALLEN_dgKOiLUPXcrYnMWL multi-experiment creation."""
        model_folder = 'MULTIALLEN_dgKOiLUPXcrYnMWL'
        exp = {
            'experiment_name': ['MULTIALLEN_dgKOiLUPXcrYnMWL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dgKOiLUPXcrYnMWL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YGReWjoCNGQHyXqv(self):
        """MULTIALLEN_YGReWjoCNGQHyXqv multi-experiment creation."""
        model_folder = 'MULTIALLEN_YGReWjoCNGQHyXqv'
        exp = {
            'experiment_name': ['MULTIALLEN_YGReWjoCNGQHyXqv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YGReWjoCNGQHyXqv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KVHDZJjSSHQyrFJe(self):
        """MULTIALLEN_KVHDZJjSSHQyrFJe multi-experiment creation."""
        model_folder = 'MULTIALLEN_KVHDZJjSSHQyrFJe'
        exp = {
            'experiment_name': ['MULTIALLEN_KVHDZJjSSHQyrFJe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KVHDZJjSSHQyrFJe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vaWGLpyKYpqriklb(self):
        """MULTIALLEN_vaWGLpyKYpqriklb multi-experiment creation."""
        model_folder = 'MULTIALLEN_vaWGLpyKYpqriklb'
        exp = {
            'experiment_name': ['MULTIALLEN_vaWGLpyKYpqriklb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vaWGLpyKYpqriklb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XiVCKirwPAIORkum(self):
        """MULTIALLEN_XiVCKirwPAIORkum multi-experiment creation."""
        model_folder = 'MULTIALLEN_XiVCKirwPAIORkum'
        exp = {
            'experiment_name': ['MULTIALLEN_XiVCKirwPAIORkum'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XiVCKirwPAIORkum']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mbuEfjzlCezISBon(self):
        """MULTIALLEN_mbuEfjzlCezISBon multi-experiment creation."""
        model_folder = 'MULTIALLEN_mbuEfjzlCezISBon'
        exp = {
            'experiment_name': ['MULTIALLEN_mbuEfjzlCezISBon'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mbuEfjzlCezISBon']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SuikUChYfQYLPwAG(self):
        """MULTIALLEN_SuikUChYfQYLPwAG multi-experiment creation."""
        model_folder = 'MULTIALLEN_SuikUChYfQYLPwAG'
        exp = {
            'experiment_name': ['MULTIALLEN_SuikUChYfQYLPwAG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SuikUChYfQYLPwAG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DnSKnYAkhrVReAha(self):
        """MULTIALLEN_DnSKnYAkhrVReAha multi-experiment creation."""
        model_folder = 'MULTIALLEN_DnSKnYAkhrVReAha'
        exp = {
            'experiment_name': ['MULTIALLEN_DnSKnYAkhrVReAha'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DnSKnYAkhrVReAha']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yAjEgdUukWSjXBpz(self):
        """MULTIALLEN_yAjEgdUukWSjXBpz multi-experiment creation."""
        model_folder = 'MULTIALLEN_yAjEgdUukWSjXBpz'
        exp = {
            'experiment_name': ['MULTIALLEN_yAjEgdUukWSjXBpz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yAjEgdUukWSjXBpz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cCrprwyrUjLiizSS(self):
        """MULTIALLEN_cCrprwyrUjLiizSS multi-experiment creation."""
        model_folder = 'MULTIALLEN_cCrprwyrUjLiizSS'
        exp = {
            'experiment_name': ['MULTIALLEN_cCrprwyrUjLiizSS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cCrprwyrUjLiizSS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uoIokWyqashVUPCh(self):
        """MULTIALLEN_uoIokWyqashVUPCh multi-experiment creation."""
        model_folder = 'MULTIALLEN_uoIokWyqashVUPCh'
        exp = {
            'experiment_name': ['MULTIALLEN_uoIokWyqashVUPCh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uoIokWyqashVUPCh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NQslDJXDBPKYHMet(self):
        """MULTIALLEN_NQslDJXDBPKYHMet multi-experiment creation."""
        model_folder = 'MULTIALLEN_NQslDJXDBPKYHMet'
        exp = {
            'experiment_name': ['MULTIALLEN_NQslDJXDBPKYHMet'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NQslDJXDBPKYHMet']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ljnzlLprWUVAMaRt(self):
        """MULTIALLEN_ljnzlLprWUVAMaRt multi-experiment creation."""
        model_folder = 'MULTIALLEN_ljnzlLprWUVAMaRt'
        exp = {
            'experiment_name': ['MULTIALLEN_ljnzlLprWUVAMaRt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ljnzlLprWUVAMaRt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wFjTshCRibiIWEZk(self):
        """MULTIALLEN_wFjTshCRibiIWEZk multi-experiment creation."""
        model_folder = 'MULTIALLEN_wFjTshCRibiIWEZk'
        exp = {
            'experiment_name': ['MULTIALLEN_wFjTshCRibiIWEZk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wFjTshCRibiIWEZk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MUGRWXpOOQGadHTM(self):
        """MULTIALLEN_MUGRWXpOOQGadHTM multi-experiment creation."""
        model_folder = 'MULTIALLEN_MUGRWXpOOQGadHTM'
        exp = {
            'experiment_name': ['MULTIALLEN_MUGRWXpOOQGadHTM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MUGRWXpOOQGadHTM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dYXIcIkcqFrHJWRy(self):
        """MULTIALLEN_dYXIcIkcqFrHJWRy multi-experiment creation."""
        model_folder = 'MULTIALLEN_dYXIcIkcqFrHJWRy'
        exp = {
            'experiment_name': ['MULTIALLEN_dYXIcIkcqFrHJWRy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dYXIcIkcqFrHJWRy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_quIoGfIZYUtHqDZs(self):
        """MULTIALLEN_quIoGfIZYUtHqDZs multi-experiment creation."""
        model_folder = 'MULTIALLEN_quIoGfIZYUtHqDZs'
        exp = {
            'experiment_name': ['MULTIALLEN_quIoGfIZYUtHqDZs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_quIoGfIZYUtHqDZs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JxrQJusYvqwRxofP(self):
        """MULTIALLEN_JxrQJusYvqwRxofP multi-experiment creation."""
        model_folder = 'MULTIALLEN_JxrQJusYvqwRxofP'
        exp = {
            'experiment_name': ['MULTIALLEN_JxrQJusYvqwRxofP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JxrQJusYvqwRxofP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kwSVxodXTqlcwgtv(self):
        """MULTIALLEN_kwSVxodXTqlcwgtv multi-experiment creation."""
        model_folder = 'MULTIALLEN_kwSVxodXTqlcwgtv'
        exp = {
            'experiment_name': ['MULTIALLEN_kwSVxodXTqlcwgtv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kwSVxodXTqlcwgtv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xmBYgTakqkwgdwGu(self):
        """MULTIALLEN_xmBYgTakqkwgdwGu multi-experiment creation."""
        model_folder = 'MULTIALLEN_xmBYgTakqkwgdwGu'
        exp = {
            'experiment_name': ['MULTIALLEN_xmBYgTakqkwgdwGu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xmBYgTakqkwgdwGu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WgjrQEuNcBnwBnOm(self):
        """MULTIALLEN_WgjrQEuNcBnwBnOm multi-experiment creation."""
        model_folder = 'MULTIALLEN_WgjrQEuNcBnwBnOm'
        exp = {
            'experiment_name': ['MULTIALLEN_WgjrQEuNcBnwBnOm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WgjrQEuNcBnwBnOm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XdiSZTKNwouOqcGq(self):
        """MULTIALLEN_XdiSZTKNwouOqcGq multi-experiment creation."""
        model_folder = 'MULTIALLEN_XdiSZTKNwouOqcGq'
        exp = {
            'experiment_name': ['MULTIALLEN_XdiSZTKNwouOqcGq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XdiSZTKNwouOqcGq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SaxhoMUUXflnJguD(self):
        """MULTIALLEN_SaxhoMUUXflnJguD multi-experiment creation."""
        model_folder = 'MULTIALLEN_SaxhoMUUXflnJguD'
        exp = {
            'experiment_name': ['MULTIALLEN_SaxhoMUUXflnJguD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SaxhoMUUXflnJguD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZXhxjMnMaofbvZxD(self):
        """MULTIALLEN_ZXhxjMnMaofbvZxD multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZXhxjMnMaofbvZxD'
        exp = {
            'experiment_name': ['MULTIALLEN_ZXhxjMnMaofbvZxD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZXhxjMnMaofbvZxD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oKhWvxNqjKFkwqKn(self):
        """MULTIALLEN_oKhWvxNqjKFkwqKn multi-experiment creation."""
        model_folder = 'MULTIALLEN_oKhWvxNqjKFkwqKn'
        exp = {
            'experiment_name': ['MULTIALLEN_oKhWvxNqjKFkwqKn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oKhWvxNqjKFkwqKn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wOexTqyoptuasOeK(self):
        """MULTIALLEN_wOexTqyoptuasOeK multi-experiment creation."""
        model_folder = 'MULTIALLEN_wOexTqyoptuasOeK'
        exp = {
            'experiment_name': ['MULTIALLEN_wOexTqyoptuasOeK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wOexTqyoptuasOeK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tmCWmtFnmTprsCbM(self):
        """MULTIALLEN_tmCWmtFnmTprsCbM multi-experiment creation."""
        model_folder = 'MULTIALLEN_tmCWmtFnmTprsCbM'
        exp = {
            'experiment_name': ['MULTIALLEN_tmCWmtFnmTprsCbM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tmCWmtFnmTprsCbM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dRYGBclgGinuFqUw(self):
        """MULTIALLEN_dRYGBclgGinuFqUw multi-experiment creation."""
        model_folder = 'MULTIALLEN_dRYGBclgGinuFqUw'
        exp = {
            'experiment_name': ['MULTIALLEN_dRYGBclgGinuFqUw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dRYGBclgGinuFqUw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XcwpCPzJySfvwlXA(self):
        """MULTIALLEN_XcwpCPzJySfvwlXA multi-experiment creation."""
        model_folder = 'MULTIALLEN_XcwpCPzJySfvwlXA'
        exp = {
            'experiment_name': ['MULTIALLEN_XcwpCPzJySfvwlXA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XcwpCPzJySfvwlXA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_byXlOnXOpFZHUBLI(self):
        """MULTIALLEN_byXlOnXOpFZHUBLI multi-experiment creation."""
        model_folder = 'MULTIALLEN_byXlOnXOpFZHUBLI'
        exp = {
            'experiment_name': ['MULTIALLEN_byXlOnXOpFZHUBLI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_byXlOnXOpFZHUBLI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DuznXgCiBwczcYvR(self):
        """MULTIALLEN_DuznXgCiBwczcYvR multi-experiment creation."""
        model_folder = 'MULTIALLEN_DuznXgCiBwczcYvR'
        exp = {
            'experiment_name': ['MULTIALLEN_DuznXgCiBwczcYvR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DuznXgCiBwczcYvR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dnCQQWJihbJVgGVp(self):
        """MULTIALLEN_dnCQQWJihbJVgGVp multi-experiment creation."""
        model_folder = 'MULTIALLEN_dnCQQWJihbJVgGVp'
        exp = {
            'experiment_name': ['MULTIALLEN_dnCQQWJihbJVgGVp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dnCQQWJihbJVgGVp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CROuVUWrjNKfJbbE(self):
        """MULTIALLEN_CROuVUWrjNKfJbbE multi-experiment creation."""
        model_folder = 'MULTIALLEN_CROuVUWrjNKfJbbE'
        exp = {
            'experiment_name': ['MULTIALLEN_CROuVUWrjNKfJbbE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CROuVUWrjNKfJbbE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZNHLqrfpKgDprVPD(self):
        """MULTIALLEN_ZNHLqrfpKgDprVPD multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZNHLqrfpKgDprVPD'
        exp = {
            'experiment_name': ['MULTIALLEN_ZNHLqrfpKgDprVPD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZNHLqrfpKgDprVPD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UgVnzNJdzyobUyTM(self):
        """MULTIALLEN_UgVnzNJdzyobUyTM multi-experiment creation."""
        model_folder = 'MULTIALLEN_UgVnzNJdzyobUyTM'
        exp = {
            'experiment_name': ['MULTIALLEN_UgVnzNJdzyobUyTM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UgVnzNJdzyobUyTM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_izoPVclAxwFapMJG(self):
        """MULTIALLEN_izoPVclAxwFapMJG multi-experiment creation."""
        model_folder = 'MULTIALLEN_izoPVclAxwFapMJG'
        exp = {
            'experiment_name': ['MULTIALLEN_izoPVclAxwFapMJG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_izoPVclAxwFapMJG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XdNxVhLZnaxlEHoo(self):
        """MULTIALLEN_XdNxVhLZnaxlEHoo multi-experiment creation."""
        model_folder = 'MULTIALLEN_XdNxVhLZnaxlEHoo'
        exp = {
            'experiment_name': ['MULTIALLEN_XdNxVhLZnaxlEHoo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XdNxVhLZnaxlEHoo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vUzIHPaVvhqbcnyk(self):
        """MULTIALLEN_vUzIHPaVvhqbcnyk multi-experiment creation."""
        model_folder = 'MULTIALLEN_vUzIHPaVvhqbcnyk'
        exp = {
            'experiment_name': ['MULTIALLEN_vUzIHPaVvhqbcnyk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vUzIHPaVvhqbcnyk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UtenFSUgaMgitgmn(self):
        """MULTIALLEN_UtenFSUgaMgitgmn multi-experiment creation."""
        model_folder = 'MULTIALLEN_UtenFSUgaMgitgmn'
        exp = {
            'experiment_name': ['MULTIALLEN_UtenFSUgaMgitgmn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UtenFSUgaMgitgmn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aENjNJyZMytzLeRH(self):
        """MULTIALLEN_aENjNJyZMytzLeRH multi-experiment creation."""
        model_folder = 'MULTIALLEN_aENjNJyZMytzLeRH'
        exp = {
            'experiment_name': ['MULTIALLEN_aENjNJyZMytzLeRH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aENjNJyZMytzLeRH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WyznHeZUXhZlkSYk(self):
        """MULTIALLEN_WyznHeZUXhZlkSYk multi-experiment creation."""
        model_folder = 'MULTIALLEN_WyznHeZUXhZlkSYk'
        exp = {
            'experiment_name': ['MULTIALLEN_WyznHeZUXhZlkSYk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WyznHeZUXhZlkSYk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TSsecVjeoBwbehQC(self):
        """MULTIALLEN_TSsecVjeoBwbehQC multi-experiment creation."""
        model_folder = 'MULTIALLEN_TSsecVjeoBwbehQC'
        exp = {
            'experiment_name': ['MULTIALLEN_TSsecVjeoBwbehQC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TSsecVjeoBwbehQC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hppEVzoaVifXabfg(self):
        """MULTIALLEN_hppEVzoaVifXabfg multi-experiment creation."""
        model_folder = 'MULTIALLEN_hppEVzoaVifXabfg'
        exp = {
            'experiment_name': ['MULTIALLEN_hppEVzoaVifXabfg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hppEVzoaVifXabfg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GiVCXSfdWDPjfIXj(self):
        """MULTIALLEN_GiVCXSfdWDPjfIXj multi-experiment creation."""
        model_folder = 'MULTIALLEN_GiVCXSfdWDPjfIXj'
        exp = {
            'experiment_name': ['MULTIALLEN_GiVCXSfdWDPjfIXj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GiVCXSfdWDPjfIXj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UJqdNpUdrQgDiROU(self):
        """MULTIALLEN_UJqdNpUdrQgDiROU multi-experiment creation."""
        model_folder = 'MULTIALLEN_UJqdNpUdrQgDiROU'
        exp = {
            'experiment_name': ['MULTIALLEN_UJqdNpUdrQgDiROU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UJqdNpUdrQgDiROU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ALWrJKPaPxIBFCHz(self):
        """MULTIALLEN_ALWrJKPaPxIBFCHz multi-experiment creation."""
        model_folder = 'MULTIALLEN_ALWrJKPaPxIBFCHz'
        exp = {
            'experiment_name': ['MULTIALLEN_ALWrJKPaPxIBFCHz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ALWrJKPaPxIBFCHz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WinaiiklRyfTKkKg(self):
        """MULTIALLEN_WinaiiklRyfTKkKg multi-experiment creation."""
        model_folder = 'MULTIALLEN_WinaiiklRyfTKkKg'
        exp = {
            'experiment_name': ['MULTIALLEN_WinaiiklRyfTKkKg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WinaiiklRyfTKkKg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qifSvwbgOryXvJNI(self):
        """MULTIALLEN_qifSvwbgOryXvJNI multi-experiment creation."""
        model_folder = 'MULTIALLEN_qifSvwbgOryXvJNI'
        exp = {
            'experiment_name': ['MULTIALLEN_qifSvwbgOryXvJNI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qifSvwbgOryXvJNI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QnMXSpfmpffdImnX(self):
        """MULTIALLEN_QnMXSpfmpffdImnX multi-experiment creation."""
        model_folder = 'MULTIALLEN_QnMXSpfmpffdImnX'
        exp = {
            'experiment_name': ['MULTIALLEN_QnMXSpfmpffdImnX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QnMXSpfmpffdImnX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DXhVBUfMYaztrHrF(self):
        """MULTIALLEN_DXhVBUfMYaztrHrF multi-experiment creation."""
        model_folder = 'MULTIALLEN_DXhVBUfMYaztrHrF'
        exp = {
            'experiment_name': ['MULTIALLEN_DXhVBUfMYaztrHrF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DXhVBUfMYaztrHrF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SzMrIqJIAsuseBsL(self):
        """MULTIALLEN_SzMrIqJIAsuseBsL multi-experiment creation."""
        model_folder = 'MULTIALLEN_SzMrIqJIAsuseBsL'
        exp = {
            'experiment_name': ['MULTIALLEN_SzMrIqJIAsuseBsL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SzMrIqJIAsuseBsL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SysGXPKGREeXVCdD(self):
        """MULTIALLEN_SysGXPKGREeXVCdD multi-experiment creation."""
        model_folder = 'MULTIALLEN_SysGXPKGREeXVCdD'
        exp = {
            'experiment_name': ['MULTIALLEN_SysGXPKGREeXVCdD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SysGXPKGREeXVCdD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qOqfqZeoEHprqyNb(self):
        """MULTIALLEN_qOqfqZeoEHprqyNb multi-experiment creation."""
        model_folder = 'MULTIALLEN_qOqfqZeoEHprqyNb'
        exp = {
            'experiment_name': ['MULTIALLEN_qOqfqZeoEHprqyNb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qOqfqZeoEHprqyNb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zinMkdEYVpwzHtqM(self):
        """MULTIALLEN_zinMkdEYVpwzHtqM multi-experiment creation."""
        model_folder = 'MULTIALLEN_zinMkdEYVpwzHtqM'
        exp = {
            'experiment_name': ['MULTIALLEN_zinMkdEYVpwzHtqM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zinMkdEYVpwzHtqM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nnoaMehyWAEqdTxi(self):
        """MULTIALLEN_nnoaMehyWAEqdTxi multi-experiment creation."""
        model_folder = 'MULTIALLEN_nnoaMehyWAEqdTxi'
        exp = {
            'experiment_name': ['MULTIALLEN_nnoaMehyWAEqdTxi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nnoaMehyWAEqdTxi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UnaVaCnpGXCvYLAn(self):
        """MULTIALLEN_UnaVaCnpGXCvYLAn multi-experiment creation."""
        model_folder = 'MULTIALLEN_UnaVaCnpGXCvYLAn'
        exp = {
            'experiment_name': ['MULTIALLEN_UnaVaCnpGXCvYLAn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UnaVaCnpGXCvYLAn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WSXItcUVZjxFbFcY(self):
        """MULTIALLEN_WSXItcUVZjxFbFcY multi-experiment creation."""
        model_folder = 'MULTIALLEN_WSXItcUVZjxFbFcY'
        exp = {
            'experiment_name': ['MULTIALLEN_WSXItcUVZjxFbFcY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WSXItcUVZjxFbFcY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gejbLQrAZvYWVITy(self):
        """MULTIALLEN_gejbLQrAZvYWVITy multi-experiment creation."""
        model_folder = 'MULTIALLEN_gejbLQrAZvYWVITy'
        exp = {
            'experiment_name': ['MULTIALLEN_gejbLQrAZvYWVITy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gejbLQrAZvYWVITy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SwenslRPEVsFZtJB(self):
        """MULTIALLEN_SwenslRPEVsFZtJB multi-experiment creation."""
        model_folder = 'MULTIALLEN_SwenslRPEVsFZtJB'
        exp = {
            'experiment_name': ['MULTIALLEN_SwenslRPEVsFZtJB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SwenslRPEVsFZtJB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HnutGhQmcFLhYOkz(self):
        """MULTIALLEN_HnutGhQmcFLhYOkz multi-experiment creation."""
        model_folder = 'MULTIALLEN_HnutGhQmcFLhYOkz'
        exp = {
            'experiment_name': ['MULTIALLEN_HnutGhQmcFLhYOkz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HnutGhQmcFLhYOkz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tSVNUDEqVWNiQRWv(self):
        """MULTIALLEN_tSVNUDEqVWNiQRWv multi-experiment creation."""
        model_folder = 'MULTIALLEN_tSVNUDEqVWNiQRWv'
        exp = {
            'experiment_name': ['MULTIALLEN_tSVNUDEqVWNiQRWv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tSVNUDEqVWNiQRWv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mBxazoOsaruHSiHt(self):
        """MULTIALLEN_mBxazoOsaruHSiHt multi-experiment creation."""
        model_folder = 'MULTIALLEN_mBxazoOsaruHSiHt'
        exp = {
            'experiment_name': ['MULTIALLEN_mBxazoOsaruHSiHt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mBxazoOsaruHSiHt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uGSbRZlUoGWPykbV(self):
        """MULTIALLEN_uGSbRZlUoGWPykbV multi-experiment creation."""
        model_folder = 'MULTIALLEN_uGSbRZlUoGWPykbV'
        exp = {
            'experiment_name': ['MULTIALLEN_uGSbRZlUoGWPykbV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uGSbRZlUoGWPykbV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_daWVwPwbdvPuruUl(self):
        """MULTIALLEN_daWVwPwbdvPuruUl multi-experiment creation."""
        model_folder = 'MULTIALLEN_daWVwPwbdvPuruUl'
        exp = {
            'experiment_name': ['MULTIALLEN_daWVwPwbdvPuruUl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_daWVwPwbdvPuruUl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QZBffRMIxawoyLtc(self):
        """MULTIALLEN_QZBffRMIxawoyLtc multi-experiment creation."""
        model_folder = 'MULTIALLEN_QZBffRMIxawoyLtc'
        exp = {
            'experiment_name': ['MULTIALLEN_QZBffRMIxawoyLtc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QZBffRMIxawoyLtc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rJpfGypoXhdjlZTk(self):
        """MULTIALLEN_rJpfGypoXhdjlZTk multi-experiment creation."""
        model_folder = 'MULTIALLEN_rJpfGypoXhdjlZTk'
        exp = {
            'experiment_name': ['MULTIALLEN_rJpfGypoXhdjlZTk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rJpfGypoXhdjlZTk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tKUTAwyGtYoqfshl(self):
        """MULTIALLEN_tKUTAwyGtYoqfshl multi-experiment creation."""
        model_folder = 'MULTIALLEN_tKUTAwyGtYoqfshl'
        exp = {
            'experiment_name': ['MULTIALLEN_tKUTAwyGtYoqfshl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tKUTAwyGtYoqfshl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WBHGpMxIXYofdrTw(self):
        """MULTIALLEN_WBHGpMxIXYofdrTw multi-experiment creation."""
        model_folder = 'MULTIALLEN_WBHGpMxIXYofdrTw'
        exp = {
            'experiment_name': ['MULTIALLEN_WBHGpMxIXYofdrTw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WBHGpMxIXYofdrTw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PXyzMucBQsxECnYF(self):
        """MULTIALLEN_PXyzMucBQsxECnYF multi-experiment creation."""
        model_folder = 'MULTIALLEN_PXyzMucBQsxECnYF'
        exp = {
            'experiment_name': ['MULTIALLEN_PXyzMucBQsxECnYF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PXyzMucBQsxECnYF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_whHHwEQxAdwmcXhU(self):
        """MULTIALLEN_whHHwEQxAdwmcXhU multi-experiment creation."""
        model_folder = 'MULTIALLEN_whHHwEQxAdwmcXhU'
        exp = {
            'experiment_name': ['MULTIALLEN_whHHwEQxAdwmcXhU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_whHHwEQxAdwmcXhU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QjzVnRfRoKNsZWUV(self):
        """MULTIALLEN_QjzVnRfRoKNsZWUV multi-experiment creation."""
        model_folder = 'MULTIALLEN_QjzVnRfRoKNsZWUV'
        exp = {
            'experiment_name': ['MULTIALLEN_QjzVnRfRoKNsZWUV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QjzVnRfRoKNsZWUV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VBFgToXKYroBbuxB(self):
        """MULTIALLEN_VBFgToXKYroBbuxB multi-experiment creation."""
        model_folder = 'MULTIALLEN_VBFgToXKYroBbuxB'
        exp = {
            'experiment_name': ['MULTIALLEN_VBFgToXKYroBbuxB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VBFgToXKYroBbuxB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aJYqtBJprHxwOZBl(self):
        """MULTIALLEN_aJYqtBJprHxwOZBl multi-experiment creation."""
        model_folder = 'MULTIALLEN_aJYqtBJprHxwOZBl'
        exp = {
            'experiment_name': ['MULTIALLEN_aJYqtBJprHxwOZBl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aJYqtBJprHxwOZBl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oEtuQDLhuqfpRbND(self):
        """MULTIALLEN_oEtuQDLhuqfpRbND multi-experiment creation."""
        model_folder = 'MULTIALLEN_oEtuQDLhuqfpRbND'
        exp = {
            'experiment_name': ['MULTIALLEN_oEtuQDLhuqfpRbND'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oEtuQDLhuqfpRbND']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mRdyZAJegQDrjEWo(self):
        """MULTIALLEN_mRdyZAJegQDrjEWo multi-experiment creation."""
        model_folder = 'MULTIALLEN_mRdyZAJegQDrjEWo'
        exp = {
            'experiment_name': ['MULTIALLEN_mRdyZAJegQDrjEWo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mRdyZAJegQDrjEWo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mbVtWiXoicFqETfb(self):
        """MULTIALLEN_mbVtWiXoicFqETfb multi-experiment creation."""
        model_folder = 'MULTIALLEN_mbVtWiXoicFqETfb'
        exp = {
            'experiment_name': ['MULTIALLEN_mbVtWiXoicFqETfb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mbVtWiXoicFqETfb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xQVXVfmTNlSoARSb(self):
        """MULTIALLEN_xQVXVfmTNlSoARSb multi-experiment creation."""
        model_folder = 'MULTIALLEN_xQVXVfmTNlSoARSb'
        exp = {
            'experiment_name': ['MULTIALLEN_xQVXVfmTNlSoARSb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xQVXVfmTNlSoARSb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BOkdtOgCvPonGmEG(self):
        """MULTIALLEN_BOkdtOgCvPonGmEG multi-experiment creation."""
        model_folder = 'MULTIALLEN_BOkdtOgCvPonGmEG'
        exp = {
            'experiment_name': ['MULTIALLEN_BOkdtOgCvPonGmEG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BOkdtOgCvPonGmEG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hBZypvudRfOmaqye(self):
        """MULTIALLEN_hBZypvudRfOmaqye multi-experiment creation."""
        model_folder = 'MULTIALLEN_hBZypvudRfOmaqye'
        exp = {
            'experiment_name': ['MULTIALLEN_hBZypvudRfOmaqye'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hBZypvudRfOmaqye']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JfKEUDYKJQQtlnEg(self):
        """MULTIALLEN_JfKEUDYKJQQtlnEg multi-experiment creation."""
        model_folder = 'MULTIALLEN_JfKEUDYKJQQtlnEg'
        exp = {
            'experiment_name': ['MULTIALLEN_JfKEUDYKJQQtlnEg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JfKEUDYKJQQtlnEg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MBPtAaQbJSzYMeGE(self):
        """MULTIALLEN_MBPtAaQbJSzYMeGE multi-experiment creation."""
        model_folder = 'MULTIALLEN_MBPtAaQbJSzYMeGE'
        exp = {
            'experiment_name': ['MULTIALLEN_MBPtAaQbJSzYMeGE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MBPtAaQbJSzYMeGE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MTWYcxVwahvdsaIT(self):
        """MULTIALLEN_MTWYcxVwahvdsaIT multi-experiment creation."""
        model_folder = 'MULTIALLEN_MTWYcxVwahvdsaIT'
        exp = {
            'experiment_name': ['MULTIALLEN_MTWYcxVwahvdsaIT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MTWYcxVwahvdsaIT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LILvdLUeaCvcyYXz(self):
        """MULTIALLEN_LILvdLUeaCvcyYXz multi-experiment creation."""
        model_folder = 'MULTIALLEN_LILvdLUeaCvcyYXz'
        exp = {
            'experiment_name': ['MULTIALLEN_LILvdLUeaCvcyYXz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LILvdLUeaCvcyYXz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JBYEwCsquJsndXgP(self):
        """MULTIALLEN_JBYEwCsquJsndXgP multi-experiment creation."""
        model_folder = 'MULTIALLEN_JBYEwCsquJsndXgP'
        exp = {
            'experiment_name': ['MULTIALLEN_JBYEwCsquJsndXgP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JBYEwCsquJsndXgP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NMwYudSMLyioYyEu(self):
        """MULTIALLEN_NMwYudSMLyioYyEu multi-experiment creation."""
        model_folder = 'MULTIALLEN_NMwYudSMLyioYyEu'
        exp = {
            'experiment_name': ['MULTIALLEN_NMwYudSMLyioYyEu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NMwYudSMLyioYyEu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zcesayennCQCleVF(self):
        """MULTIALLEN_zcesayennCQCleVF multi-experiment creation."""
        model_folder = 'MULTIALLEN_zcesayennCQCleVF'
        exp = {
            'experiment_name': ['MULTIALLEN_zcesayennCQCleVF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zcesayennCQCleVF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ezPqHiDrTMYokGSM(self):
        """MULTIALLEN_ezPqHiDrTMYokGSM multi-experiment creation."""
        model_folder = 'MULTIALLEN_ezPqHiDrTMYokGSM'
        exp = {
            'experiment_name': ['MULTIALLEN_ezPqHiDrTMYokGSM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ezPqHiDrTMYokGSM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hMGvShNzsUAnjRxR(self):
        """MULTIALLEN_hMGvShNzsUAnjRxR multi-experiment creation."""
        model_folder = 'MULTIALLEN_hMGvShNzsUAnjRxR'
        exp = {
            'experiment_name': ['MULTIALLEN_hMGvShNzsUAnjRxR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hMGvShNzsUAnjRxR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DDXExvAGaMfkDhSl(self):
        """MULTIALLEN_DDXExvAGaMfkDhSl multi-experiment creation."""
        model_folder = 'MULTIALLEN_DDXExvAGaMfkDhSl'
        exp = {
            'experiment_name': ['MULTIALLEN_DDXExvAGaMfkDhSl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DDXExvAGaMfkDhSl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sJERYoHADrPqwRPt(self):
        """MULTIALLEN_sJERYoHADrPqwRPt multi-experiment creation."""
        model_folder = 'MULTIALLEN_sJERYoHADrPqwRPt'
        exp = {
            'experiment_name': ['MULTIALLEN_sJERYoHADrPqwRPt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sJERYoHADrPqwRPt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uDNxCSTFauELfFLH(self):
        """MULTIALLEN_uDNxCSTFauELfFLH multi-experiment creation."""
        model_folder = 'MULTIALLEN_uDNxCSTFauELfFLH'
        exp = {
            'experiment_name': ['MULTIALLEN_uDNxCSTFauELfFLH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uDNxCSTFauELfFLH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_INqyQCeMwokspcuX(self):
        """MULTIALLEN_INqyQCeMwokspcuX multi-experiment creation."""
        model_folder = 'MULTIALLEN_INqyQCeMwokspcuX'
        exp = {
            'experiment_name': ['MULTIALLEN_INqyQCeMwokspcuX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_INqyQCeMwokspcuX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CpnnUMkdWVEEOENV(self):
        """MULTIALLEN_CpnnUMkdWVEEOENV multi-experiment creation."""
        model_folder = 'MULTIALLEN_CpnnUMkdWVEEOENV'
        exp = {
            'experiment_name': ['MULTIALLEN_CpnnUMkdWVEEOENV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CpnnUMkdWVEEOENV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qmfyLldynEbMbzAF(self):
        """MULTIALLEN_qmfyLldynEbMbzAF multi-experiment creation."""
        model_folder = 'MULTIALLEN_qmfyLldynEbMbzAF'
        exp = {
            'experiment_name': ['MULTIALLEN_qmfyLldynEbMbzAF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qmfyLldynEbMbzAF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iXMixFgnjGOBgPrs(self):
        """MULTIALLEN_iXMixFgnjGOBgPrs multi-experiment creation."""
        model_folder = 'MULTIALLEN_iXMixFgnjGOBgPrs'
        exp = {
            'experiment_name': ['MULTIALLEN_iXMixFgnjGOBgPrs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iXMixFgnjGOBgPrs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KCMFSFEhhdoHYTOp(self):
        """MULTIALLEN_KCMFSFEhhdoHYTOp multi-experiment creation."""
        model_folder = 'MULTIALLEN_KCMFSFEhhdoHYTOp'
        exp = {
            'experiment_name': ['MULTIALLEN_KCMFSFEhhdoHYTOp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KCMFSFEhhdoHYTOp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_imbYPSlGjXgfLlqc(self):
        """MULTIALLEN_imbYPSlGjXgfLlqc multi-experiment creation."""
        model_folder = 'MULTIALLEN_imbYPSlGjXgfLlqc'
        exp = {
            'experiment_name': ['MULTIALLEN_imbYPSlGjXgfLlqc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_imbYPSlGjXgfLlqc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GSOUfmneFzyxMBYi(self):
        """MULTIALLEN_GSOUfmneFzyxMBYi multi-experiment creation."""
        model_folder = 'MULTIALLEN_GSOUfmneFzyxMBYi'
        exp = {
            'experiment_name': ['MULTIALLEN_GSOUfmneFzyxMBYi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GSOUfmneFzyxMBYi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EgVlbXkxdmLpFkWp(self):
        """MULTIALLEN_EgVlbXkxdmLpFkWp multi-experiment creation."""
        model_folder = 'MULTIALLEN_EgVlbXkxdmLpFkWp'
        exp = {
            'experiment_name': ['MULTIALLEN_EgVlbXkxdmLpFkWp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EgVlbXkxdmLpFkWp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tbDBHFZuSRhzKBmZ(self):
        """MULTIALLEN_tbDBHFZuSRhzKBmZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_tbDBHFZuSRhzKBmZ'
        exp = {
            'experiment_name': ['MULTIALLEN_tbDBHFZuSRhzKBmZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tbDBHFZuSRhzKBmZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nfhGFFEgcpSepTWm(self):
        """MULTIALLEN_nfhGFFEgcpSepTWm multi-experiment creation."""
        model_folder = 'MULTIALLEN_nfhGFFEgcpSepTWm'
        exp = {
            'experiment_name': ['MULTIALLEN_nfhGFFEgcpSepTWm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nfhGFFEgcpSepTWm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YUakKAHWDTpuPwkF(self):
        """MULTIALLEN_YUakKAHWDTpuPwkF multi-experiment creation."""
        model_folder = 'MULTIALLEN_YUakKAHWDTpuPwkF'
        exp = {
            'experiment_name': ['MULTIALLEN_YUakKAHWDTpuPwkF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YUakKAHWDTpuPwkF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sZtxQieIwvlCVqsK(self):
        """MULTIALLEN_sZtxQieIwvlCVqsK multi-experiment creation."""
        model_folder = 'MULTIALLEN_sZtxQieIwvlCVqsK'
        exp = {
            'experiment_name': ['MULTIALLEN_sZtxQieIwvlCVqsK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sZtxQieIwvlCVqsK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uNoFYUKlhzbaAzot(self):
        """MULTIALLEN_uNoFYUKlhzbaAzot multi-experiment creation."""
        model_folder = 'MULTIALLEN_uNoFYUKlhzbaAzot'
        exp = {
            'experiment_name': ['MULTIALLEN_uNoFYUKlhzbaAzot'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uNoFYUKlhzbaAzot']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xRQDCrulaueulaCU(self):
        """MULTIALLEN_xRQDCrulaueulaCU multi-experiment creation."""
        model_folder = 'MULTIALLEN_xRQDCrulaueulaCU'
        exp = {
            'experiment_name': ['MULTIALLEN_xRQDCrulaueulaCU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xRQDCrulaueulaCU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RdZdFIeIpXNNYQqo(self):
        """MULTIALLEN_RdZdFIeIpXNNYQqo multi-experiment creation."""
        model_folder = 'MULTIALLEN_RdZdFIeIpXNNYQqo'
        exp = {
            'experiment_name': ['MULTIALLEN_RdZdFIeIpXNNYQqo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RdZdFIeIpXNNYQqo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sKpLRqiUXrbbzBMT(self):
        """MULTIALLEN_sKpLRqiUXrbbzBMT multi-experiment creation."""
        model_folder = 'MULTIALLEN_sKpLRqiUXrbbzBMT'
        exp = {
            'experiment_name': ['MULTIALLEN_sKpLRqiUXrbbzBMT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sKpLRqiUXrbbzBMT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UMADswnOHxAgoorz(self):
        """MULTIALLEN_UMADswnOHxAgoorz multi-experiment creation."""
        model_folder = 'MULTIALLEN_UMADswnOHxAgoorz'
        exp = {
            'experiment_name': ['MULTIALLEN_UMADswnOHxAgoorz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UMADswnOHxAgoorz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZawqLIibChzTdhao(self):
        """MULTIALLEN_ZawqLIibChzTdhao multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZawqLIibChzTdhao'
        exp = {
            'experiment_name': ['MULTIALLEN_ZawqLIibChzTdhao'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZawqLIibChzTdhao']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ghoMXKfjXvDLnqzY(self):
        """MULTIALLEN_ghoMXKfjXvDLnqzY multi-experiment creation."""
        model_folder = 'MULTIALLEN_ghoMXKfjXvDLnqzY'
        exp = {
            'experiment_name': ['MULTIALLEN_ghoMXKfjXvDLnqzY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ghoMXKfjXvDLnqzY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AJJTlDfRoISYHQop(self):
        """MULTIALLEN_AJJTlDfRoISYHQop multi-experiment creation."""
        model_folder = 'MULTIALLEN_AJJTlDfRoISYHQop'
        exp = {
            'experiment_name': ['MULTIALLEN_AJJTlDfRoISYHQop'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AJJTlDfRoISYHQop']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ThgcQkAGvGEvqNYe(self):
        """MULTIALLEN_ThgcQkAGvGEvqNYe multi-experiment creation."""
        model_folder = 'MULTIALLEN_ThgcQkAGvGEvqNYe'
        exp = {
            'experiment_name': ['MULTIALLEN_ThgcQkAGvGEvqNYe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ThgcQkAGvGEvqNYe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jJYNyQmaBNFMoVuK(self):
        """MULTIALLEN_jJYNyQmaBNFMoVuK multi-experiment creation."""
        model_folder = 'MULTIALLEN_jJYNyQmaBNFMoVuK'
        exp = {
            'experiment_name': ['MULTIALLEN_jJYNyQmaBNFMoVuK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jJYNyQmaBNFMoVuK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PQakbsCZzZFGxZjC(self):
        """MULTIALLEN_PQakbsCZzZFGxZjC multi-experiment creation."""
        model_folder = 'MULTIALLEN_PQakbsCZzZFGxZjC'
        exp = {
            'experiment_name': ['MULTIALLEN_PQakbsCZzZFGxZjC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PQakbsCZzZFGxZjC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pyiBfXDgxzgxMbkx(self):
        """MULTIALLEN_pyiBfXDgxzgxMbkx multi-experiment creation."""
        model_folder = 'MULTIALLEN_pyiBfXDgxzgxMbkx'
        exp = {
            'experiment_name': ['MULTIALLEN_pyiBfXDgxzgxMbkx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pyiBfXDgxzgxMbkx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vTsYxmtYXtmVbJQw(self):
        """MULTIALLEN_vTsYxmtYXtmVbJQw multi-experiment creation."""
        model_folder = 'MULTIALLEN_vTsYxmtYXtmVbJQw'
        exp = {
            'experiment_name': ['MULTIALLEN_vTsYxmtYXtmVbJQw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vTsYxmtYXtmVbJQw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sLlcsggkRarZDZQi(self):
        """MULTIALLEN_sLlcsggkRarZDZQi multi-experiment creation."""
        model_folder = 'MULTIALLEN_sLlcsggkRarZDZQi'
        exp = {
            'experiment_name': ['MULTIALLEN_sLlcsggkRarZDZQi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sLlcsggkRarZDZQi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EvMVJDgTaWYdIBVE(self):
        """MULTIALLEN_EvMVJDgTaWYdIBVE multi-experiment creation."""
        model_folder = 'MULTIALLEN_EvMVJDgTaWYdIBVE'
        exp = {
            'experiment_name': ['MULTIALLEN_EvMVJDgTaWYdIBVE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EvMVJDgTaWYdIBVE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YBCvTCohaUKybdcp(self):
        """MULTIALLEN_YBCvTCohaUKybdcp multi-experiment creation."""
        model_folder = 'MULTIALLEN_YBCvTCohaUKybdcp'
        exp = {
            'experiment_name': ['MULTIALLEN_YBCvTCohaUKybdcp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YBCvTCohaUKybdcp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bJAEPPESVIlopvgH(self):
        """MULTIALLEN_bJAEPPESVIlopvgH multi-experiment creation."""
        model_folder = 'MULTIALLEN_bJAEPPESVIlopvgH'
        exp = {
            'experiment_name': ['MULTIALLEN_bJAEPPESVIlopvgH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bJAEPPESVIlopvgH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DhYjstKdzqjIzZRa(self):
        """MULTIALLEN_DhYjstKdzqjIzZRa multi-experiment creation."""
        model_folder = 'MULTIALLEN_DhYjstKdzqjIzZRa'
        exp = {
            'experiment_name': ['MULTIALLEN_DhYjstKdzqjIzZRa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DhYjstKdzqjIzZRa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xLFMUVTJMyzHtKvv(self):
        """MULTIALLEN_xLFMUVTJMyzHtKvv multi-experiment creation."""
        model_folder = 'MULTIALLEN_xLFMUVTJMyzHtKvv'
        exp = {
            'experiment_name': ['MULTIALLEN_xLFMUVTJMyzHtKvv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xLFMUVTJMyzHtKvv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pirWGVaMoIhGqJWu(self):
        """MULTIALLEN_pirWGVaMoIhGqJWu multi-experiment creation."""
        model_folder = 'MULTIALLEN_pirWGVaMoIhGqJWu'
        exp = {
            'experiment_name': ['MULTIALLEN_pirWGVaMoIhGqJWu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pirWGVaMoIhGqJWu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xwAgpyTfLMvtUUWA(self):
        """MULTIALLEN_xwAgpyTfLMvtUUWA multi-experiment creation."""
        model_folder = 'MULTIALLEN_xwAgpyTfLMvtUUWA'
        exp = {
            'experiment_name': ['MULTIALLEN_xwAgpyTfLMvtUUWA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xwAgpyTfLMvtUUWA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dHUEbwLHmiElfWXR(self):
        """MULTIALLEN_dHUEbwLHmiElfWXR multi-experiment creation."""
        model_folder = 'MULTIALLEN_dHUEbwLHmiElfWXR'
        exp = {
            'experiment_name': ['MULTIALLEN_dHUEbwLHmiElfWXR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dHUEbwLHmiElfWXR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dYcjroPamntVtvoP(self):
        """MULTIALLEN_dYcjroPamntVtvoP multi-experiment creation."""
        model_folder = 'MULTIALLEN_dYcjroPamntVtvoP'
        exp = {
            'experiment_name': ['MULTIALLEN_dYcjroPamntVtvoP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dYcjroPamntVtvoP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gtfNjkLrcSGnwhic(self):
        """MULTIALLEN_gtfNjkLrcSGnwhic multi-experiment creation."""
        model_folder = 'MULTIALLEN_gtfNjkLrcSGnwhic'
        exp = {
            'experiment_name': ['MULTIALLEN_gtfNjkLrcSGnwhic'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gtfNjkLrcSGnwhic']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GIFtTDLFgPqjDLjR(self):
        """MULTIALLEN_GIFtTDLFgPqjDLjR multi-experiment creation."""
        model_folder = 'MULTIALLEN_GIFtTDLFgPqjDLjR'
        exp = {
            'experiment_name': ['MULTIALLEN_GIFtTDLFgPqjDLjR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GIFtTDLFgPqjDLjR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tkTAZZmUEXlUPVOU(self):
        """MULTIALLEN_tkTAZZmUEXlUPVOU multi-experiment creation."""
        model_folder = 'MULTIALLEN_tkTAZZmUEXlUPVOU'
        exp = {
            'experiment_name': ['MULTIALLEN_tkTAZZmUEXlUPVOU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tkTAZZmUEXlUPVOU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xGRDxlcTsZpQQRHs(self):
        """MULTIALLEN_xGRDxlcTsZpQQRHs multi-experiment creation."""
        model_folder = 'MULTIALLEN_xGRDxlcTsZpQQRHs'
        exp = {
            'experiment_name': ['MULTIALLEN_xGRDxlcTsZpQQRHs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xGRDxlcTsZpQQRHs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RiQyzXBeoBNIabpn(self):
        """MULTIALLEN_RiQyzXBeoBNIabpn multi-experiment creation."""
        model_folder = 'MULTIALLEN_RiQyzXBeoBNIabpn'
        exp = {
            'experiment_name': ['MULTIALLEN_RiQyzXBeoBNIabpn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RiQyzXBeoBNIabpn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hiCzqmOOpEGvMWeQ(self):
        """MULTIALLEN_hiCzqmOOpEGvMWeQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_hiCzqmOOpEGvMWeQ'
        exp = {
            'experiment_name': ['MULTIALLEN_hiCzqmOOpEGvMWeQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hiCzqmOOpEGvMWeQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UAEBzsXPFQNARTjd(self):
        """MULTIALLEN_UAEBzsXPFQNARTjd multi-experiment creation."""
        model_folder = 'MULTIALLEN_UAEBzsXPFQNARTjd'
        exp = {
            'experiment_name': ['MULTIALLEN_UAEBzsXPFQNARTjd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UAEBzsXPFQNARTjd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_piDZPOlYxhCfyQZE(self):
        """MULTIALLEN_piDZPOlYxhCfyQZE multi-experiment creation."""
        model_folder = 'MULTIALLEN_piDZPOlYxhCfyQZE'
        exp = {
            'experiment_name': ['MULTIALLEN_piDZPOlYxhCfyQZE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_piDZPOlYxhCfyQZE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fSNDeuDZoQhgPyKo(self):
        """MULTIALLEN_fSNDeuDZoQhgPyKo multi-experiment creation."""
        model_folder = 'MULTIALLEN_fSNDeuDZoQhgPyKo'
        exp = {
            'experiment_name': ['MULTIALLEN_fSNDeuDZoQhgPyKo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fSNDeuDZoQhgPyKo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ejJabirHJkCpNaMT(self):
        """MULTIALLEN_ejJabirHJkCpNaMT multi-experiment creation."""
        model_folder = 'MULTIALLEN_ejJabirHJkCpNaMT'
        exp = {
            'experiment_name': ['MULTIALLEN_ejJabirHJkCpNaMT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ejJabirHJkCpNaMT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MXOUkUMUkcxrYgXW(self):
        """MULTIALLEN_MXOUkUMUkcxrYgXW multi-experiment creation."""
        model_folder = 'MULTIALLEN_MXOUkUMUkcxrYgXW'
        exp = {
            'experiment_name': ['MULTIALLEN_MXOUkUMUkcxrYgXW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MXOUkUMUkcxrYgXW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ggGhlbIOmcdoMCLI(self):
        """MULTIALLEN_ggGhlbIOmcdoMCLI multi-experiment creation."""
        model_folder = 'MULTIALLEN_ggGhlbIOmcdoMCLI'
        exp = {
            'experiment_name': ['MULTIALLEN_ggGhlbIOmcdoMCLI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ggGhlbIOmcdoMCLI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VIZKgBHSCzLvGSvI(self):
        """MULTIALLEN_VIZKgBHSCzLvGSvI multi-experiment creation."""
        model_folder = 'MULTIALLEN_VIZKgBHSCzLvGSvI'
        exp = {
            'experiment_name': ['MULTIALLEN_VIZKgBHSCzLvGSvI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VIZKgBHSCzLvGSvI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LEXmswypqEgVsUSL(self):
        """MULTIALLEN_LEXmswypqEgVsUSL multi-experiment creation."""
        model_folder = 'MULTIALLEN_LEXmswypqEgVsUSL'
        exp = {
            'experiment_name': ['MULTIALLEN_LEXmswypqEgVsUSL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LEXmswypqEgVsUSL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_phBsQOwshnVIrOcn(self):
        """MULTIALLEN_phBsQOwshnVIrOcn multi-experiment creation."""
        model_folder = 'MULTIALLEN_phBsQOwshnVIrOcn'
        exp = {
            'experiment_name': ['MULTIALLEN_phBsQOwshnVIrOcn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_phBsQOwshnVIrOcn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gyfXnYWnXsqETVBk(self):
        """MULTIALLEN_gyfXnYWnXsqETVBk multi-experiment creation."""
        model_folder = 'MULTIALLEN_gyfXnYWnXsqETVBk'
        exp = {
            'experiment_name': ['MULTIALLEN_gyfXnYWnXsqETVBk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gyfXnYWnXsqETVBk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HntZKQUdDdtXODvT(self):
        """MULTIALLEN_HntZKQUdDdtXODvT multi-experiment creation."""
        model_folder = 'MULTIALLEN_HntZKQUdDdtXODvT'
        exp = {
            'experiment_name': ['MULTIALLEN_HntZKQUdDdtXODvT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HntZKQUdDdtXODvT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DspQXwgBLzvNSLiV(self):
        """MULTIALLEN_DspQXwgBLzvNSLiV multi-experiment creation."""
        model_folder = 'MULTIALLEN_DspQXwgBLzvNSLiV'
        exp = {
            'experiment_name': ['MULTIALLEN_DspQXwgBLzvNSLiV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DspQXwgBLzvNSLiV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zwTkSCBlWrtFDxdl(self):
        """MULTIALLEN_zwTkSCBlWrtFDxdl multi-experiment creation."""
        model_folder = 'MULTIALLEN_zwTkSCBlWrtFDxdl'
        exp = {
            'experiment_name': ['MULTIALLEN_zwTkSCBlWrtFDxdl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zwTkSCBlWrtFDxdl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UvxYMZQvbZseGAhG(self):
        """MULTIALLEN_UvxYMZQvbZseGAhG multi-experiment creation."""
        model_folder = 'MULTIALLEN_UvxYMZQvbZseGAhG'
        exp = {
            'experiment_name': ['MULTIALLEN_UvxYMZQvbZseGAhG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UvxYMZQvbZseGAhG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nqUTzcYLGnstoezu(self):
        """MULTIALLEN_nqUTzcYLGnstoezu multi-experiment creation."""
        model_folder = 'MULTIALLEN_nqUTzcYLGnstoezu'
        exp = {
            'experiment_name': ['MULTIALLEN_nqUTzcYLGnstoezu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nqUTzcYLGnstoezu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HJvXplgMuYnpmJdI(self):
        """MULTIALLEN_HJvXplgMuYnpmJdI multi-experiment creation."""
        model_folder = 'MULTIALLEN_HJvXplgMuYnpmJdI'
        exp = {
            'experiment_name': ['MULTIALLEN_HJvXplgMuYnpmJdI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HJvXplgMuYnpmJdI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uyGbfuqbmEkJswBU(self):
        """MULTIALLEN_uyGbfuqbmEkJswBU multi-experiment creation."""
        model_folder = 'MULTIALLEN_uyGbfuqbmEkJswBU'
        exp = {
            'experiment_name': ['MULTIALLEN_uyGbfuqbmEkJswBU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uyGbfuqbmEkJswBU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pprvAvKHSGypYZHB(self):
        """MULTIALLEN_pprvAvKHSGypYZHB multi-experiment creation."""
        model_folder = 'MULTIALLEN_pprvAvKHSGypYZHB'
        exp = {
            'experiment_name': ['MULTIALLEN_pprvAvKHSGypYZHB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pprvAvKHSGypYZHB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OXXETfqhRALKMnLc(self):
        """MULTIALLEN_OXXETfqhRALKMnLc multi-experiment creation."""
        model_folder = 'MULTIALLEN_OXXETfqhRALKMnLc'
        exp = {
            'experiment_name': ['MULTIALLEN_OXXETfqhRALKMnLc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OXXETfqhRALKMnLc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fsBhTXAHBrYfgUnR(self):
        """MULTIALLEN_fsBhTXAHBrYfgUnR multi-experiment creation."""
        model_folder = 'MULTIALLEN_fsBhTXAHBrYfgUnR'
        exp = {
            'experiment_name': ['MULTIALLEN_fsBhTXAHBrYfgUnR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fsBhTXAHBrYfgUnR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LwqszLvzGHYlqwqZ(self):
        """MULTIALLEN_LwqszLvzGHYlqwqZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_LwqszLvzGHYlqwqZ'
        exp = {
            'experiment_name': ['MULTIALLEN_LwqszLvzGHYlqwqZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LwqszLvzGHYlqwqZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qlAHvJyFXruJUXbc(self):
        """MULTIALLEN_qlAHvJyFXruJUXbc multi-experiment creation."""
        model_folder = 'MULTIALLEN_qlAHvJyFXruJUXbc'
        exp = {
            'experiment_name': ['MULTIALLEN_qlAHvJyFXruJUXbc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qlAHvJyFXruJUXbc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WACMwGuQsqEjfXaI(self):
        """MULTIALLEN_WACMwGuQsqEjfXaI multi-experiment creation."""
        model_folder = 'MULTIALLEN_WACMwGuQsqEjfXaI'
        exp = {
            'experiment_name': ['MULTIALLEN_WACMwGuQsqEjfXaI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WACMwGuQsqEjfXaI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IZHquxigayVXUscR(self):
        """MULTIALLEN_IZHquxigayVXUscR multi-experiment creation."""
        model_folder = 'MULTIALLEN_IZHquxigayVXUscR'
        exp = {
            'experiment_name': ['MULTIALLEN_IZHquxigayVXUscR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IZHquxigayVXUscR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DwYevHXMYqTUNQwk(self):
        """MULTIALLEN_DwYevHXMYqTUNQwk multi-experiment creation."""
        model_folder = 'MULTIALLEN_DwYevHXMYqTUNQwk'
        exp = {
            'experiment_name': ['MULTIALLEN_DwYevHXMYqTUNQwk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DwYevHXMYqTUNQwk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DhFtwLuVBYYZVgpE(self):
        """MULTIALLEN_DhFtwLuVBYYZVgpE multi-experiment creation."""
        model_folder = 'MULTIALLEN_DhFtwLuVBYYZVgpE'
        exp = {
            'experiment_name': ['MULTIALLEN_DhFtwLuVBYYZVgpE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DhFtwLuVBYYZVgpE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BkJUfwLVMGJwXtZv(self):
        """MULTIALLEN_BkJUfwLVMGJwXtZv multi-experiment creation."""
        model_folder = 'MULTIALLEN_BkJUfwLVMGJwXtZv'
        exp = {
            'experiment_name': ['MULTIALLEN_BkJUfwLVMGJwXtZv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BkJUfwLVMGJwXtZv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IxcekEpORslNnRLT(self):
        """MULTIALLEN_IxcekEpORslNnRLT multi-experiment creation."""
        model_folder = 'MULTIALLEN_IxcekEpORslNnRLT'
        exp = {
            'experiment_name': ['MULTIALLEN_IxcekEpORslNnRLT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IxcekEpORslNnRLT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MFCwAxUdAMErvXIE(self):
        """MULTIALLEN_MFCwAxUdAMErvXIE multi-experiment creation."""
        model_folder = 'MULTIALLEN_MFCwAxUdAMErvXIE'
        exp = {
            'experiment_name': ['MULTIALLEN_MFCwAxUdAMErvXIE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MFCwAxUdAMErvXIE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rlBhlJbpeijrrgje(self):
        """MULTIALLEN_rlBhlJbpeijrrgje multi-experiment creation."""
        model_folder = 'MULTIALLEN_rlBhlJbpeijrrgje'
        exp = {
            'experiment_name': ['MULTIALLEN_rlBhlJbpeijrrgje'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rlBhlJbpeijrrgje']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GQWMCwpBzzAGiKmp(self):
        """MULTIALLEN_GQWMCwpBzzAGiKmp multi-experiment creation."""
        model_folder = 'MULTIALLEN_GQWMCwpBzzAGiKmp'
        exp = {
            'experiment_name': ['MULTIALLEN_GQWMCwpBzzAGiKmp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GQWMCwpBzzAGiKmp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BlADUejqwrjYqUlF(self):
        """MULTIALLEN_BlADUejqwrjYqUlF multi-experiment creation."""
        model_folder = 'MULTIALLEN_BlADUejqwrjYqUlF'
        exp = {
            'experiment_name': ['MULTIALLEN_BlADUejqwrjYqUlF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BlADUejqwrjYqUlF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hVLMePDFKjMZtRoY(self):
        """MULTIALLEN_hVLMePDFKjMZtRoY multi-experiment creation."""
        model_folder = 'MULTIALLEN_hVLMePDFKjMZtRoY'
        exp = {
            'experiment_name': ['MULTIALLEN_hVLMePDFKjMZtRoY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hVLMePDFKjMZtRoY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wPOdYIfykRPVWLKS(self):
        """MULTIALLEN_wPOdYIfykRPVWLKS multi-experiment creation."""
        model_folder = 'MULTIALLEN_wPOdYIfykRPVWLKS'
        exp = {
            'experiment_name': ['MULTIALLEN_wPOdYIfykRPVWLKS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wPOdYIfykRPVWLKS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cwFAdoaNteYiCDuZ(self):
        """MULTIALLEN_cwFAdoaNteYiCDuZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_cwFAdoaNteYiCDuZ'
        exp = {
            'experiment_name': ['MULTIALLEN_cwFAdoaNteYiCDuZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cwFAdoaNteYiCDuZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cmnCouvGsznuMnpV(self):
        """MULTIALLEN_cmnCouvGsznuMnpV multi-experiment creation."""
        model_folder = 'MULTIALLEN_cmnCouvGsznuMnpV'
        exp = {
            'experiment_name': ['MULTIALLEN_cmnCouvGsznuMnpV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cmnCouvGsznuMnpV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JaNzUwDEPSjQMJfe(self):
        """MULTIALLEN_JaNzUwDEPSjQMJfe multi-experiment creation."""
        model_folder = 'MULTIALLEN_JaNzUwDEPSjQMJfe'
        exp = {
            'experiment_name': ['MULTIALLEN_JaNzUwDEPSjQMJfe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JaNzUwDEPSjQMJfe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZnuPofchoeIDNCOt(self):
        """MULTIALLEN_ZnuPofchoeIDNCOt multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZnuPofchoeIDNCOt'
        exp = {
            'experiment_name': ['MULTIALLEN_ZnuPofchoeIDNCOt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZnuPofchoeIDNCOt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GNmGkVtuxaBGyYol(self):
        """MULTIALLEN_GNmGkVtuxaBGyYol multi-experiment creation."""
        model_folder = 'MULTIALLEN_GNmGkVtuxaBGyYol'
        exp = {
            'experiment_name': ['MULTIALLEN_GNmGkVtuxaBGyYol'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GNmGkVtuxaBGyYol']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zSYOtQAYIyTsHGes(self):
        """MULTIALLEN_zSYOtQAYIyTsHGes multi-experiment creation."""
        model_folder = 'MULTIALLEN_zSYOtQAYIyTsHGes'
        exp = {
            'experiment_name': ['MULTIALLEN_zSYOtQAYIyTsHGes'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zSYOtQAYIyTsHGes']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kuVqqQHHvNsSJdim(self):
        """MULTIALLEN_kuVqqQHHvNsSJdim multi-experiment creation."""
        model_folder = 'MULTIALLEN_kuVqqQHHvNsSJdim'
        exp = {
            'experiment_name': ['MULTIALLEN_kuVqqQHHvNsSJdim'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kuVqqQHHvNsSJdim']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lhzIIKLCxSFiyQkt(self):
        """MULTIALLEN_lhzIIKLCxSFiyQkt multi-experiment creation."""
        model_folder = 'MULTIALLEN_lhzIIKLCxSFiyQkt'
        exp = {
            'experiment_name': ['MULTIALLEN_lhzIIKLCxSFiyQkt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lhzIIKLCxSFiyQkt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BjTErIPLxEuSnGpp(self):
        """MULTIALLEN_BjTErIPLxEuSnGpp multi-experiment creation."""
        model_folder = 'MULTIALLEN_BjTErIPLxEuSnGpp'
        exp = {
            'experiment_name': ['MULTIALLEN_BjTErIPLxEuSnGpp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BjTErIPLxEuSnGpp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kBqzKFyzaMuleTcv(self):
        """MULTIALLEN_kBqzKFyzaMuleTcv multi-experiment creation."""
        model_folder = 'MULTIALLEN_kBqzKFyzaMuleTcv'
        exp = {
            'experiment_name': ['MULTIALLEN_kBqzKFyzaMuleTcv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kBqzKFyzaMuleTcv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_djftWcRYmCbZqeNI(self):
        """MULTIALLEN_djftWcRYmCbZqeNI multi-experiment creation."""
        model_folder = 'MULTIALLEN_djftWcRYmCbZqeNI'
        exp = {
            'experiment_name': ['MULTIALLEN_djftWcRYmCbZqeNI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_djftWcRYmCbZqeNI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rGeCUmMhrXZwfIYx(self):
        """MULTIALLEN_rGeCUmMhrXZwfIYx multi-experiment creation."""
        model_folder = 'MULTIALLEN_rGeCUmMhrXZwfIYx'
        exp = {
            'experiment_name': ['MULTIALLEN_rGeCUmMhrXZwfIYx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rGeCUmMhrXZwfIYx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rTzqBCtELNTejuUF(self):
        """MULTIALLEN_rTzqBCtELNTejuUF multi-experiment creation."""
        model_folder = 'MULTIALLEN_rTzqBCtELNTejuUF'
        exp = {
            'experiment_name': ['MULTIALLEN_rTzqBCtELNTejuUF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rTzqBCtELNTejuUF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SDTeYlMxZrRPPwKG(self):
        """MULTIALLEN_SDTeYlMxZrRPPwKG multi-experiment creation."""
        model_folder = 'MULTIALLEN_SDTeYlMxZrRPPwKG'
        exp = {
            'experiment_name': ['MULTIALLEN_SDTeYlMxZrRPPwKG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SDTeYlMxZrRPPwKG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lJYMlaRnbHMrTXLk(self):
        """MULTIALLEN_lJYMlaRnbHMrTXLk multi-experiment creation."""
        model_folder = 'MULTIALLEN_lJYMlaRnbHMrTXLk'
        exp = {
            'experiment_name': ['MULTIALLEN_lJYMlaRnbHMrTXLk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lJYMlaRnbHMrTXLk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eSezUHXVvJefzcJu(self):
        """MULTIALLEN_eSezUHXVvJefzcJu multi-experiment creation."""
        model_folder = 'MULTIALLEN_eSezUHXVvJefzcJu'
        exp = {
            'experiment_name': ['MULTIALLEN_eSezUHXVvJefzcJu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eSezUHXVvJefzcJu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sxKxbwsVDSwQBtSh(self):
        """MULTIALLEN_sxKxbwsVDSwQBtSh multi-experiment creation."""
        model_folder = 'MULTIALLEN_sxKxbwsVDSwQBtSh'
        exp = {
            'experiment_name': ['MULTIALLEN_sxKxbwsVDSwQBtSh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sxKxbwsVDSwQBtSh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GnKFdUqXKeLrzXxv(self):
        """MULTIALLEN_GnKFdUqXKeLrzXxv multi-experiment creation."""
        model_folder = 'MULTIALLEN_GnKFdUqXKeLrzXxv'
        exp = {
            'experiment_name': ['MULTIALLEN_GnKFdUqXKeLrzXxv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GnKFdUqXKeLrzXxv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QYCoBZiaxTWTUhAz(self):
        """MULTIALLEN_QYCoBZiaxTWTUhAz multi-experiment creation."""
        model_folder = 'MULTIALLEN_QYCoBZiaxTWTUhAz'
        exp = {
            'experiment_name': ['MULTIALLEN_QYCoBZiaxTWTUhAz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QYCoBZiaxTWTUhAz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sdjQUTxengoOhEJk(self):
        """MULTIALLEN_sdjQUTxengoOhEJk multi-experiment creation."""
        model_folder = 'MULTIALLEN_sdjQUTxengoOhEJk'
        exp = {
            'experiment_name': ['MULTIALLEN_sdjQUTxengoOhEJk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sdjQUTxengoOhEJk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IHKbUHzLOSrxbjvo(self):
        """MULTIALLEN_IHKbUHzLOSrxbjvo multi-experiment creation."""
        model_folder = 'MULTIALLEN_IHKbUHzLOSrxbjvo'
        exp = {
            'experiment_name': ['MULTIALLEN_IHKbUHzLOSrxbjvo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IHKbUHzLOSrxbjvo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jnQwoHHvRvdDKDit(self):
        """MULTIALLEN_jnQwoHHvRvdDKDit multi-experiment creation."""
        model_folder = 'MULTIALLEN_jnQwoHHvRvdDKDit'
        exp = {
            'experiment_name': ['MULTIALLEN_jnQwoHHvRvdDKDit'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jnQwoHHvRvdDKDit']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KvailTWSwoHUcQRf(self):
        """MULTIALLEN_KvailTWSwoHUcQRf multi-experiment creation."""
        model_folder = 'MULTIALLEN_KvailTWSwoHUcQRf'
        exp = {
            'experiment_name': ['MULTIALLEN_KvailTWSwoHUcQRf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KvailTWSwoHUcQRf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YgHNCUcvxegKInUE(self):
        """MULTIALLEN_YgHNCUcvxegKInUE multi-experiment creation."""
        model_folder = 'MULTIALLEN_YgHNCUcvxegKInUE'
        exp = {
            'experiment_name': ['MULTIALLEN_YgHNCUcvxegKInUE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YgHNCUcvxegKInUE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XmffSBxpdAWxAkoW(self):
        """MULTIALLEN_XmffSBxpdAWxAkoW multi-experiment creation."""
        model_folder = 'MULTIALLEN_XmffSBxpdAWxAkoW'
        exp = {
            'experiment_name': ['MULTIALLEN_XmffSBxpdAWxAkoW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XmffSBxpdAWxAkoW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZIvIItcLrYjsPODr(self):
        """MULTIALLEN_ZIvIItcLrYjsPODr multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZIvIItcLrYjsPODr'
        exp = {
            'experiment_name': ['MULTIALLEN_ZIvIItcLrYjsPODr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZIvIItcLrYjsPODr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pbSDfTPWPYskgXQH(self):
        """MULTIALLEN_pbSDfTPWPYskgXQH multi-experiment creation."""
        model_folder = 'MULTIALLEN_pbSDfTPWPYskgXQH'
        exp = {
            'experiment_name': ['MULTIALLEN_pbSDfTPWPYskgXQH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pbSDfTPWPYskgXQH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gQzVGTenMPprHbaG(self):
        """MULTIALLEN_gQzVGTenMPprHbaG multi-experiment creation."""
        model_folder = 'MULTIALLEN_gQzVGTenMPprHbaG'
        exp = {
            'experiment_name': ['MULTIALLEN_gQzVGTenMPprHbaG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gQzVGTenMPprHbaG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ldkNMcGnhPXDTIpK(self):
        """MULTIALLEN_ldkNMcGnhPXDTIpK multi-experiment creation."""
        model_folder = 'MULTIALLEN_ldkNMcGnhPXDTIpK'
        exp = {
            'experiment_name': ['MULTIALLEN_ldkNMcGnhPXDTIpK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ldkNMcGnhPXDTIpK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GQFZtDWwjLTzzpxj(self):
        """MULTIALLEN_GQFZtDWwjLTzzpxj multi-experiment creation."""
        model_folder = 'MULTIALLEN_GQFZtDWwjLTzzpxj'
        exp = {
            'experiment_name': ['MULTIALLEN_GQFZtDWwjLTzzpxj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GQFZtDWwjLTzzpxj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xkwrfVfucKwpFEIh(self):
        """MULTIALLEN_xkwrfVfucKwpFEIh multi-experiment creation."""
        model_folder = 'MULTIALLEN_xkwrfVfucKwpFEIh'
        exp = {
            'experiment_name': ['MULTIALLEN_xkwrfVfucKwpFEIh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xkwrfVfucKwpFEIh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IARiHimxisOUYuvw(self):
        """MULTIALLEN_IARiHimxisOUYuvw multi-experiment creation."""
        model_folder = 'MULTIALLEN_IARiHimxisOUYuvw'
        exp = {
            'experiment_name': ['MULTIALLEN_IARiHimxisOUYuvw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IARiHimxisOUYuvw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xhsOaTitbzoGtbfS(self):
        """MULTIALLEN_xhsOaTitbzoGtbfS multi-experiment creation."""
        model_folder = 'MULTIALLEN_xhsOaTitbzoGtbfS'
        exp = {
            'experiment_name': ['MULTIALLEN_xhsOaTitbzoGtbfS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xhsOaTitbzoGtbfS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LjFuiHbLnlesfBiF(self):
        """MULTIALLEN_LjFuiHbLnlesfBiF multi-experiment creation."""
        model_folder = 'MULTIALLEN_LjFuiHbLnlesfBiF'
        exp = {
            'experiment_name': ['MULTIALLEN_LjFuiHbLnlesfBiF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LjFuiHbLnlesfBiF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QVkUdTtuDpKMbEpC(self):
        """MULTIALLEN_QVkUdTtuDpKMbEpC multi-experiment creation."""
        model_folder = 'MULTIALLEN_QVkUdTtuDpKMbEpC'
        exp = {
            'experiment_name': ['MULTIALLEN_QVkUdTtuDpKMbEpC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QVkUdTtuDpKMbEpC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_roeJalXMJHfTNtni(self):
        """MULTIALLEN_roeJalXMJHfTNtni multi-experiment creation."""
        model_folder = 'MULTIALLEN_roeJalXMJHfTNtni'
        exp = {
            'experiment_name': ['MULTIALLEN_roeJalXMJHfTNtni'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_roeJalXMJHfTNtni']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_csDUWdBAnRxVJljF(self):
        """MULTIALLEN_csDUWdBAnRxVJljF multi-experiment creation."""
        model_folder = 'MULTIALLEN_csDUWdBAnRxVJljF'
        exp = {
            'experiment_name': ['MULTIALLEN_csDUWdBAnRxVJljF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_csDUWdBAnRxVJljF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_psrgMOfXPJZsKdGO(self):
        """MULTIALLEN_psrgMOfXPJZsKdGO multi-experiment creation."""
        model_folder = 'MULTIALLEN_psrgMOfXPJZsKdGO'
        exp = {
            'experiment_name': ['MULTIALLEN_psrgMOfXPJZsKdGO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_psrgMOfXPJZsKdGO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VVnYYYxLsmjeflea(self):
        """MULTIALLEN_VVnYYYxLsmjeflea multi-experiment creation."""
        model_folder = 'MULTIALLEN_VVnYYYxLsmjeflea'
        exp = {
            'experiment_name': ['MULTIALLEN_VVnYYYxLsmjeflea'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VVnYYYxLsmjeflea']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eHyAZyNuGKlLsOID(self):
        """MULTIALLEN_eHyAZyNuGKlLsOID multi-experiment creation."""
        model_folder = 'MULTIALLEN_eHyAZyNuGKlLsOID'
        exp = {
            'experiment_name': ['MULTIALLEN_eHyAZyNuGKlLsOID'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eHyAZyNuGKlLsOID']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PmjpYmixeNicXWtq(self):
        """MULTIALLEN_PmjpYmixeNicXWtq multi-experiment creation."""
        model_folder = 'MULTIALLEN_PmjpYmixeNicXWtq'
        exp = {
            'experiment_name': ['MULTIALLEN_PmjpYmixeNicXWtq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PmjpYmixeNicXWtq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZmqDGAgMyfvNPxPm(self):
        """MULTIALLEN_ZmqDGAgMyfvNPxPm multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZmqDGAgMyfvNPxPm'
        exp = {
            'experiment_name': ['MULTIALLEN_ZmqDGAgMyfvNPxPm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZmqDGAgMyfvNPxPm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BcCTFSwqDjaaxKaK(self):
        """MULTIALLEN_BcCTFSwqDjaaxKaK multi-experiment creation."""
        model_folder = 'MULTIALLEN_BcCTFSwqDjaaxKaK'
        exp = {
            'experiment_name': ['MULTIALLEN_BcCTFSwqDjaaxKaK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BcCTFSwqDjaaxKaK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XwOxayhdiYFnNTsa(self):
        """MULTIALLEN_XwOxayhdiYFnNTsa multi-experiment creation."""
        model_folder = 'MULTIALLEN_XwOxayhdiYFnNTsa'
        exp = {
            'experiment_name': ['MULTIALLEN_XwOxayhdiYFnNTsa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XwOxayhdiYFnNTsa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aVODdXQtspreAchd(self):
        """MULTIALLEN_aVODdXQtspreAchd multi-experiment creation."""
        model_folder = 'MULTIALLEN_aVODdXQtspreAchd'
        exp = {
            'experiment_name': ['MULTIALLEN_aVODdXQtspreAchd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aVODdXQtspreAchd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BWvTHJfQthvCnHpo(self):
        """MULTIALLEN_BWvTHJfQthvCnHpo multi-experiment creation."""
        model_folder = 'MULTIALLEN_BWvTHJfQthvCnHpo'
        exp = {
            'experiment_name': ['MULTIALLEN_BWvTHJfQthvCnHpo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BWvTHJfQthvCnHpo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GQSosSMOxvoKjmAj(self):
        """MULTIALLEN_GQSosSMOxvoKjmAj multi-experiment creation."""
        model_folder = 'MULTIALLEN_GQSosSMOxvoKjmAj'
        exp = {
            'experiment_name': ['MULTIALLEN_GQSosSMOxvoKjmAj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GQSosSMOxvoKjmAj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JsZMKrwYOmbdjtxZ(self):
        """MULTIALLEN_JsZMKrwYOmbdjtxZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_JsZMKrwYOmbdjtxZ'
        exp = {
            'experiment_name': ['MULTIALLEN_JsZMKrwYOmbdjtxZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JsZMKrwYOmbdjtxZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aeZFifSgJzseLJcb(self):
        """MULTIALLEN_aeZFifSgJzseLJcb multi-experiment creation."""
        model_folder = 'MULTIALLEN_aeZFifSgJzseLJcb'
        exp = {
            'experiment_name': ['MULTIALLEN_aeZFifSgJzseLJcb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aeZFifSgJzseLJcb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wDFhvxhpmsDvcmJs(self):
        """MULTIALLEN_wDFhvxhpmsDvcmJs multi-experiment creation."""
        model_folder = 'MULTIALLEN_wDFhvxhpmsDvcmJs'
        exp = {
            'experiment_name': ['MULTIALLEN_wDFhvxhpmsDvcmJs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wDFhvxhpmsDvcmJs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tbAXKwlYufaroJUl(self):
        """MULTIALLEN_tbAXKwlYufaroJUl multi-experiment creation."""
        model_folder = 'MULTIALLEN_tbAXKwlYufaroJUl'
        exp = {
            'experiment_name': ['MULTIALLEN_tbAXKwlYufaroJUl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tbAXKwlYufaroJUl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HCMtjXnYIguEIDjV(self):
        """MULTIALLEN_HCMtjXnYIguEIDjV multi-experiment creation."""
        model_folder = 'MULTIALLEN_HCMtjXnYIguEIDjV'
        exp = {
            'experiment_name': ['MULTIALLEN_HCMtjXnYIguEIDjV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HCMtjXnYIguEIDjV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fkTrZbqOVxavdJdp(self):
        """MULTIALLEN_fkTrZbqOVxavdJdp multi-experiment creation."""
        model_folder = 'MULTIALLEN_fkTrZbqOVxavdJdp'
        exp = {
            'experiment_name': ['MULTIALLEN_fkTrZbqOVxavdJdp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fkTrZbqOVxavdJdp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ReFynrHUvloMNvAa(self):
        """MULTIALLEN_ReFynrHUvloMNvAa multi-experiment creation."""
        model_folder = 'MULTIALLEN_ReFynrHUvloMNvAa'
        exp = {
            'experiment_name': ['MULTIALLEN_ReFynrHUvloMNvAa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ReFynrHUvloMNvAa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LMtRdTxmdNgvqfPG(self):
        """MULTIALLEN_LMtRdTxmdNgvqfPG multi-experiment creation."""
        model_folder = 'MULTIALLEN_LMtRdTxmdNgvqfPG'
        exp = {
            'experiment_name': ['MULTIALLEN_LMtRdTxmdNgvqfPG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LMtRdTxmdNgvqfPG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XbFMRaudjcGqPsfG(self):
        """MULTIALLEN_XbFMRaudjcGqPsfG multi-experiment creation."""
        model_folder = 'MULTIALLEN_XbFMRaudjcGqPsfG'
        exp = {
            'experiment_name': ['MULTIALLEN_XbFMRaudjcGqPsfG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XbFMRaudjcGqPsfG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ELoJuOQNeUlIYAcq(self):
        """MULTIALLEN_ELoJuOQNeUlIYAcq multi-experiment creation."""
        model_folder = 'MULTIALLEN_ELoJuOQNeUlIYAcq'
        exp = {
            'experiment_name': ['MULTIALLEN_ELoJuOQNeUlIYAcq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ELoJuOQNeUlIYAcq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HxKPKkZfJuseBIVz(self):
        """MULTIALLEN_HxKPKkZfJuseBIVz multi-experiment creation."""
        model_folder = 'MULTIALLEN_HxKPKkZfJuseBIVz'
        exp = {
            'experiment_name': ['MULTIALLEN_HxKPKkZfJuseBIVz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HxKPKkZfJuseBIVz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_srWwoCVUsqJdSEzA(self):
        """MULTIALLEN_srWwoCVUsqJdSEzA multi-experiment creation."""
        model_folder = 'MULTIALLEN_srWwoCVUsqJdSEzA'
        exp = {
            'experiment_name': ['MULTIALLEN_srWwoCVUsqJdSEzA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_srWwoCVUsqJdSEzA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kgYdZdpySZzJcWHC(self):
        """MULTIALLEN_kgYdZdpySZzJcWHC multi-experiment creation."""
        model_folder = 'MULTIALLEN_kgYdZdpySZzJcWHC'
        exp = {
            'experiment_name': ['MULTIALLEN_kgYdZdpySZzJcWHC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kgYdZdpySZzJcWHC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VmOtVOJqtBIayewq(self):
        """MULTIALLEN_VmOtVOJqtBIayewq multi-experiment creation."""
        model_folder = 'MULTIALLEN_VmOtVOJqtBIayewq'
        exp = {
            'experiment_name': ['MULTIALLEN_VmOtVOJqtBIayewq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VmOtVOJqtBIayewq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WUyTqKBtjCPngcAp(self):
        """MULTIALLEN_WUyTqKBtjCPngcAp multi-experiment creation."""
        model_folder = 'MULTIALLEN_WUyTqKBtjCPngcAp'
        exp = {
            'experiment_name': ['MULTIALLEN_WUyTqKBtjCPngcAp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WUyTqKBtjCPngcAp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bqxLMLNLYkZCBWjM(self):
        """MULTIALLEN_bqxLMLNLYkZCBWjM multi-experiment creation."""
        model_folder = 'MULTIALLEN_bqxLMLNLYkZCBWjM'
        exp = {
            'experiment_name': ['MULTIALLEN_bqxLMLNLYkZCBWjM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bqxLMLNLYkZCBWjM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LGmYFhqCZvGeTtAx(self):
        """MULTIALLEN_LGmYFhqCZvGeTtAx multi-experiment creation."""
        model_folder = 'MULTIALLEN_LGmYFhqCZvGeTtAx'
        exp = {
            'experiment_name': ['MULTIALLEN_LGmYFhqCZvGeTtAx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LGmYFhqCZvGeTtAx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CyeILlTpdwRurWxS(self):
        """MULTIALLEN_CyeILlTpdwRurWxS multi-experiment creation."""
        model_folder = 'MULTIALLEN_CyeILlTpdwRurWxS'
        exp = {
            'experiment_name': ['MULTIALLEN_CyeILlTpdwRurWxS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CyeILlTpdwRurWxS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EolzbwgwQTNAWHpz(self):
        """MULTIALLEN_EolzbwgwQTNAWHpz multi-experiment creation."""
        model_folder = 'MULTIALLEN_EolzbwgwQTNAWHpz'
        exp = {
            'experiment_name': ['MULTIALLEN_EolzbwgwQTNAWHpz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EolzbwgwQTNAWHpz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dhqhZOkkBtAmFSLu(self):
        """MULTIALLEN_dhqhZOkkBtAmFSLu multi-experiment creation."""
        model_folder = 'MULTIALLEN_dhqhZOkkBtAmFSLu'
        exp = {
            'experiment_name': ['MULTIALLEN_dhqhZOkkBtAmFSLu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dhqhZOkkBtAmFSLu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BjglAfdJGGtkAbUg(self):
        """MULTIALLEN_BjglAfdJGGtkAbUg multi-experiment creation."""
        model_folder = 'MULTIALLEN_BjglAfdJGGtkAbUg'
        exp = {
            'experiment_name': ['MULTIALLEN_BjglAfdJGGtkAbUg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BjglAfdJGGtkAbUg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YlhyxvHuiRTqNNUX(self):
        """MULTIALLEN_YlhyxvHuiRTqNNUX multi-experiment creation."""
        model_folder = 'MULTIALLEN_YlhyxvHuiRTqNNUX'
        exp = {
            'experiment_name': ['MULTIALLEN_YlhyxvHuiRTqNNUX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YlhyxvHuiRTqNNUX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SJyqlxpkuCqNFtcl(self):
        """MULTIALLEN_SJyqlxpkuCqNFtcl multi-experiment creation."""
        model_folder = 'MULTIALLEN_SJyqlxpkuCqNFtcl'
        exp = {
            'experiment_name': ['MULTIALLEN_SJyqlxpkuCqNFtcl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SJyqlxpkuCqNFtcl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PtxnambHDkJdyZeb(self):
        """MULTIALLEN_PtxnambHDkJdyZeb multi-experiment creation."""
        model_folder = 'MULTIALLEN_PtxnambHDkJdyZeb'
        exp = {
            'experiment_name': ['MULTIALLEN_PtxnambHDkJdyZeb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PtxnambHDkJdyZeb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pzmbQsslDJeojfwP(self):
        """MULTIALLEN_pzmbQsslDJeojfwP multi-experiment creation."""
        model_folder = 'MULTIALLEN_pzmbQsslDJeojfwP'
        exp = {
            'experiment_name': ['MULTIALLEN_pzmbQsslDJeojfwP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pzmbQsslDJeojfwP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NeFRHfDEgbyeLaUQ(self):
        """MULTIALLEN_NeFRHfDEgbyeLaUQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_NeFRHfDEgbyeLaUQ'
        exp = {
            'experiment_name': ['MULTIALLEN_NeFRHfDEgbyeLaUQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NeFRHfDEgbyeLaUQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pYuSEPQUGKmwkxNC(self):
        """MULTIALLEN_pYuSEPQUGKmwkxNC multi-experiment creation."""
        model_folder = 'MULTIALLEN_pYuSEPQUGKmwkxNC'
        exp = {
            'experiment_name': ['MULTIALLEN_pYuSEPQUGKmwkxNC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pYuSEPQUGKmwkxNC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RWMweJvWSydRKBxS(self):
        """MULTIALLEN_RWMweJvWSydRKBxS multi-experiment creation."""
        model_folder = 'MULTIALLEN_RWMweJvWSydRKBxS'
        exp = {
            'experiment_name': ['MULTIALLEN_RWMweJvWSydRKBxS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RWMweJvWSydRKBxS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kapDsfmGzMFEEAwF(self):
        """MULTIALLEN_kapDsfmGzMFEEAwF multi-experiment creation."""
        model_folder = 'MULTIALLEN_kapDsfmGzMFEEAwF'
        exp = {
            'experiment_name': ['MULTIALLEN_kapDsfmGzMFEEAwF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kapDsfmGzMFEEAwF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bDxESAbodODBvQNv(self):
        """MULTIALLEN_bDxESAbodODBvQNv multi-experiment creation."""
        model_folder = 'MULTIALLEN_bDxESAbodODBvQNv'
        exp = {
            'experiment_name': ['MULTIALLEN_bDxESAbodODBvQNv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bDxESAbodODBvQNv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KWvnpatwPQWDHtHv(self):
        """MULTIALLEN_KWvnpatwPQWDHtHv multi-experiment creation."""
        model_folder = 'MULTIALLEN_KWvnpatwPQWDHtHv'
        exp = {
            'experiment_name': ['MULTIALLEN_KWvnpatwPQWDHtHv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KWvnpatwPQWDHtHv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NNqhivrkKgOMGJwR(self):
        """MULTIALLEN_NNqhivrkKgOMGJwR multi-experiment creation."""
        model_folder = 'MULTIALLEN_NNqhivrkKgOMGJwR'
        exp = {
            'experiment_name': ['MULTIALLEN_NNqhivrkKgOMGJwR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NNqhivrkKgOMGJwR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OBkxQTJsNYzmUrXe(self):
        """MULTIALLEN_OBkxQTJsNYzmUrXe multi-experiment creation."""
        model_folder = 'MULTIALLEN_OBkxQTJsNYzmUrXe'
        exp = {
            'experiment_name': ['MULTIALLEN_OBkxQTJsNYzmUrXe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OBkxQTJsNYzmUrXe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WcbulgXCLhWHzrJz(self):
        """MULTIALLEN_WcbulgXCLhWHzrJz multi-experiment creation."""
        model_folder = 'MULTIALLEN_WcbulgXCLhWHzrJz'
        exp = {
            'experiment_name': ['MULTIALLEN_WcbulgXCLhWHzrJz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WcbulgXCLhWHzrJz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OqZsltNHSTmKuccI(self):
        """MULTIALLEN_OqZsltNHSTmKuccI multi-experiment creation."""
        model_folder = 'MULTIALLEN_OqZsltNHSTmKuccI'
        exp = {
            'experiment_name': ['MULTIALLEN_OqZsltNHSTmKuccI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OqZsltNHSTmKuccI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WBmUvUDQcLjKaZHl(self):
        """MULTIALLEN_WBmUvUDQcLjKaZHl multi-experiment creation."""
        model_folder = 'MULTIALLEN_WBmUvUDQcLjKaZHl'
        exp = {
            'experiment_name': ['MULTIALLEN_WBmUvUDQcLjKaZHl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WBmUvUDQcLjKaZHl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kelLSSUMjbUVvQuT(self):
        """MULTIALLEN_kelLSSUMjbUVvQuT multi-experiment creation."""
        model_folder = 'MULTIALLEN_kelLSSUMjbUVvQuT'
        exp = {
            'experiment_name': ['MULTIALLEN_kelLSSUMjbUVvQuT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kelLSSUMjbUVvQuT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KIxMouwMLHxZSfWE(self):
        """MULTIALLEN_KIxMouwMLHxZSfWE multi-experiment creation."""
        model_folder = 'MULTIALLEN_KIxMouwMLHxZSfWE'
        exp = {
            'experiment_name': ['MULTIALLEN_KIxMouwMLHxZSfWE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KIxMouwMLHxZSfWE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nrhZUQqXdHzhOwBb(self):
        """MULTIALLEN_nrhZUQqXdHzhOwBb multi-experiment creation."""
        model_folder = 'MULTIALLEN_nrhZUQqXdHzhOwBb'
        exp = {
            'experiment_name': ['MULTIALLEN_nrhZUQqXdHzhOwBb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nrhZUQqXdHzhOwBb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hTTZuQYgVVJKRRNz(self):
        """MULTIALLEN_hTTZuQYgVVJKRRNz multi-experiment creation."""
        model_folder = 'MULTIALLEN_hTTZuQYgVVJKRRNz'
        exp = {
            'experiment_name': ['MULTIALLEN_hTTZuQYgVVJKRRNz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hTTZuQYgVVJKRRNz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oTggNzwiRbvkTjFG(self):
        """MULTIALLEN_oTggNzwiRbvkTjFG multi-experiment creation."""
        model_folder = 'MULTIALLEN_oTggNzwiRbvkTjFG'
        exp = {
            'experiment_name': ['MULTIALLEN_oTggNzwiRbvkTjFG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oTggNzwiRbvkTjFG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YZvETQSJpvHTiFWM(self):
        """MULTIALLEN_YZvETQSJpvHTiFWM multi-experiment creation."""
        model_folder = 'MULTIALLEN_YZvETQSJpvHTiFWM'
        exp = {
            'experiment_name': ['MULTIALLEN_YZvETQSJpvHTiFWM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YZvETQSJpvHTiFWM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YOqiDHCWROsJqAEj(self):
        """MULTIALLEN_YOqiDHCWROsJqAEj multi-experiment creation."""
        model_folder = 'MULTIALLEN_YOqiDHCWROsJqAEj'
        exp = {
            'experiment_name': ['MULTIALLEN_YOqiDHCWROsJqAEj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YOqiDHCWROsJqAEj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IlstCfDFuKPpsyJM(self):
        """MULTIALLEN_IlstCfDFuKPpsyJM multi-experiment creation."""
        model_folder = 'MULTIALLEN_IlstCfDFuKPpsyJM'
        exp = {
            'experiment_name': ['MULTIALLEN_IlstCfDFuKPpsyJM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IlstCfDFuKPpsyJM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eWgkUVPfkNEvUuxz(self):
        """MULTIALLEN_eWgkUVPfkNEvUuxz multi-experiment creation."""
        model_folder = 'MULTIALLEN_eWgkUVPfkNEvUuxz'
        exp = {
            'experiment_name': ['MULTIALLEN_eWgkUVPfkNEvUuxz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eWgkUVPfkNEvUuxz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xyVGzpPeDpgnFZvl(self):
        """MULTIALLEN_xyVGzpPeDpgnFZvl multi-experiment creation."""
        model_folder = 'MULTIALLEN_xyVGzpPeDpgnFZvl'
        exp = {
            'experiment_name': ['MULTIALLEN_xyVGzpPeDpgnFZvl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xyVGzpPeDpgnFZvl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QvGZbTsWOYRnAmqN(self):
        """MULTIALLEN_QvGZbTsWOYRnAmqN multi-experiment creation."""
        model_folder = 'MULTIALLEN_QvGZbTsWOYRnAmqN'
        exp = {
            'experiment_name': ['MULTIALLEN_QvGZbTsWOYRnAmqN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QvGZbTsWOYRnAmqN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rkhBXarkAiuGZGtz(self):
        """MULTIALLEN_rkhBXarkAiuGZGtz multi-experiment creation."""
        model_folder = 'MULTIALLEN_rkhBXarkAiuGZGtz'
        exp = {
            'experiment_name': ['MULTIALLEN_rkhBXarkAiuGZGtz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rkhBXarkAiuGZGtz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VpdBLDlxiqfopEhG(self):
        """MULTIALLEN_VpdBLDlxiqfopEhG multi-experiment creation."""
        model_folder = 'MULTIALLEN_VpdBLDlxiqfopEhG'
        exp = {
            'experiment_name': ['MULTIALLEN_VpdBLDlxiqfopEhG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VpdBLDlxiqfopEhG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SrfjljOmzqNSyTJh(self):
        """MULTIALLEN_SrfjljOmzqNSyTJh multi-experiment creation."""
        model_folder = 'MULTIALLEN_SrfjljOmzqNSyTJh'
        exp = {
            'experiment_name': ['MULTIALLEN_SrfjljOmzqNSyTJh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SrfjljOmzqNSyTJh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RQibictECCVedLWl(self):
        """MULTIALLEN_RQibictECCVedLWl multi-experiment creation."""
        model_folder = 'MULTIALLEN_RQibictECCVedLWl'
        exp = {
            'experiment_name': ['MULTIALLEN_RQibictECCVedLWl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RQibictECCVedLWl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BQdwSrypVWxnHFLP(self):
        """MULTIALLEN_BQdwSrypVWxnHFLP multi-experiment creation."""
        model_folder = 'MULTIALLEN_BQdwSrypVWxnHFLP'
        exp = {
            'experiment_name': ['MULTIALLEN_BQdwSrypVWxnHFLP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BQdwSrypVWxnHFLP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KDkwzHTAMAnDXtBu(self):
        """MULTIALLEN_KDkwzHTAMAnDXtBu multi-experiment creation."""
        model_folder = 'MULTIALLEN_KDkwzHTAMAnDXtBu'
        exp = {
            'experiment_name': ['MULTIALLEN_KDkwzHTAMAnDXtBu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KDkwzHTAMAnDXtBu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wzVVVpcxKuXChwva(self):
        """MULTIALLEN_wzVVVpcxKuXChwva multi-experiment creation."""
        model_folder = 'MULTIALLEN_wzVVVpcxKuXChwva'
        exp = {
            'experiment_name': ['MULTIALLEN_wzVVVpcxKuXChwva'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wzVVVpcxKuXChwva']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UFgslzQMbWCqFIUF(self):
        """MULTIALLEN_UFgslzQMbWCqFIUF multi-experiment creation."""
        model_folder = 'MULTIALLEN_UFgslzQMbWCqFIUF'
        exp = {
            'experiment_name': ['MULTIALLEN_UFgslzQMbWCqFIUF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UFgslzQMbWCqFIUF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JCqLELivCFxlfXbs(self):
        """MULTIALLEN_JCqLELivCFxlfXbs multi-experiment creation."""
        model_folder = 'MULTIALLEN_JCqLELivCFxlfXbs'
        exp = {
            'experiment_name': ['MULTIALLEN_JCqLELivCFxlfXbs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JCqLELivCFxlfXbs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yshiPScmGjrxYpAa(self):
        """MULTIALLEN_yshiPScmGjrxYpAa multi-experiment creation."""
        model_folder = 'MULTIALLEN_yshiPScmGjrxYpAa'
        exp = {
            'experiment_name': ['MULTIALLEN_yshiPScmGjrxYpAa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yshiPScmGjrxYpAa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pgLociycnCZnsBGm(self):
        """MULTIALLEN_pgLociycnCZnsBGm multi-experiment creation."""
        model_folder = 'MULTIALLEN_pgLociycnCZnsBGm'
        exp = {
            'experiment_name': ['MULTIALLEN_pgLociycnCZnsBGm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pgLociycnCZnsBGm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_utmtbixhBkWoLRHf(self):
        """MULTIALLEN_utmtbixhBkWoLRHf multi-experiment creation."""
        model_folder = 'MULTIALLEN_utmtbixhBkWoLRHf'
        exp = {
            'experiment_name': ['MULTIALLEN_utmtbixhBkWoLRHf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_utmtbixhBkWoLRHf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eIiCsxKpuHVbzSsP(self):
        """MULTIALLEN_eIiCsxKpuHVbzSsP multi-experiment creation."""
        model_folder = 'MULTIALLEN_eIiCsxKpuHVbzSsP'
        exp = {
            'experiment_name': ['MULTIALLEN_eIiCsxKpuHVbzSsP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eIiCsxKpuHVbzSsP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EXMivQbyiajWJBgY(self):
        """MULTIALLEN_EXMivQbyiajWJBgY multi-experiment creation."""
        model_folder = 'MULTIALLEN_EXMivQbyiajWJBgY'
        exp = {
            'experiment_name': ['MULTIALLEN_EXMivQbyiajWJBgY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EXMivQbyiajWJBgY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wIgJRdZlLvxTXPWx(self):
        """MULTIALLEN_wIgJRdZlLvxTXPWx multi-experiment creation."""
        model_folder = 'MULTIALLEN_wIgJRdZlLvxTXPWx'
        exp = {
            'experiment_name': ['MULTIALLEN_wIgJRdZlLvxTXPWx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wIgJRdZlLvxTXPWx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iRFYpsXClgIGfSKP(self):
        """MULTIALLEN_iRFYpsXClgIGfSKP multi-experiment creation."""
        model_folder = 'MULTIALLEN_iRFYpsXClgIGfSKP'
        exp = {
            'experiment_name': ['MULTIALLEN_iRFYpsXClgIGfSKP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iRFYpsXClgIGfSKP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DZzDUUZWiLeQUSJx(self):
        """MULTIALLEN_DZzDUUZWiLeQUSJx multi-experiment creation."""
        model_folder = 'MULTIALLEN_DZzDUUZWiLeQUSJx'
        exp = {
            'experiment_name': ['MULTIALLEN_DZzDUUZWiLeQUSJx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DZzDUUZWiLeQUSJx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fKROTrTbCwItheoP(self):
        """MULTIALLEN_fKROTrTbCwItheoP multi-experiment creation."""
        model_folder = 'MULTIALLEN_fKROTrTbCwItheoP'
        exp = {
            'experiment_name': ['MULTIALLEN_fKROTrTbCwItheoP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fKROTrTbCwItheoP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HVRPAVOUnFgCVARl(self):
        """MULTIALLEN_HVRPAVOUnFgCVARl multi-experiment creation."""
        model_folder = 'MULTIALLEN_HVRPAVOUnFgCVARl'
        exp = {
            'experiment_name': ['MULTIALLEN_HVRPAVOUnFgCVARl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HVRPAVOUnFgCVARl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gvpBqSImMPeoeBkm(self):
        """MULTIALLEN_gvpBqSImMPeoeBkm multi-experiment creation."""
        model_folder = 'MULTIALLEN_gvpBqSImMPeoeBkm'
        exp = {
            'experiment_name': ['MULTIALLEN_gvpBqSImMPeoeBkm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gvpBqSImMPeoeBkm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iWsvJFgCjwRZZWRt(self):
        """MULTIALLEN_iWsvJFgCjwRZZWRt multi-experiment creation."""
        model_folder = 'MULTIALLEN_iWsvJFgCjwRZZWRt'
        exp = {
            'experiment_name': ['MULTIALLEN_iWsvJFgCjwRZZWRt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iWsvJFgCjwRZZWRt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SGwxBxOLrzvYKRku(self):
        """MULTIALLEN_SGwxBxOLrzvYKRku multi-experiment creation."""
        model_folder = 'MULTIALLEN_SGwxBxOLrzvYKRku'
        exp = {
            'experiment_name': ['MULTIALLEN_SGwxBxOLrzvYKRku'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SGwxBxOLrzvYKRku']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EKyTPcZLeKhbccqU(self):
        """MULTIALLEN_EKyTPcZLeKhbccqU multi-experiment creation."""
        model_folder = 'MULTIALLEN_EKyTPcZLeKhbccqU'
        exp = {
            'experiment_name': ['MULTIALLEN_EKyTPcZLeKhbccqU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EKyTPcZLeKhbccqU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IqbJTVWKJjKFOhCv(self):
        """MULTIALLEN_IqbJTVWKJjKFOhCv multi-experiment creation."""
        model_folder = 'MULTIALLEN_IqbJTVWKJjKFOhCv'
        exp = {
            'experiment_name': ['MULTIALLEN_IqbJTVWKJjKFOhCv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IqbJTVWKJjKFOhCv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vsXPwHLHCufVZYus(self):
        """MULTIALLEN_vsXPwHLHCufVZYus multi-experiment creation."""
        model_folder = 'MULTIALLEN_vsXPwHLHCufVZYus'
        exp = {
            'experiment_name': ['MULTIALLEN_vsXPwHLHCufVZYus'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vsXPwHLHCufVZYus']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gyXiPhWwpcznskTm(self):
        """MULTIALLEN_gyXiPhWwpcznskTm multi-experiment creation."""
        model_folder = 'MULTIALLEN_gyXiPhWwpcznskTm'
        exp = {
            'experiment_name': ['MULTIALLEN_gyXiPhWwpcznskTm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gyXiPhWwpcznskTm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cvbYliUTqjmpNoEV(self):
        """MULTIALLEN_cvbYliUTqjmpNoEV multi-experiment creation."""
        model_folder = 'MULTIALLEN_cvbYliUTqjmpNoEV'
        exp = {
            'experiment_name': ['MULTIALLEN_cvbYliUTqjmpNoEV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cvbYliUTqjmpNoEV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TTZgzsBAWowXqxbj(self):
        """MULTIALLEN_TTZgzsBAWowXqxbj multi-experiment creation."""
        model_folder = 'MULTIALLEN_TTZgzsBAWowXqxbj'
        exp = {
            'experiment_name': ['MULTIALLEN_TTZgzsBAWowXqxbj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TTZgzsBAWowXqxbj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xMbOCbBSLFzcjpsq(self):
        """MULTIALLEN_xMbOCbBSLFzcjpsq multi-experiment creation."""
        model_folder = 'MULTIALLEN_xMbOCbBSLFzcjpsq'
        exp = {
            'experiment_name': ['MULTIALLEN_xMbOCbBSLFzcjpsq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xMbOCbBSLFzcjpsq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UfgPBffPevBhOQMa(self):
        """MULTIALLEN_UfgPBffPevBhOQMa multi-experiment creation."""
        model_folder = 'MULTIALLEN_UfgPBffPevBhOQMa'
        exp = {
            'experiment_name': ['MULTIALLEN_UfgPBffPevBhOQMa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UfgPBffPevBhOQMa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iljKbTVxNglKqZVB(self):
        """MULTIALLEN_iljKbTVxNglKqZVB multi-experiment creation."""
        model_folder = 'MULTIALLEN_iljKbTVxNglKqZVB'
        exp = {
            'experiment_name': ['MULTIALLEN_iljKbTVxNglKqZVB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iljKbTVxNglKqZVB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sKtBxoOdpCxoWtSc(self):
        """MULTIALLEN_sKtBxoOdpCxoWtSc multi-experiment creation."""
        model_folder = 'MULTIALLEN_sKtBxoOdpCxoWtSc'
        exp = {
            'experiment_name': ['MULTIALLEN_sKtBxoOdpCxoWtSc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sKtBxoOdpCxoWtSc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kRWcCZdDIgoCiNaO(self):
        """MULTIALLEN_kRWcCZdDIgoCiNaO multi-experiment creation."""
        model_folder = 'MULTIALLEN_kRWcCZdDIgoCiNaO'
        exp = {
            'experiment_name': ['MULTIALLEN_kRWcCZdDIgoCiNaO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kRWcCZdDIgoCiNaO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gScdjRdhiQabKKPN(self):
        """MULTIALLEN_gScdjRdhiQabKKPN multi-experiment creation."""
        model_folder = 'MULTIALLEN_gScdjRdhiQabKKPN'
        exp = {
            'experiment_name': ['MULTIALLEN_gScdjRdhiQabKKPN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gScdjRdhiQabKKPN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lpttaBSckYXdgefA(self):
        """MULTIALLEN_lpttaBSckYXdgefA multi-experiment creation."""
        model_folder = 'MULTIALLEN_lpttaBSckYXdgefA'
        exp = {
            'experiment_name': ['MULTIALLEN_lpttaBSckYXdgefA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lpttaBSckYXdgefA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LEXFJEXtkLmKIyBQ(self):
        """MULTIALLEN_LEXFJEXtkLmKIyBQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_LEXFJEXtkLmKIyBQ'
        exp = {
            'experiment_name': ['MULTIALLEN_LEXFJEXtkLmKIyBQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LEXFJEXtkLmKIyBQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MihHcpZLfWZNxtgP(self):
        """MULTIALLEN_MihHcpZLfWZNxtgP multi-experiment creation."""
        model_folder = 'MULTIALLEN_MihHcpZLfWZNxtgP'
        exp = {
            'experiment_name': ['MULTIALLEN_MihHcpZLfWZNxtgP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MihHcpZLfWZNxtgP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ShkzYXupIPbdibes(self):
        """MULTIALLEN_ShkzYXupIPbdibes multi-experiment creation."""
        model_folder = 'MULTIALLEN_ShkzYXupIPbdibes'
        exp = {
            'experiment_name': ['MULTIALLEN_ShkzYXupIPbdibes'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ShkzYXupIPbdibes']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HTfNwoLfybwQzmCX(self):
        """MULTIALLEN_HTfNwoLfybwQzmCX multi-experiment creation."""
        model_folder = 'MULTIALLEN_HTfNwoLfybwQzmCX'
        exp = {
            'experiment_name': ['MULTIALLEN_HTfNwoLfybwQzmCX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HTfNwoLfybwQzmCX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gKzBcSlkUZCDIskj(self):
        """MULTIALLEN_gKzBcSlkUZCDIskj multi-experiment creation."""
        model_folder = 'MULTIALLEN_gKzBcSlkUZCDIskj'
        exp = {
            'experiment_name': ['MULTIALLEN_gKzBcSlkUZCDIskj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gKzBcSlkUZCDIskj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cMgtjPqVzBCtnzaF(self):
        """MULTIALLEN_cMgtjPqVzBCtnzaF multi-experiment creation."""
        model_folder = 'MULTIALLEN_cMgtjPqVzBCtnzaF'
        exp = {
            'experiment_name': ['MULTIALLEN_cMgtjPqVzBCtnzaF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cMgtjPqVzBCtnzaF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IGYtszrsEgJaktME(self):
        """MULTIALLEN_IGYtszrsEgJaktME multi-experiment creation."""
        model_folder = 'MULTIALLEN_IGYtszrsEgJaktME'
        exp = {
            'experiment_name': ['MULTIALLEN_IGYtszrsEgJaktME'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IGYtszrsEgJaktME']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UvCxCTzqQWVJPUgy(self):
        """MULTIALLEN_UvCxCTzqQWVJPUgy multi-experiment creation."""
        model_folder = 'MULTIALLEN_UvCxCTzqQWVJPUgy'
        exp = {
            'experiment_name': ['MULTIALLEN_UvCxCTzqQWVJPUgy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UvCxCTzqQWVJPUgy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dZEPDVSCHUdXWZCx(self):
        """MULTIALLEN_dZEPDVSCHUdXWZCx multi-experiment creation."""
        model_folder = 'MULTIALLEN_dZEPDVSCHUdXWZCx'
        exp = {
            'experiment_name': ['MULTIALLEN_dZEPDVSCHUdXWZCx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dZEPDVSCHUdXWZCx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PjQHeRQSUHHBfdrM(self):
        """MULTIALLEN_PjQHeRQSUHHBfdrM multi-experiment creation."""
        model_folder = 'MULTIALLEN_PjQHeRQSUHHBfdrM'
        exp = {
            'experiment_name': ['MULTIALLEN_PjQHeRQSUHHBfdrM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PjQHeRQSUHHBfdrM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zbYTeNcmKOYQdmtK(self):
        """MULTIALLEN_zbYTeNcmKOYQdmtK multi-experiment creation."""
        model_folder = 'MULTIALLEN_zbYTeNcmKOYQdmtK'
        exp = {
            'experiment_name': ['MULTIALLEN_zbYTeNcmKOYQdmtK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zbYTeNcmKOYQdmtK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jrRFtFYkGLMQQzTz(self):
        """MULTIALLEN_jrRFtFYkGLMQQzTz multi-experiment creation."""
        model_folder = 'MULTIALLEN_jrRFtFYkGLMQQzTz'
        exp = {
            'experiment_name': ['MULTIALLEN_jrRFtFYkGLMQQzTz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jrRFtFYkGLMQQzTz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JGDXHoGegbdRsAEp(self):
        """MULTIALLEN_JGDXHoGegbdRsAEp multi-experiment creation."""
        model_folder = 'MULTIALLEN_JGDXHoGegbdRsAEp'
        exp = {
            'experiment_name': ['MULTIALLEN_JGDXHoGegbdRsAEp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JGDXHoGegbdRsAEp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yubRXqLKqSgBqcpk(self):
        """MULTIALLEN_yubRXqLKqSgBqcpk multi-experiment creation."""
        model_folder = 'MULTIALLEN_yubRXqLKqSgBqcpk'
        exp = {
            'experiment_name': ['MULTIALLEN_yubRXqLKqSgBqcpk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yubRXqLKqSgBqcpk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eIHZslEDjhZgXkfk(self):
        """MULTIALLEN_eIHZslEDjhZgXkfk multi-experiment creation."""
        model_folder = 'MULTIALLEN_eIHZslEDjhZgXkfk'
        exp = {
            'experiment_name': ['MULTIALLEN_eIHZslEDjhZgXkfk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eIHZslEDjhZgXkfk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hSiwbiSrwuBLhNVt(self):
        """MULTIALLEN_hSiwbiSrwuBLhNVt multi-experiment creation."""
        model_folder = 'MULTIALLEN_hSiwbiSrwuBLhNVt'
        exp = {
            'experiment_name': ['MULTIALLEN_hSiwbiSrwuBLhNVt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hSiwbiSrwuBLhNVt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ouCJKGUUIkqEoBAZ(self):
        """MULTIALLEN_ouCJKGUUIkqEoBAZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_ouCJKGUUIkqEoBAZ'
        exp = {
            'experiment_name': ['MULTIALLEN_ouCJKGUUIkqEoBAZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ouCJKGUUIkqEoBAZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JKbvBhIQYZkyYWCc(self):
        """MULTIALLEN_JKbvBhIQYZkyYWCc multi-experiment creation."""
        model_folder = 'MULTIALLEN_JKbvBhIQYZkyYWCc'
        exp = {
            'experiment_name': ['MULTIALLEN_JKbvBhIQYZkyYWCc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JKbvBhIQYZkyYWCc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LJyRZPSrEKsuTlmG(self):
        """MULTIALLEN_LJyRZPSrEKsuTlmG multi-experiment creation."""
        model_folder = 'MULTIALLEN_LJyRZPSrEKsuTlmG'
        exp = {
            'experiment_name': ['MULTIALLEN_LJyRZPSrEKsuTlmG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LJyRZPSrEKsuTlmG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PinYdoImVaSUAjuP(self):
        """MULTIALLEN_PinYdoImVaSUAjuP multi-experiment creation."""
        model_folder = 'MULTIALLEN_PinYdoImVaSUAjuP'
        exp = {
            'experiment_name': ['MULTIALLEN_PinYdoImVaSUAjuP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PinYdoImVaSUAjuP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_avQwcJpmpMMYkpmL(self):
        """MULTIALLEN_avQwcJpmpMMYkpmL multi-experiment creation."""
        model_folder = 'MULTIALLEN_avQwcJpmpMMYkpmL'
        exp = {
            'experiment_name': ['MULTIALLEN_avQwcJpmpMMYkpmL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_avQwcJpmpMMYkpmL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SCrmcWIZnRzJpFhd(self):
        """MULTIALLEN_SCrmcWIZnRzJpFhd multi-experiment creation."""
        model_folder = 'MULTIALLEN_SCrmcWIZnRzJpFhd'
        exp = {
            'experiment_name': ['MULTIALLEN_SCrmcWIZnRzJpFhd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SCrmcWIZnRzJpFhd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cfTBpNDVMFnOxXro(self):
        """MULTIALLEN_cfTBpNDVMFnOxXro multi-experiment creation."""
        model_folder = 'MULTIALLEN_cfTBpNDVMFnOxXro'
        exp = {
            'experiment_name': ['MULTIALLEN_cfTBpNDVMFnOxXro'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cfTBpNDVMFnOxXro']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tsrHnnMDlaIOiBkh(self):
        """MULTIALLEN_tsrHnnMDlaIOiBkh multi-experiment creation."""
        model_folder = 'MULTIALLEN_tsrHnnMDlaIOiBkh'
        exp = {
            'experiment_name': ['MULTIALLEN_tsrHnnMDlaIOiBkh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tsrHnnMDlaIOiBkh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lcfHLqOozjeuMSut(self):
        """MULTIALLEN_lcfHLqOozjeuMSut multi-experiment creation."""
        model_folder = 'MULTIALLEN_lcfHLqOozjeuMSut'
        exp = {
            'experiment_name': ['MULTIALLEN_lcfHLqOozjeuMSut'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lcfHLqOozjeuMSut']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aUjaMjyIFBEPLwLi(self):
        """MULTIALLEN_aUjaMjyIFBEPLwLi multi-experiment creation."""
        model_folder = 'MULTIALLEN_aUjaMjyIFBEPLwLi'
        exp = {
            'experiment_name': ['MULTIALLEN_aUjaMjyIFBEPLwLi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aUjaMjyIFBEPLwLi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XKresCCVEmskmIDL(self):
        """MULTIALLEN_XKresCCVEmskmIDL multi-experiment creation."""
        model_folder = 'MULTIALLEN_XKresCCVEmskmIDL'
        exp = {
            'experiment_name': ['MULTIALLEN_XKresCCVEmskmIDL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XKresCCVEmskmIDL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xSARgLIKEmeviUsO(self):
        """MULTIALLEN_xSARgLIKEmeviUsO multi-experiment creation."""
        model_folder = 'MULTIALLEN_xSARgLIKEmeviUsO'
        exp = {
            'experiment_name': ['MULTIALLEN_xSARgLIKEmeviUsO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xSARgLIKEmeviUsO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AmfRUGeXrjXFHOQY(self):
        """MULTIALLEN_AmfRUGeXrjXFHOQY multi-experiment creation."""
        model_folder = 'MULTIALLEN_AmfRUGeXrjXFHOQY'
        exp = {
            'experiment_name': ['MULTIALLEN_AmfRUGeXrjXFHOQY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AmfRUGeXrjXFHOQY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SOrgifEskztwEErP(self):
        """MULTIALLEN_SOrgifEskztwEErP multi-experiment creation."""
        model_folder = 'MULTIALLEN_SOrgifEskztwEErP'
        exp = {
            'experiment_name': ['MULTIALLEN_SOrgifEskztwEErP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SOrgifEskztwEErP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sHmBJWjTGantvhne(self):
        """MULTIALLEN_sHmBJWjTGantvhne multi-experiment creation."""
        model_folder = 'MULTIALLEN_sHmBJWjTGantvhne'
        exp = {
            'experiment_name': ['MULTIALLEN_sHmBJWjTGantvhne'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sHmBJWjTGantvhne']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DvqCWVDMSNXwEtGw(self):
        """MULTIALLEN_DvqCWVDMSNXwEtGw multi-experiment creation."""
        model_folder = 'MULTIALLEN_DvqCWVDMSNXwEtGw'
        exp = {
            'experiment_name': ['MULTIALLEN_DvqCWVDMSNXwEtGw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DvqCWVDMSNXwEtGw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZtBiQMSiQJqPZFgn(self):
        """MULTIALLEN_ZtBiQMSiQJqPZFgn multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZtBiQMSiQJqPZFgn'
        exp = {
            'experiment_name': ['MULTIALLEN_ZtBiQMSiQJqPZFgn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZtBiQMSiQJqPZFgn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_noAJvLdgSeQYKkFy(self):
        """MULTIALLEN_noAJvLdgSeQYKkFy multi-experiment creation."""
        model_folder = 'MULTIALLEN_noAJvLdgSeQYKkFy'
        exp = {
            'experiment_name': ['MULTIALLEN_noAJvLdgSeQYKkFy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_noAJvLdgSeQYKkFy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DAslnOlVLXMoxdaG(self):
        """MULTIALLEN_DAslnOlVLXMoxdaG multi-experiment creation."""
        model_folder = 'MULTIALLEN_DAslnOlVLXMoxdaG'
        exp = {
            'experiment_name': ['MULTIALLEN_DAslnOlVLXMoxdaG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DAslnOlVLXMoxdaG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aPBVnTOEHHdWDJnz(self):
        """MULTIALLEN_aPBVnTOEHHdWDJnz multi-experiment creation."""
        model_folder = 'MULTIALLEN_aPBVnTOEHHdWDJnz'
        exp = {
            'experiment_name': ['MULTIALLEN_aPBVnTOEHHdWDJnz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aPBVnTOEHHdWDJnz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MmfSSwNbYJhWUZWf(self):
        """MULTIALLEN_MmfSSwNbYJhWUZWf multi-experiment creation."""
        model_folder = 'MULTIALLEN_MmfSSwNbYJhWUZWf'
        exp = {
            'experiment_name': ['MULTIALLEN_MmfSSwNbYJhWUZWf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MmfSSwNbYJhWUZWf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nxlbJxFyRvdwqZpZ(self):
        """MULTIALLEN_nxlbJxFyRvdwqZpZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_nxlbJxFyRvdwqZpZ'
        exp = {
            'experiment_name': ['MULTIALLEN_nxlbJxFyRvdwqZpZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nxlbJxFyRvdwqZpZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kDBqUHEscSGOPvMp(self):
        """MULTIALLEN_kDBqUHEscSGOPvMp multi-experiment creation."""
        model_folder = 'MULTIALLEN_kDBqUHEscSGOPvMp'
        exp = {
            'experiment_name': ['MULTIALLEN_kDBqUHEscSGOPvMp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kDBqUHEscSGOPvMp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NOKqUmfFwWFSAnpA(self):
        """MULTIALLEN_NOKqUmfFwWFSAnpA multi-experiment creation."""
        model_folder = 'MULTIALLEN_NOKqUmfFwWFSAnpA'
        exp = {
            'experiment_name': ['MULTIALLEN_NOKqUmfFwWFSAnpA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NOKqUmfFwWFSAnpA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zmecSXhCSZLxJJwT(self):
        """MULTIALLEN_zmecSXhCSZLxJJwT multi-experiment creation."""
        model_folder = 'MULTIALLEN_zmecSXhCSZLxJJwT'
        exp = {
            'experiment_name': ['MULTIALLEN_zmecSXhCSZLxJJwT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zmecSXhCSZLxJJwT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BJPAOinibxItQamL(self):
        """MULTIALLEN_BJPAOinibxItQamL multi-experiment creation."""
        model_folder = 'MULTIALLEN_BJPAOinibxItQamL'
        exp = {
            'experiment_name': ['MULTIALLEN_BJPAOinibxItQamL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BJPAOinibxItQamL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hHGoXQaQCOsGlZBa(self):
        """MULTIALLEN_hHGoXQaQCOsGlZBa multi-experiment creation."""
        model_folder = 'MULTIALLEN_hHGoXQaQCOsGlZBa'
        exp = {
            'experiment_name': ['MULTIALLEN_hHGoXQaQCOsGlZBa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hHGoXQaQCOsGlZBa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NnvJYaIuwWkuoINg(self):
        """MULTIALLEN_NnvJYaIuwWkuoINg multi-experiment creation."""
        model_folder = 'MULTIALLEN_NnvJYaIuwWkuoINg'
        exp = {
            'experiment_name': ['MULTIALLEN_NnvJYaIuwWkuoINg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NnvJYaIuwWkuoINg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yCgcFCQkBvijUgrC(self):
        """MULTIALLEN_yCgcFCQkBvijUgrC multi-experiment creation."""
        model_folder = 'MULTIALLEN_yCgcFCQkBvijUgrC'
        exp = {
            'experiment_name': ['MULTIALLEN_yCgcFCQkBvijUgrC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yCgcFCQkBvijUgrC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EPAfhareyZrHhAVz(self):
        """MULTIALLEN_EPAfhareyZrHhAVz multi-experiment creation."""
        model_folder = 'MULTIALLEN_EPAfhareyZrHhAVz'
        exp = {
            'experiment_name': ['MULTIALLEN_EPAfhareyZrHhAVz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EPAfhareyZrHhAVz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CeaNDboMFqMNoFpD(self):
        """MULTIALLEN_CeaNDboMFqMNoFpD multi-experiment creation."""
        model_folder = 'MULTIALLEN_CeaNDboMFqMNoFpD'
        exp = {
            'experiment_name': ['MULTIALLEN_CeaNDboMFqMNoFpD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CeaNDboMFqMNoFpD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FddEZEqbzeuYVJxw(self):
        """MULTIALLEN_FddEZEqbzeuYVJxw multi-experiment creation."""
        model_folder = 'MULTIALLEN_FddEZEqbzeuYVJxw'
        exp = {
            'experiment_name': ['MULTIALLEN_FddEZEqbzeuYVJxw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FddEZEqbzeuYVJxw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gubDGfXGKrZSNdNo(self):
        """MULTIALLEN_gubDGfXGKrZSNdNo multi-experiment creation."""
        model_folder = 'MULTIALLEN_gubDGfXGKrZSNdNo'
        exp = {
            'experiment_name': ['MULTIALLEN_gubDGfXGKrZSNdNo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gubDGfXGKrZSNdNo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bPPqgNiIyGdNHmvk(self):
        """MULTIALLEN_bPPqgNiIyGdNHmvk multi-experiment creation."""
        model_folder = 'MULTIALLEN_bPPqgNiIyGdNHmvk'
        exp = {
            'experiment_name': ['MULTIALLEN_bPPqgNiIyGdNHmvk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bPPqgNiIyGdNHmvk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tkTYPyDNOtyMCSCD(self):
        """MULTIALLEN_tkTYPyDNOtyMCSCD multi-experiment creation."""
        model_folder = 'MULTIALLEN_tkTYPyDNOtyMCSCD'
        exp = {
            'experiment_name': ['MULTIALLEN_tkTYPyDNOtyMCSCD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tkTYPyDNOtyMCSCD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NeVwoPmDlozHLlzA(self):
        """MULTIALLEN_NeVwoPmDlozHLlzA multi-experiment creation."""
        model_folder = 'MULTIALLEN_NeVwoPmDlozHLlzA'
        exp = {
            'experiment_name': ['MULTIALLEN_NeVwoPmDlozHLlzA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NeVwoPmDlozHLlzA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AkbOGvdPVmSRZgKZ(self):
        """MULTIALLEN_AkbOGvdPVmSRZgKZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_AkbOGvdPVmSRZgKZ'
        exp = {
            'experiment_name': ['MULTIALLEN_AkbOGvdPVmSRZgKZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AkbOGvdPVmSRZgKZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fldKcyqQlxraUNNn(self):
        """MULTIALLEN_fldKcyqQlxraUNNn multi-experiment creation."""
        model_folder = 'MULTIALLEN_fldKcyqQlxraUNNn'
        exp = {
            'experiment_name': ['MULTIALLEN_fldKcyqQlxraUNNn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fldKcyqQlxraUNNn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EnbbVdWTTpJEtomP(self):
        """MULTIALLEN_EnbbVdWTTpJEtomP multi-experiment creation."""
        model_folder = 'MULTIALLEN_EnbbVdWTTpJEtomP'
        exp = {
            'experiment_name': ['MULTIALLEN_EnbbVdWTTpJEtomP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EnbbVdWTTpJEtomP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ChIUBzkLmtebIsge(self):
        """MULTIALLEN_ChIUBzkLmtebIsge multi-experiment creation."""
        model_folder = 'MULTIALLEN_ChIUBzkLmtebIsge'
        exp = {
            'experiment_name': ['MULTIALLEN_ChIUBzkLmtebIsge'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ChIUBzkLmtebIsge']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SHfrFFlvDpHtAbuW(self):
        """MULTIALLEN_SHfrFFlvDpHtAbuW multi-experiment creation."""
        model_folder = 'MULTIALLEN_SHfrFFlvDpHtAbuW'
        exp = {
            'experiment_name': ['MULTIALLEN_SHfrFFlvDpHtAbuW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SHfrFFlvDpHtAbuW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yzUKGzOyvCkMZHKS(self):
        """MULTIALLEN_yzUKGzOyvCkMZHKS multi-experiment creation."""
        model_folder = 'MULTIALLEN_yzUKGzOyvCkMZHKS'
        exp = {
            'experiment_name': ['MULTIALLEN_yzUKGzOyvCkMZHKS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yzUKGzOyvCkMZHKS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QHOaabLxqExJsGel(self):
        """MULTIALLEN_QHOaabLxqExJsGel multi-experiment creation."""
        model_folder = 'MULTIALLEN_QHOaabLxqExJsGel'
        exp = {
            'experiment_name': ['MULTIALLEN_QHOaabLxqExJsGel'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QHOaabLxqExJsGel']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_prXOiFbsiUJPyYSu(self):
        """MULTIALLEN_prXOiFbsiUJPyYSu multi-experiment creation."""
        model_folder = 'MULTIALLEN_prXOiFbsiUJPyYSu'
        exp = {
            'experiment_name': ['MULTIALLEN_prXOiFbsiUJPyYSu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_prXOiFbsiUJPyYSu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oBWGgMXWxdrjbVtt(self):
        """MULTIALLEN_oBWGgMXWxdrjbVtt multi-experiment creation."""
        model_folder = 'MULTIALLEN_oBWGgMXWxdrjbVtt'
        exp = {
            'experiment_name': ['MULTIALLEN_oBWGgMXWxdrjbVtt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oBWGgMXWxdrjbVtt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ezOJnEdDdTTiOddB(self):
        """MULTIALLEN_ezOJnEdDdTTiOddB multi-experiment creation."""
        model_folder = 'MULTIALLEN_ezOJnEdDdTTiOddB'
        exp = {
            'experiment_name': ['MULTIALLEN_ezOJnEdDdTTiOddB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ezOJnEdDdTTiOddB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RJHKOHGCOStYsVlW(self):
        """MULTIALLEN_RJHKOHGCOStYsVlW multi-experiment creation."""
        model_folder = 'MULTIALLEN_RJHKOHGCOStYsVlW'
        exp = {
            'experiment_name': ['MULTIALLEN_RJHKOHGCOStYsVlW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RJHKOHGCOStYsVlW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qYuQDLypEIirSpFm(self):
        """MULTIALLEN_qYuQDLypEIirSpFm multi-experiment creation."""
        model_folder = 'MULTIALLEN_qYuQDLypEIirSpFm'
        exp = {
            'experiment_name': ['MULTIALLEN_qYuQDLypEIirSpFm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qYuQDLypEIirSpFm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gbdWHnJaujCMmYdk(self):
        """MULTIALLEN_gbdWHnJaujCMmYdk multi-experiment creation."""
        model_folder = 'MULTIALLEN_gbdWHnJaujCMmYdk'
        exp = {
            'experiment_name': ['MULTIALLEN_gbdWHnJaujCMmYdk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gbdWHnJaujCMmYdk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DsitFKDMbeAoJmbP(self):
        """MULTIALLEN_DsitFKDMbeAoJmbP multi-experiment creation."""
        model_folder = 'MULTIALLEN_DsitFKDMbeAoJmbP'
        exp = {
            'experiment_name': ['MULTIALLEN_DsitFKDMbeAoJmbP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DsitFKDMbeAoJmbP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YntSvyaykFHDPKoM(self):
        """MULTIALLEN_YntSvyaykFHDPKoM multi-experiment creation."""
        model_folder = 'MULTIALLEN_YntSvyaykFHDPKoM'
        exp = {
            'experiment_name': ['MULTIALLEN_YntSvyaykFHDPKoM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YntSvyaykFHDPKoM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PJPDQrtanHNeovhB(self):
        """MULTIALLEN_PJPDQrtanHNeovhB multi-experiment creation."""
        model_folder = 'MULTIALLEN_PJPDQrtanHNeovhB'
        exp = {
            'experiment_name': ['MULTIALLEN_PJPDQrtanHNeovhB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PJPDQrtanHNeovhB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kMGVcPtCYePxWfcc(self):
        """MULTIALLEN_kMGVcPtCYePxWfcc multi-experiment creation."""
        model_folder = 'MULTIALLEN_kMGVcPtCYePxWfcc'
        exp = {
            'experiment_name': ['MULTIALLEN_kMGVcPtCYePxWfcc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kMGVcPtCYePxWfcc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TMKFOxkYauljtJny(self):
        """MULTIALLEN_TMKFOxkYauljtJny multi-experiment creation."""
        model_folder = 'MULTIALLEN_TMKFOxkYauljtJny'
        exp = {
            'experiment_name': ['MULTIALLEN_TMKFOxkYauljtJny'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TMKFOxkYauljtJny']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nDyxGJvBFLldCttf(self):
        """MULTIALLEN_nDyxGJvBFLldCttf multi-experiment creation."""
        model_folder = 'MULTIALLEN_nDyxGJvBFLldCttf'
        exp = {
            'experiment_name': ['MULTIALLEN_nDyxGJvBFLldCttf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nDyxGJvBFLldCttf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NIZTboHoGOCdsTDY(self):
        """MULTIALLEN_NIZTboHoGOCdsTDY multi-experiment creation."""
        model_folder = 'MULTIALLEN_NIZTboHoGOCdsTDY'
        exp = {
            'experiment_name': ['MULTIALLEN_NIZTboHoGOCdsTDY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NIZTboHoGOCdsTDY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xbOdsWdbLGXhuoMr(self):
        """MULTIALLEN_xbOdsWdbLGXhuoMr multi-experiment creation."""
        model_folder = 'MULTIALLEN_xbOdsWdbLGXhuoMr'
        exp = {
            'experiment_name': ['MULTIALLEN_xbOdsWdbLGXhuoMr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xbOdsWdbLGXhuoMr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MkXSjVSzjScQNuXn(self):
        """MULTIALLEN_MkXSjVSzjScQNuXn multi-experiment creation."""
        model_folder = 'MULTIALLEN_MkXSjVSzjScQNuXn'
        exp = {
            'experiment_name': ['MULTIALLEN_MkXSjVSzjScQNuXn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MkXSjVSzjScQNuXn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bmbuRhSyncRUbDbk(self):
        """MULTIALLEN_bmbuRhSyncRUbDbk multi-experiment creation."""
        model_folder = 'MULTIALLEN_bmbuRhSyncRUbDbk'
        exp = {
            'experiment_name': ['MULTIALLEN_bmbuRhSyncRUbDbk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bmbuRhSyncRUbDbk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BYkxfDuGzSBGBcEQ(self):
        """MULTIALLEN_BYkxfDuGzSBGBcEQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_BYkxfDuGzSBGBcEQ'
        exp = {
            'experiment_name': ['MULTIALLEN_BYkxfDuGzSBGBcEQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BYkxfDuGzSBGBcEQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ugagLyjQsUOEGBSJ(self):
        """MULTIALLEN_ugagLyjQsUOEGBSJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_ugagLyjQsUOEGBSJ'
        exp = {
            'experiment_name': ['MULTIALLEN_ugagLyjQsUOEGBSJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ugagLyjQsUOEGBSJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wqPOpiLNoGDqjaIZ(self):
        """MULTIALLEN_wqPOpiLNoGDqjaIZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_wqPOpiLNoGDqjaIZ'
        exp = {
            'experiment_name': ['MULTIALLEN_wqPOpiLNoGDqjaIZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wqPOpiLNoGDqjaIZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lJFPYSpfiecOgXRX(self):
        """MULTIALLEN_lJFPYSpfiecOgXRX multi-experiment creation."""
        model_folder = 'MULTIALLEN_lJFPYSpfiecOgXRX'
        exp = {
            'experiment_name': ['MULTIALLEN_lJFPYSpfiecOgXRX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lJFPYSpfiecOgXRX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ftydzVpiPGdpXJqP(self):
        """MULTIALLEN_ftydzVpiPGdpXJqP multi-experiment creation."""
        model_folder = 'MULTIALLEN_ftydzVpiPGdpXJqP'
        exp = {
            'experiment_name': ['MULTIALLEN_ftydzVpiPGdpXJqP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ftydzVpiPGdpXJqP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OibtGarSWcdxnImF(self):
        """MULTIALLEN_OibtGarSWcdxnImF multi-experiment creation."""
        model_folder = 'MULTIALLEN_OibtGarSWcdxnImF'
        exp = {
            'experiment_name': ['MULTIALLEN_OibtGarSWcdxnImF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OibtGarSWcdxnImF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LWIrFMWSXfUywwZL(self):
        """MULTIALLEN_LWIrFMWSXfUywwZL multi-experiment creation."""
        model_folder = 'MULTIALLEN_LWIrFMWSXfUywwZL'
        exp = {
            'experiment_name': ['MULTIALLEN_LWIrFMWSXfUywwZL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LWIrFMWSXfUywwZL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MxPxSSwQyzERaNfD(self):
        """MULTIALLEN_MxPxSSwQyzERaNfD multi-experiment creation."""
        model_folder = 'MULTIALLEN_MxPxSSwQyzERaNfD'
        exp = {
            'experiment_name': ['MULTIALLEN_MxPxSSwQyzERaNfD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MxPxSSwQyzERaNfD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ewZAdTcxWuUYmyGU(self):
        """MULTIALLEN_ewZAdTcxWuUYmyGU multi-experiment creation."""
        model_folder = 'MULTIALLEN_ewZAdTcxWuUYmyGU'
        exp = {
            'experiment_name': ['MULTIALLEN_ewZAdTcxWuUYmyGU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ewZAdTcxWuUYmyGU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZVqYhHEkdmHrcEJH(self):
        """MULTIALLEN_ZVqYhHEkdmHrcEJH multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZVqYhHEkdmHrcEJH'
        exp = {
            'experiment_name': ['MULTIALLEN_ZVqYhHEkdmHrcEJH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZVqYhHEkdmHrcEJH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gVaHBwiKlfAAxfsD(self):
        """MULTIALLEN_gVaHBwiKlfAAxfsD multi-experiment creation."""
        model_folder = 'MULTIALLEN_gVaHBwiKlfAAxfsD'
        exp = {
            'experiment_name': ['MULTIALLEN_gVaHBwiKlfAAxfsD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gVaHBwiKlfAAxfsD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EswyTbgrsZhxVXMp(self):
        """MULTIALLEN_EswyTbgrsZhxVXMp multi-experiment creation."""
        model_folder = 'MULTIALLEN_EswyTbgrsZhxVXMp'
        exp = {
            'experiment_name': ['MULTIALLEN_EswyTbgrsZhxVXMp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EswyTbgrsZhxVXMp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VFkBYBamEujsiCbw(self):
        """MULTIALLEN_VFkBYBamEujsiCbw multi-experiment creation."""
        model_folder = 'MULTIALLEN_VFkBYBamEujsiCbw'
        exp = {
            'experiment_name': ['MULTIALLEN_VFkBYBamEujsiCbw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VFkBYBamEujsiCbw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pCnAerrCwXHbjCmP(self):
        """MULTIALLEN_pCnAerrCwXHbjCmP multi-experiment creation."""
        model_folder = 'MULTIALLEN_pCnAerrCwXHbjCmP'
        exp = {
            'experiment_name': ['MULTIALLEN_pCnAerrCwXHbjCmP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pCnAerrCwXHbjCmP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nzrKFjuliXPkyeRQ(self):
        """MULTIALLEN_nzrKFjuliXPkyeRQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_nzrKFjuliXPkyeRQ'
        exp = {
            'experiment_name': ['MULTIALLEN_nzrKFjuliXPkyeRQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nzrKFjuliXPkyeRQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yhRYapxoUwIjSoml(self):
        """MULTIALLEN_yhRYapxoUwIjSoml multi-experiment creation."""
        model_folder = 'MULTIALLEN_yhRYapxoUwIjSoml'
        exp = {
            'experiment_name': ['MULTIALLEN_yhRYapxoUwIjSoml'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yhRYapxoUwIjSoml']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KXcTjWpsICnsvnfM(self):
        """MULTIALLEN_KXcTjWpsICnsvnfM multi-experiment creation."""
        model_folder = 'MULTIALLEN_KXcTjWpsICnsvnfM'
        exp = {
            'experiment_name': ['MULTIALLEN_KXcTjWpsICnsvnfM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KXcTjWpsICnsvnfM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NEJtVXheHrBovlxT(self):
        """MULTIALLEN_NEJtVXheHrBovlxT multi-experiment creation."""
        model_folder = 'MULTIALLEN_NEJtVXheHrBovlxT'
        exp = {
            'experiment_name': ['MULTIALLEN_NEJtVXheHrBovlxT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NEJtVXheHrBovlxT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nhcZbCEIWOlurIUx(self):
        """MULTIALLEN_nhcZbCEIWOlurIUx multi-experiment creation."""
        model_folder = 'MULTIALLEN_nhcZbCEIWOlurIUx'
        exp = {
            'experiment_name': ['MULTIALLEN_nhcZbCEIWOlurIUx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nhcZbCEIWOlurIUx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ypQIduSCjIRiMawr(self):
        """MULTIALLEN_ypQIduSCjIRiMawr multi-experiment creation."""
        model_folder = 'MULTIALLEN_ypQIduSCjIRiMawr'
        exp = {
            'experiment_name': ['MULTIALLEN_ypQIduSCjIRiMawr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ypQIduSCjIRiMawr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HUbgikbQQdZXMzhz(self):
        """MULTIALLEN_HUbgikbQQdZXMzhz multi-experiment creation."""
        model_folder = 'MULTIALLEN_HUbgikbQQdZXMzhz'
        exp = {
            'experiment_name': ['MULTIALLEN_HUbgikbQQdZXMzhz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HUbgikbQQdZXMzhz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XCTJaBwArGjEGYSr(self):
        """MULTIALLEN_XCTJaBwArGjEGYSr multi-experiment creation."""
        model_folder = 'MULTIALLEN_XCTJaBwArGjEGYSr'
        exp = {
            'experiment_name': ['MULTIALLEN_XCTJaBwArGjEGYSr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XCTJaBwArGjEGYSr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HqDPYkcyHfEVjwDY(self):
        """MULTIALLEN_HqDPYkcyHfEVjwDY multi-experiment creation."""
        model_folder = 'MULTIALLEN_HqDPYkcyHfEVjwDY'
        exp = {
            'experiment_name': ['MULTIALLEN_HqDPYkcyHfEVjwDY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HqDPYkcyHfEVjwDY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CUrugQAAeJKVDWZO(self):
        """MULTIALLEN_CUrugQAAeJKVDWZO multi-experiment creation."""
        model_folder = 'MULTIALLEN_CUrugQAAeJKVDWZO'
        exp = {
            'experiment_name': ['MULTIALLEN_CUrugQAAeJKVDWZO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CUrugQAAeJKVDWZO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FXnRpFzhwfHROYzI(self):
        """MULTIALLEN_FXnRpFzhwfHROYzI multi-experiment creation."""
        model_folder = 'MULTIALLEN_FXnRpFzhwfHROYzI'
        exp = {
            'experiment_name': ['MULTIALLEN_FXnRpFzhwfHROYzI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FXnRpFzhwfHROYzI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_koTBazdKCwhPIEDI(self):
        """MULTIALLEN_koTBazdKCwhPIEDI multi-experiment creation."""
        model_folder = 'MULTIALLEN_koTBazdKCwhPIEDI'
        exp = {
            'experiment_name': ['MULTIALLEN_koTBazdKCwhPIEDI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_koTBazdKCwhPIEDI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cgVefHZwWSMHgFyP(self):
        """MULTIALLEN_cgVefHZwWSMHgFyP multi-experiment creation."""
        model_folder = 'MULTIALLEN_cgVefHZwWSMHgFyP'
        exp = {
            'experiment_name': ['MULTIALLEN_cgVefHZwWSMHgFyP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cgVefHZwWSMHgFyP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kLhmhRXfTmhIMNnR(self):
        """MULTIALLEN_kLhmhRXfTmhIMNnR multi-experiment creation."""
        model_folder = 'MULTIALLEN_kLhmhRXfTmhIMNnR'
        exp = {
            'experiment_name': ['MULTIALLEN_kLhmhRXfTmhIMNnR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kLhmhRXfTmhIMNnR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ncdggclQXFvbDrhL(self):
        """MULTIALLEN_ncdggclQXFvbDrhL multi-experiment creation."""
        model_folder = 'MULTIALLEN_ncdggclQXFvbDrhL'
        exp = {
            'experiment_name': ['MULTIALLEN_ncdggclQXFvbDrhL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ncdggclQXFvbDrhL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cKOdGhYGTDChelED(self):
        """MULTIALLEN_cKOdGhYGTDChelED multi-experiment creation."""
        model_folder = 'MULTIALLEN_cKOdGhYGTDChelED'
        exp = {
            'experiment_name': ['MULTIALLEN_cKOdGhYGTDChelED'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cKOdGhYGTDChelED']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uxmToikpCDixDPCh(self):
        """MULTIALLEN_uxmToikpCDixDPCh multi-experiment creation."""
        model_folder = 'MULTIALLEN_uxmToikpCDixDPCh'
        exp = {
            'experiment_name': ['MULTIALLEN_uxmToikpCDixDPCh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uxmToikpCDixDPCh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qYwGrXpcMktyiatE(self):
        """MULTIALLEN_qYwGrXpcMktyiatE multi-experiment creation."""
        model_folder = 'MULTIALLEN_qYwGrXpcMktyiatE'
        exp = {
            'experiment_name': ['MULTIALLEN_qYwGrXpcMktyiatE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qYwGrXpcMktyiatE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UKhwFpurycezEIAH(self):
        """MULTIALLEN_UKhwFpurycezEIAH multi-experiment creation."""
        model_folder = 'MULTIALLEN_UKhwFpurycezEIAH'
        exp = {
            'experiment_name': ['MULTIALLEN_UKhwFpurycezEIAH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UKhwFpurycezEIAH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fwaXSNDNcferkBPc(self):
        """MULTIALLEN_fwaXSNDNcferkBPc multi-experiment creation."""
        model_folder = 'MULTIALLEN_fwaXSNDNcferkBPc'
        exp = {
            'experiment_name': ['MULTIALLEN_fwaXSNDNcferkBPc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fwaXSNDNcferkBPc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uQvhscmpLeMQUJaK(self):
        """MULTIALLEN_uQvhscmpLeMQUJaK multi-experiment creation."""
        model_folder = 'MULTIALLEN_uQvhscmpLeMQUJaK'
        exp = {
            'experiment_name': ['MULTIALLEN_uQvhscmpLeMQUJaK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uQvhscmpLeMQUJaK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bGEROHiZPasqKRSI(self):
        """MULTIALLEN_bGEROHiZPasqKRSI multi-experiment creation."""
        model_folder = 'MULTIALLEN_bGEROHiZPasqKRSI'
        exp = {
            'experiment_name': ['MULTIALLEN_bGEROHiZPasqKRSI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bGEROHiZPasqKRSI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TeKIqhvpXvVEoepJ(self):
        """MULTIALLEN_TeKIqhvpXvVEoepJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_TeKIqhvpXvVEoepJ'
        exp = {
            'experiment_name': ['MULTIALLEN_TeKIqhvpXvVEoepJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TeKIqhvpXvVEoepJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yDpebfIlyMJEoDTP(self):
        """MULTIALLEN_yDpebfIlyMJEoDTP multi-experiment creation."""
        model_folder = 'MULTIALLEN_yDpebfIlyMJEoDTP'
        exp = {
            'experiment_name': ['MULTIALLEN_yDpebfIlyMJEoDTP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yDpebfIlyMJEoDTP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FSilCvoTuimvjXik(self):
        """MULTIALLEN_FSilCvoTuimvjXik multi-experiment creation."""
        model_folder = 'MULTIALLEN_FSilCvoTuimvjXik'
        exp = {
            'experiment_name': ['MULTIALLEN_FSilCvoTuimvjXik'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FSilCvoTuimvjXik']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zFMEOjBEcQcLbYAN(self):
        """MULTIALLEN_zFMEOjBEcQcLbYAN multi-experiment creation."""
        model_folder = 'MULTIALLEN_zFMEOjBEcQcLbYAN'
        exp = {
            'experiment_name': ['MULTIALLEN_zFMEOjBEcQcLbYAN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zFMEOjBEcQcLbYAN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eceyBtkeFGANqYIM(self):
        """MULTIALLEN_eceyBtkeFGANqYIM multi-experiment creation."""
        model_folder = 'MULTIALLEN_eceyBtkeFGANqYIM'
        exp = {
            'experiment_name': ['MULTIALLEN_eceyBtkeFGANqYIM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eceyBtkeFGANqYIM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CfAHBfCkQDKPtsFe(self):
        """MULTIALLEN_CfAHBfCkQDKPtsFe multi-experiment creation."""
        model_folder = 'MULTIALLEN_CfAHBfCkQDKPtsFe'
        exp = {
            'experiment_name': ['MULTIALLEN_CfAHBfCkQDKPtsFe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CfAHBfCkQDKPtsFe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fNrylvNhLGhChoQO(self):
        """MULTIALLEN_fNrylvNhLGhChoQO multi-experiment creation."""
        model_folder = 'MULTIALLEN_fNrylvNhLGhChoQO'
        exp = {
            'experiment_name': ['MULTIALLEN_fNrylvNhLGhChoQO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fNrylvNhLGhChoQO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lPiAysRSNjedYyHc(self):
        """MULTIALLEN_lPiAysRSNjedYyHc multi-experiment creation."""
        model_folder = 'MULTIALLEN_lPiAysRSNjedYyHc'
        exp = {
            'experiment_name': ['MULTIALLEN_lPiAysRSNjedYyHc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lPiAysRSNjedYyHc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CbzgZHPrRqAnxgbl(self):
        """MULTIALLEN_CbzgZHPrRqAnxgbl multi-experiment creation."""
        model_folder = 'MULTIALLEN_CbzgZHPrRqAnxgbl'
        exp = {
            'experiment_name': ['MULTIALLEN_CbzgZHPrRqAnxgbl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CbzgZHPrRqAnxgbl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BQoXEfjckAaJkwfd(self):
        """MULTIALLEN_BQoXEfjckAaJkwfd multi-experiment creation."""
        model_folder = 'MULTIALLEN_BQoXEfjckAaJkwfd'
        exp = {
            'experiment_name': ['MULTIALLEN_BQoXEfjckAaJkwfd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BQoXEfjckAaJkwfd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uiNhvErJfeZtiMba(self):
        """MULTIALLEN_uiNhvErJfeZtiMba multi-experiment creation."""
        model_folder = 'MULTIALLEN_uiNhvErJfeZtiMba'
        exp = {
            'experiment_name': ['MULTIALLEN_uiNhvErJfeZtiMba'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uiNhvErJfeZtiMba']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nkoBWjIjhmacWbEs(self):
        """MULTIALLEN_nkoBWjIjhmacWbEs multi-experiment creation."""
        model_folder = 'MULTIALLEN_nkoBWjIjhmacWbEs'
        exp = {
            'experiment_name': ['MULTIALLEN_nkoBWjIjhmacWbEs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nkoBWjIjhmacWbEs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RCNNDWSctyAeEYtr(self):
        """MULTIALLEN_RCNNDWSctyAeEYtr multi-experiment creation."""
        model_folder = 'MULTIALLEN_RCNNDWSctyAeEYtr'
        exp = {
            'experiment_name': ['MULTIALLEN_RCNNDWSctyAeEYtr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RCNNDWSctyAeEYtr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kSZPlgHFFFqKCeaY(self):
        """MULTIALLEN_kSZPlgHFFFqKCeaY multi-experiment creation."""
        model_folder = 'MULTIALLEN_kSZPlgHFFFqKCeaY'
        exp = {
            'experiment_name': ['MULTIALLEN_kSZPlgHFFFqKCeaY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kSZPlgHFFFqKCeaY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OWcVIysSzXpbLNIT(self):
        """MULTIALLEN_OWcVIysSzXpbLNIT multi-experiment creation."""
        model_folder = 'MULTIALLEN_OWcVIysSzXpbLNIT'
        exp = {
            'experiment_name': ['MULTIALLEN_OWcVIysSzXpbLNIT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OWcVIysSzXpbLNIT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CIPJFiVHItbtuWeW(self):
        """MULTIALLEN_CIPJFiVHItbtuWeW multi-experiment creation."""
        model_folder = 'MULTIALLEN_CIPJFiVHItbtuWeW'
        exp = {
            'experiment_name': ['MULTIALLEN_CIPJFiVHItbtuWeW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CIPJFiVHItbtuWeW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VIzONiohPhsFZVpH(self):
        """MULTIALLEN_VIzONiohPhsFZVpH multi-experiment creation."""
        model_folder = 'MULTIALLEN_VIzONiohPhsFZVpH'
        exp = {
            'experiment_name': ['MULTIALLEN_VIzONiohPhsFZVpH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VIzONiohPhsFZVpH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mAZxFdOLwHpSvcbx(self):
        """MULTIALLEN_mAZxFdOLwHpSvcbx multi-experiment creation."""
        model_folder = 'MULTIALLEN_mAZxFdOLwHpSvcbx'
        exp = {
            'experiment_name': ['MULTIALLEN_mAZxFdOLwHpSvcbx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mAZxFdOLwHpSvcbx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QDTrzZTOHIEuEsOw(self):
        """MULTIALLEN_QDTrzZTOHIEuEsOw multi-experiment creation."""
        model_folder = 'MULTIALLEN_QDTrzZTOHIEuEsOw'
        exp = {
            'experiment_name': ['MULTIALLEN_QDTrzZTOHIEuEsOw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QDTrzZTOHIEuEsOw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dbOOrNSaavjGmjXF(self):
        """MULTIALLEN_dbOOrNSaavjGmjXF multi-experiment creation."""
        model_folder = 'MULTIALLEN_dbOOrNSaavjGmjXF'
        exp = {
            'experiment_name': ['MULTIALLEN_dbOOrNSaavjGmjXF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dbOOrNSaavjGmjXF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EAReepKnGPkLhobQ(self):
        """MULTIALLEN_EAReepKnGPkLhobQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_EAReepKnGPkLhobQ'
        exp = {
            'experiment_name': ['MULTIALLEN_EAReepKnGPkLhobQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EAReepKnGPkLhobQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_inUzMGYpxlvvOnuq(self):
        """MULTIALLEN_inUzMGYpxlvvOnuq multi-experiment creation."""
        model_folder = 'MULTIALLEN_inUzMGYpxlvvOnuq'
        exp = {
            'experiment_name': ['MULTIALLEN_inUzMGYpxlvvOnuq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_inUzMGYpxlvvOnuq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pMxBbkbcorttYwHU(self):
        """MULTIALLEN_pMxBbkbcorttYwHU multi-experiment creation."""
        model_folder = 'MULTIALLEN_pMxBbkbcorttYwHU'
        exp = {
            'experiment_name': ['MULTIALLEN_pMxBbkbcorttYwHU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pMxBbkbcorttYwHU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UzxYNFOQoIjxQyiS(self):
        """MULTIALLEN_UzxYNFOQoIjxQyiS multi-experiment creation."""
        model_folder = 'MULTIALLEN_UzxYNFOQoIjxQyiS'
        exp = {
            'experiment_name': ['MULTIALLEN_UzxYNFOQoIjxQyiS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UzxYNFOQoIjxQyiS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TvZmWEnZLMeqtyDg(self):
        """MULTIALLEN_TvZmWEnZLMeqtyDg multi-experiment creation."""
        model_folder = 'MULTIALLEN_TvZmWEnZLMeqtyDg'
        exp = {
            'experiment_name': ['MULTIALLEN_TvZmWEnZLMeqtyDg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TvZmWEnZLMeqtyDg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DdDvyRLOvYmpftFr(self):
        """MULTIALLEN_DdDvyRLOvYmpftFr multi-experiment creation."""
        model_folder = 'MULTIALLEN_DdDvyRLOvYmpftFr'
        exp = {
            'experiment_name': ['MULTIALLEN_DdDvyRLOvYmpftFr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DdDvyRLOvYmpftFr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WkBVuIzCUGMZOXKi(self):
        """MULTIALLEN_WkBVuIzCUGMZOXKi multi-experiment creation."""
        model_folder = 'MULTIALLEN_WkBVuIzCUGMZOXKi'
        exp = {
            'experiment_name': ['MULTIALLEN_WkBVuIzCUGMZOXKi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WkBVuIzCUGMZOXKi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oWpqNrrufKqAtqBg(self):
        """MULTIALLEN_oWpqNrrufKqAtqBg multi-experiment creation."""
        model_folder = 'MULTIALLEN_oWpqNrrufKqAtqBg'
        exp = {
            'experiment_name': ['MULTIALLEN_oWpqNrrufKqAtqBg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oWpqNrrufKqAtqBg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AmJBBdriknNLCpBf(self):
        """MULTIALLEN_AmJBBdriknNLCpBf multi-experiment creation."""
        model_folder = 'MULTIALLEN_AmJBBdriknNLCpBf'
        exp = {
            'experiment_name': ['MULTIALLEN_AmJBBdriknNLCpBf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AmJBBdriknNLCpBf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wBRQXTNEwYXljOsF(self):
        """MULTIALLEN_wBRQXTNEwYXljOsF multi-experiment creation."""
        model_folder = 'MULTIALLEN_wBRQXTNEwYXljOsF'
        exp = {
            'experiment_name': ['MULTIALLEN_wBRQXTNEwYXljOsF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wBRQXTNEwYXljOsF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qieYqZHTUqUAqoOg(self):
        """MULTIALLEN_qieYqZHTUqUAqoOg multi-experiment creation."""
        model_folder = 'MULTIALLEN_qieYqZHTUqUAqoOg'
        exp = {
            'experiment_name': ['MULTIALLEN_qieYqZHTUqUAqoOg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qieYqZHTUqUAqoOg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tTRSoViVJivOyhjP(self):
        """MULTIALLEN_tTRSoViVJivOyhjP multi-experiment creation."""
        model_folder = 'MULTIALLEN_tTRSoViVJivOyhjP'
        exp = {
            'experiment_name': ['MULTIALLEN_tTRSoViVJivOyhjP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tTRSoViVJivOyhjP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rvGONcHuhLvtCReZ(self):
        """MULTIALLEN_rvGONcHuhLvtCReZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_rvGONcHuhLvtCReZ'
        exp = {
            'experiment_name': ['MULTIALLEN_rvGONcHuhLvtCReZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rvGONcHuhLvtCReZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tdQwUbZhFAvphxQO(self):
        """MULTIALLEN_tdQwUbZhFAvphxQO multi-experiment creation."""
        model_folder = 'MULTIALLEN_tdQwUbZhFAvphxQO'
        exp = {
            'experiment_name': ['MULTIALLEN_tdQwUbZhFAvphxQO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tdQwUbZhFAvphxQO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GmEKsaMjCFfkCjFG(self):
        """MULTIALLEN_GmEKsaMjCFfkCjFG multi-experiment creation."""
        model_folder = 'MULTIALLEN_GmEKsaMjCFfkCjFG'
        exp = {
            'experiment_name': ['MULTIALLEN_GmEKsaMjCFfkCjFG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GmEKsaMjCFfkCjFG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NRBiDVlayMoZkikH(self):
        """MULTIALLEN_NRBiDVlayMoZkikH multi-experiment creation."""
        model_folder = 'MULTIALLEN_NRBiDVlayMoZkikH'
        exp = {
            'experiment_name': ['MULTIALLEN_NRBiDVlayMoZkikH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NRBiDVlayMoZkikH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sFcOuVcYBuVdNpdi(self):
        """MULTIALLEN_sFcOuVcYBuVdNpdi multi-experiment creation."""
        model_folder = 'MULTIALLEN_sFcOuVcYBuVdNpdi'
        exp = {
            'experiment_name': ['MULTIALLEN_sFcOuVcYBuVdNpdi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sFcOuVcYBuVdNpdi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FvTxiyHqeRvbkrwX(self):
        """MULTIALLEN_FvTxiyHqeRvbkrwX multi-experiment creation."""
        model_folder = 'MULTIALLEN_FvTxiyHqeRvbkrwX'
        exp = {
            'experiment_name': ['MULTIALLEN_FvTxiyHqeRvbkrwX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FvTxiyHqeRvbkrwX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dLYfOELbkwRvPxOY(self):
        """MULTIALLEN_dLYfOELbkwRvPxOY multi-experiment creation."""
        model_folder = 'MULTIALLEN_dLYfOELbkwRvPxOY'
        exp = {
            'experiment_name': ['MULTIALLEN_dLYfOELbkwRvPxOY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dLYfOELbkwRvPxOY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RuviiKuoBCzcXbIR(self):
        """MULTIALLEN_RuviiKuoBCzcXbIR multi-experiment creation."""
        model_folder = 'MULTIALLEN_RuviiKuoBCzcXbIR'
        exp = {
            'experiment_name': ['MULTIALLEN_RuviiKuoBCzcXbIR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RuviiKuoBCzcXbIR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VSNcQdDTtJFMNNUy(self):
        """MULTIALLEN_VSNcQdDTtJFMNNUy multi-experiment creation."""
        model_folder = 'MULTIALLEN_VSNcQdDTtJFMNNUy'
        exp = {
            'experiment_name': ['MULTIALLEN_VSNcQdDTtJFMNNUy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VSNcQdDTtJFMNNUy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SEdpErshlEyoVpDT(self):
        """MULTIALLEN_SEdpErshlEyoVpDT multi-experiment creation."""
        model_folder = 'MULTIALLEN_SEdpErshlEyoVpDT'
        exp = {
            'experiment_name': ['MULTIALLEN_SEdpErshlEyoVpDT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SEdpErshlEyoVpDT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NWzUOVJXZGuTClZZ(self):
        """MULTIALLEN_NWzUOVJXZGuTClZZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_NWzUOVJXZGuTClZZ'
        exp = {
            'experiment_name': ['MULTIALLEN_NWzUOVJXZGuTClZZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NWzUOVJXZGuTClZZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oGSExNLQfgLSxRTX(self):
        """MULTIALLEN_oGSExNLQfgLSxRTX multi-experiment creation."""
        model_folder = 'MULTIALLEN_oGSExNLQfgLSxRTX'
        exp = {
            'experiment_name': ['MULTIALLEN_oGSExNLQfgLSxRTX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oGSExNLQfgLSxRTX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lnChqdwXxvuRGImq(self):
        """MULTIALLEN_lnChqdwXxvuRGImq multi-experiment creation."""
        model_folder = 'MULTIALLEN_lnChqdwXxvuRGImq'
        exp = {
            'experiment_name': ['MULTIALLEN_lnChqdwXxvuRGImq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lnChqdwXxvuRGImq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EajlEWpmREZbVGJr(self):
        """MULTIALLEN_EajlEWpmREZbVGJr multi-experiment creation."""
        model_folder = 'MULTIALLEN_EajlEWpmREZbVGJr'
        exp = {
            'experiment_name': ['MULTIALLEN_EajlEWpmREZbVGJr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EajlEWpmREZbVGJr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TIEHQEVoFLIzukJy(self):
        """MULTIALLEN_TIEHQEVoFLIzukJy multi-experiment creation."""
        model_folder = 'MULTIALLEN_TIEHQEVoFLIzukJy'
        exp = {
            'experiment_name': ['MULTIALLEN_TIEHQEVoFLIzukJy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TIEHQEVoFLIzukJy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_usscFlSrOlOqdkFZ(self):
        """MULTIALLEN_usscFlSrOlOqdkFZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_usscFlSrOlOqdkFZ'
        exp = {
            'experiment_name': ['MULTIALLEN_usscFlSrOlOqdkFZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_usscFlSrOlOqdkFZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kKcClEZqVJJfrrDV(self):
        """MULTIALLEN_kKcClEZqVJJfrrDV multi-experiment creation."""
        model_folder = 'MULTIALLEN_kKcClEZqVJJfrrDV'
        exp = {
            'experiment_name': ['MULTIALLEN_kKcClEZqVJJfrrDV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kKcClEZqVJJfrrDV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qXuncTHaXLoSRHcf(self):
        """MULTIALLEN_qXuncTHaXLoSRHcf multi-experiment creation."""
        model_folder = 'MULTIALLEN_qXuncTHaXLoSRHcf'
        exp = {
            'experiment_name': ['MULTIALLEN_qXuncTHaXLoSRHcf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qXuncTHaXLoSRHcf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XhTyaYtmJAUNtmuT(self):
        """MULTIALLEN_XhTyaYtmJAUNtmuT multi-experiment creation."""
        model_folder = 'MULTIALLEN_XhTyaYtmJAUNtmuT'
        exp = {
            'experiment_name': ['MULTIALLEN_XhTyaYtmJAUNtmuT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XhTyaYtmJAUNtmuT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ebBzlNvscHMbmLFg(self):
        """MULTIALLEN_ebBzlNvscHMbmLFg multi-experiment creation."""
        model_folder = 'MULTIALLEN_ebBzlNvscHMbmLFg'
        exp = {
            'experiment_name': ['MULTIALLEN_ebBzlNvscHMbmLFg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ebBzlNvscHMbmLFg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EJKUGPWDIKkNOikE(self):
        """MULTIALLEN_EJKUGPWDIKkNOikE multi-experiment creation."""
        model_folder = 'MULTIALLEN_EJKUGPWDIKkNOikE'
        exp = {
            'experiment_name': ['MULTIALLEN_EJKUGPWDIKkNOikE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EJKUGPWDIKkNOikE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jWMOOvaWzSKvbgyE(self):
        """MULTIALLEN_jWMOOvaWzSKvbgyE multi-experiment creation."""
        model_folder = 'MULTIALLEN_jWMOOvaWzSKvbgyE'
        exp = {
            'experiment_name': ['MULTIALLEN_jWMOOvaWzSKvbgyE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jWMOOvaWzSKvbgyE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rrUwoiVWMWeqFnOE(self):
        """MULTIALLEN_rrUwoiVWMWeqFnOE multi-experiment creation."""
        model_folder = 'MULTIALLEN_rrUwoiVWMWeqFnOE'
        exp = {
            'experiment_name': ['MULTIALLEN_rrUwoiVWMWeqFnOE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rrUwoiVWMWeqFnOE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QQpfIfEOuueoISPx(self):
        """MULTIALLEN_QQpfIfEOuueoISPx multi-experiment creation."""
        model_folder = 'MULTIALLEN_QQpfIfEOuueoISPx'
        exp = {
            'experiment_name': ['MULTIALLEN_QQpfIfEOuueoISPx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QQpfIfEOuueoISPx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MmCctJmcafprlQuK(self):
        """MULTIALLEN_MmCctJmcafprlQuK multi-experiment creation."""
        model_folder = 'MULTIALLEN_MmCctJmcafprlQuK'
        exp = {
            'experiment_name': ['MULTIALLEN_MmCctJmcafprlQuK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MmCctJmcafprlQuK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_StUmKYxxPSZZiCTd(self):
        """MULTIALLEN_StUmKYxxPSZZiCTd multi-experiment creation."""
        model_folder = 'MULTIALLEN_StUmKYxxPSZZiCTd'
        exp = {
            'experiment_name': ['MULTIALLEN_StUmKYxxPSZZiCTd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_StUmKYxxPSZZiCTd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AoClYhVdoOxpqLIr(self):
        """MULTIALLEN_AoClYhVdoOxpqLIr multi-experiment creation."""
        model_folder = 'MULTIALLEN_AoClYhVdoOxpqLIr'
        exp = {
            'experiment_name': ['MULTIALLEN_AoClYhVdoOxpqLIr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AoClYhVdoOxpqLIr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zaNZxgSKuCrJLpvq(self):
        """MULTIALLEN_zaNZxgSKuCrJLpvq multi-experiment creation."""
        model_folder = 'MULTIALLEN_zaNZxgSKuCrJLpvq'
        exp = {
            'experiment_name': ['MULTIALLEN_zaNZxgSKuCrJLpvq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zaNZxgSKuCrJLpvq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LkxmKOksUBQvNmVy(self):
        """MULTIALLEN_LkxmKOksUBQvNmVy multi-experiment creation."""
        model_folder = 'MULTIALLEN_LkxmKOksUBQvNmVy'
        exp = {
            'experiment_name': ['MULTIALLEN_LkxmKOksUBQvNmVy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LkxmKOksUBQvNmVy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aApmxNpobVIzAtuD(self):
        """MULTIALLEN_aApmxNpobVIzAtuD multi-experiment creation."""
        model_folder = 'MULTIALLEN_aApmxNpobVIzAtuD'
        exp = {
            'experiment_name': ['MULTIALLEN_aApmxNpobVIzAtuD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aApmxNpobVIzAtuD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BzRfrFRDkOAwTPSa(self):
        """MULTIALLEN_BzRfrFRDkOAwTPSa multi-experiment creation."""
        model_folder = 'MULTIALLEN_BzRfrFRDkOAwTPSa'
        exp = {
            'experiment_name': ['MULTIALLEN_BzRfrFRDkOAwTPSa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BzRfrFRDkOAwTPSa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WVNoWbRjXMGidwVL(self):
        """MULTIALLEN_WVNoWbRjXMGidwVL multi-experiment creation."""
        model_folder = 'MULTIALLEN_WVNoWbRjXMGidwVL'
        exp = {
            'experiment_name': ['MULTIALLEN_WVNoWbRjXMGidwVL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WVNoWbRjXMGidwVL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aGdEnQqpuSWlIPRc(self):
        """MULTIALLEN_aGdEnQqpuSWlIPRc multi-experiment creation."""
        model_folder = 'MULTIALLEN_aGdEnQqpuSWlIPRc'
        exp = {
            'experiment_name': ['MULTIALLEN_aGdEnQqpuSWlIPRc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aGdEnQqpuSWlIPRc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eGbHCHLMlpdiJQOZ(self):
        """MULTIALLEN_eGbHCHLMlpdiJQOZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_eGbHCHLMlpdiJQOZ'
        exp = {
            'experiment_name': ['MULTIALLEN_eGbHCHLMlpdiJQOZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eGbHCHLMlpdiJQOZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JltyEdMoICxzxybn(self):
        """MULTIALLEN_JltyEdMoICxzxybn multi-experiment creation."""
        model_folder = 'MULTIALLEN_JltyEdMoICxzxybn'
        exp = {
            'experiment_name': ['MULTIALLEN_JltyEdMoICxzxybn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JltyEdMoICxzxybn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ikgYfkvBolQUlVTe(self):
        """MULTIALLEN_ikgYfkvBolQUlVTe multi-experiment creation."""
        model_folder = 'MULTIALLEN_ikgYfkvBolQUlVTe'
        exp = {
            'experiment_name': ['MULTIALLEN_ikgYfkvBolQUlVTe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ikgYfkvBolQUlVTe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hhsVKzjcOkIZPrHN(self):
        """MULTIALLEN_hhsVKzjcOkIZPrHN multi-experiment creation."""
        model_folder = 'MULTIALLEN_hhsVKzjcOkIZPrHN'
        exp = {
            'experiment_name': ['MULTIALLEN_hhsVKzjcOkIZPrHN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hhsVKzjcOkIZPrHN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SgpcFRoupxiUHnHF(self):
        """MULTIALLEN_SgpcFRoupxiUHnHF multi-experiment creation."""
        model_folder = 'MULTIALLEN_SgpcFRoupxiUHnHF'
        exp = {
            'experiment_name': ['MULTIALLEN_SgpcFRoupxiUHnHF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SgpcFRoupxiUHnHF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jcVvvVolYweWcKWq(self):
        """MULTIALLEN_jcVvvVolYweWcKWq multi-experiment creation."""
        model_folder = 'MULTIALLEN_jcVvvVolYweWcKWq'
        exp = {
            'experiment_name': ['MULTIALLEN_jcVvvVolYweWcKWq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jcVvvVolYweWcKWq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yZQlkKCrEVYMEJLY(self):
        """MULTIALLEN_yZQlkKCrEVYMEJLY multi-experiment creation."""
        model_folder = 'MULTIALLEN_yZQlkKCrEVYMEJLY'
        exp = {
            'experiment_name': ['MULTIALLEN_yZQlkKCrEVYMEJLY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yZQlkKCrEVYMEJLY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PoTHkAypgEfDRFQa(self):
        """MULTIALLEN_PoTHkAypgEfDRFQa multi-experiment creation."""
        model_folder = 'MULTIALLEN_PoTHkAypgEfDRFQa'
        exp = {
            'experiment_name': ['MULTIALLEN_PoTHkAypgEfDRFQa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PoTHkAypgEfDRFQa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uflBGfgPbEBiFdmB(self):
        """MULTIALLEN_uflBGfgPbEBiFdmB multi-experiment creation."""
        model_folder = 'MULTIALLEN_uflBGfgPbEBiFdmB'
        exp = {
            'experiment_name': ['MULTIALLEN_uflBGfgPbEBiFdmB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uflBGfgPbEBiFdmB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CoOvkZHQwcWrfIwy(self):
        """MULTIALLEN_CoOvkZHQwcWrfIwy multi-experiment creation."""
        model_folder = 'MULTIALLEN_CoOvkZHQwcWrfIwy'
        exp = {
            'experiment_name': ['MULTIALLEN_CoOvkZHQwcWrfIwy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CoOvkZHQwcWrfIwy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fELfNCFKiqhGBZio(self):
        """MULTIALLEN_fELfNCFKiqhGBZio multi-experiment creation."""
        model_folder = 'MULTIALLEN_fELfNCFKiqhGBZio'
        exp = {
            'experiment_name': ['MULTIALLEN_fELfNCFKiqhGBZio'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fELfNCFKiqhGBZio']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gbIIZbIAYKlJllAi(self):
        """MULTIALLEN_gbIIZbIAYKlJllAi multi-experiment creation."""
        model_folder = 'MULTIALLEN_gbIIZbIAYKlJllAi'
        exp = {
            'experiment_name': ['MULTIALLEN_gbIIZbIAYKlJllAi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gbIIZbIAYKlJllAi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TeVvBTKneBaOLvtQ(self):
        """MULTIALLEN_TeVvBTKneBaOLvtQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_TeVvBTKneBaOLvtQ'
        exp = {
            'experiment_name': ['MULTIALLEN_TeVvBTKneBaOLvtQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TeVvBTKneBaOLvtQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DKPbiLoMwCKLmpiQ(self):
        """MULTIALLEN_DKPbiLoMwCKLmpiQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_DKPbiLoMwCKLmpiQ'
        exp = {
            'experiment_name': ['MULTIALLEN_DKPbiLoMwCKLmpiQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DKPbiLoMwCKLmpiQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GMooNqciBbmoFOia(self):
        """MULTIALLEN_GMooNqciBbmoFOia multi-experiment creation."""
        model_folder = 'MULTIALLEN_GMooNqciBbmoFOia'
        exp = {
            'experiment_name': ['MULTIALLEN_GMooNqciBbmoFOia'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GMooNqciBbmoFOia']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DrlWLtiRToMTewoN(self):
        """MULTIALLEN_DrlWLtiRToMTewoN multi-experiment creation."""
        model_folder = 'MULTIALLEN_DrlWLtiRToMTewoN'
        exp = {
            'experiment_name': ['MULTIALLEN_DrlWLtiRToMTewoN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DrlWLtiRToMTewoN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cjCOPmXeLthLsgVY(self):
        """MULTIALLEN_cjCOPmXeLthLsgVY multi-experiment creation."""
        model_folder = 'MULTIALLEN_cjCOPmXeLthLsgVY'
        exp = {
            'experiment_name': ['MULTIALLEN_cjCOPmXeLthLsgVY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cjCOPmXeLthLsgVY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SklriYYxwZwSuDKc(self):
        """MULTIALLEN_SklriYYxwZwSuDKc multi-experiment creation."""
        model_folder = 'MULTIALLEN_SklriYYxwZwSuDKc'
        exp = {
            'experiment_name': ['MULTIALLEN_SklriYYxwZwSuDKc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SklriYYxwZwSuDKc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_olNrMUiwdoofujjr(self):
        """MULTIALLEN_olNrMUiwdoofujjr multi-experiment creation."""
        model_folder = 'MULTIALLEN_olNrMUiwdoofujjr'
        exp = {
            'experiment_name': ['MULTIALLEN_olNrMUiwdoofujjr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_olNrMUiwdoofujjr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ufDIxbxUHGFpElcl(self):
        """MULTIALLEN_ufDIxbxUHGFpElcl multi-experiment creation."""
        model_folder = 'MULTIALLEN_ufDIxbxUHGFpElcl'
        exp = {
            'experiment_name': ['MULTIALLEN_ufDIxbxUHGFpElcl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ufDIxbxUHGFpElcl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jpiAUkxUykoAwMaS(self):
        """MULTIALLEN_jpiAUkxUykoAwMaS multi-experiment creation."""
        model_folder = 'MULTIALLEN_jpiAUkxUykoAwMaS'
        exp = {
            'experiment_name': ['MULTIALLEN_jpiAUkxUykoAwMaS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jpiAUkxUykoAwMaS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LEabpXmBeiPkMRaY(self):
        """MULTIALLEN_LEabpXmBeiPkMRaY multi-experiment creation."""
        model_folder = 'MULTIALLEN_LEabpXmBeiPkMRaY'
        exp = {
            'experiment_name': ['MULTIALLEN_LEabpXmBeiPkMRaY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LEabpXmBeiPkMRaY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_drvqtXCuyLgodkBh(self):
        """MULTIALLEN_drvqtXCuyLgodkBh multi-experiment creation."""
        model_folder = 'MULTIALLEN_drvqtXCuyLgodkBh'
        exp = {
            'experiment_name': ['MULTIALLEN_drvqtXCuyLgodkBh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_drvqtXCuyLgodkBh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ycSoQDHtxgwOvbmD(self):
        """MULTIALLEN_ycSoQDHtxgwOvbmD multi-experiment creation."""
        model_folder = 'MULTIALLEN_ycSoQDHtxgwOvbmD'
        exp = {
            'experiment_name': ['MULTIALLEN_ycSoQDHtxgwOvbmD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ycSoQDHtxgwOvbmD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AjSWIaGllTykrwln(self):
        """MULTIALLEN_AjSWIaGllTykrwln multi-experiment creation."""
        model_folder = 'MULTIALLEN_AjSWIaGllTykrwln'
        exp = {
            'experiment_name': ['MULTIALLEN_AjSWIaGllTykrwln'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AjSWIaGllTykrwln']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SGlpLxvXvrBFuOVh(self):
        """MULTIALLEN_SGlpLxvXvrBFuOVh multi-experiment creation."""
        model_folder = 'MULTIALLEN_SGlpLxvXvrBFuOVh'
        exp = {
            'experiment_name': ['MULTIALLEN_SGlpLxvXvrBFuOVh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SGlpLxvXvrBFuOVh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AuvCyezsodYWYWKD(self):
        """MULTIALLEN_AuvCyezsodYWYWKD multi-experiment creation."""
        model_folder = 'MULTIALLEN_AuvCyezsodYWYWKD'
        exp = {
            'experiment_name': ['MULTIALLEN_AuvCyezsodYWYWKD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AuvCyezsodYWYWKD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JdxWDJKquXDaeNbr(self):
        """MULTIALLEN_JdxWDJKquXDaeNbr multi-experiment creation."""
        model_folder = 'MULTIALLEN_JdxWDJKquXDaeNbr'
        exp = {
            'experiment_name': ['MULTIALLEN_JdxWDJKquXDaeNbr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JdxWDJKquXDaeNbr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iYllnTtsUjtjxjwy(self):
        """MULTIALLEN_iYllnTtsUjtjxjwy multi-experiment creation."""
        model_folder = 'MULTIALLEN_iYllnTtsUjtjxjwy'
        exp = {
            'experiment_name': ['MULTIALLEN_iYllnTtsUjtjxjwy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iYllnTtsUjtjxjwy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hgSRNtiNEzQTsnOF(self):
        """MULTIALLEN_hgSRNtiNEzQTsnOF multi-experiment creation."""
        model_folder = 'MULTIALLEN_hgSRNtiNEzQTsnOF'
        exp = {
            'experiment_name': ['MULTIALLEN_hgSRNtiNEzQTsnOF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hgSRNtiNEzQTsnOF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qfIHZOPFNilEUHdy(self):
        """MULTIALLEN_qfIHZOPFNilEUHdy multi-experiment creation."""
        model_folder = 'MULTIALLEN_qfIHZOPFNilEUHdy'
        exp = {
            'experiment_name': ['MULTIALLEN_qfIHZOPFNilEUHdy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qfIHZOPFNilEUHdy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LLrIMcUGRtKscDDm(self):
        """MULTIALLEN_LLrIMcUGRtKscDDm multi-experiment creation."""
        model_folder = 'MULTIALLEN_LLrIMcUGRtKscDDm'
        exp = {
            'experiment_name': ['MULTIALLEN_LLrIMcUGRtKscDDm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LLrIMcUGRtKscDDm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sOimSbBTARrSaLnI(self):
        """MULTIALLEN_sOimSbBTARrSaLnI multi-experiment creation."""
        model_folder = 'MULTIALLEN_sOimSbBTARrSaLnI'
        exp = {
            'experiment_name': ['MULTIALLEN_sOimSbBTARrSaLnI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sOimSbBTARrSaLnI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PDtpSqGywxlJDRtY(self):
        """MULTIALLEN_PDtpSqGywxlJDRtY multi-experiment creation."""
        model_folder = 'MULTIALLEN_PDtpSqGywxlJDRtY'
        exp = {
            'experiment_name': ['MULTIALLEN_PDtpSqGywxlJDRtY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PDtpSqGywxlJDRtY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KKIcVdjFDwZUYYDq(self):
        """MULTIALLEN_KKIcVdjFDwZUYYDq multi-experiment creation."""
        model_folder = 'MULTIALLEN_KKIcVdjFDwZUYYDq'
        exp = {
            'experiment_name': ['MULTIALLEN_KKIcVdjFDwZUYYDq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KKIcVdjFDwZUYYDq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lBjXjqofJjHwsjTN(self):
        """MULTIALLEN_lBjXjqofJjHwsjTN multi-experiment creation."""
        model_folder = 'MULTIALLEN_lBjXjqofJjHwsjTN'
        exp = {
            'experiment_name': ['MULTIALLEN_lBjXjqofJjHwsjTN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lBjXjqofJjHwsjTN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_onecQxuhAumPnoZM(self):
        """MULTIALLEN_onecQxuhAumPnoZM multi-experiment creation."""
        model_folder = 'MULTIALLEN_onecQxuhAumPnoZM'
        exp = {
            'experiment_name': ['MULTIALLEN_onecQxuhAumPnoZM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_onecQxuhAumPnoZM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bfVjNWpsPEAnPnSI(self):
        """MULTIALLEN_bfVjNWpsPEAnPnSI multi-experiment creation."""
        model_folder = 'MULTIALLEN_bfVjNWpsPEAnPnSI'
        exp = {
            'experiment_name': ['MULTIALLEN_bfVjNWpsPEAnPnSI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bfVjNWpsPEAnPnSI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hRnxKQeXnzrSbfwq(self):
        """MULTIALLEN_hRnxKQeXnzrSbfwq multi-experiment creation."""
        model_folder = 'MULTIALLEN_hRnxKQeXnzrSbfwq'
        exp = {
            'experiment_name': ['MULTIALLEN_hRnxKQeXnzrSbfwq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hRnxKQeXnzrSbfwq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IbloSXSJpttjIHbc(self):
        """MULTIALLEN_IbloSXSJpttjIHbc multi-experiment creation."""
        model_folder = 'MULTIALLEN_IbloSXSJpttjIHbc'
        exp = {
            'experiment_name': ['MULTIALLEN_IbloSXSJpttjIHbc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IbloSXSJpttjIHbc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PmcrkyeDdbhFvgxM(self):
        """MULTIALLEN_PmcrkyeDdbhFvgxM multi-experiment creation."""
        model_folder = 'MULTIALLEN_PmcrkyeDdbhFvgxM'
        exp = {
            'experiment_name': ['MULTIALLEN_PmcrkyeDdbhFvgxM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PmcrkyeDdbhFvgxM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SZVqxIiiuWExRgQI(self):
        """MULTIALLEN_SZVqxIiiuWExRgQI multi-experiment creation."""
        model_folder = 'MULTIALLEN_SZVqxIiiuWExRgQI'
        exp = {
            'experiment_name': ['MULTIALLEN_SZVqxIiiuWExRgQI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SZVqxIiiuWExRgQI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kRnkBZBfZFskIZSj(self):
        """MULTIALLEN_kRnkBZBfZFskIZSj multi-experiment creation."""
        model_folder = 'MULTIALLEN_kRnkBZBfZFskIZSj'
        exp = {
            'experiment_name': ['MULTIALLEN_kRnkBZBfZFskIZSj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kRnkBZBfZFskIZSj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ebDNgHuydBWChRjW(self):
        """MULTIALLEN_ebDNgHuydBWChRjW multi-experiment creation."""
        model_folder = 'MULTIALLEN_ebDNgHuydBWChRjW'
        exp = {
            'experiment_name': ['MULTIALLEN_ebDNgHuydBWChRjW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ebDNgHuydBWChRjW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vyHpxZdcWwPSzMXq(self):
        """MULTIALLEN_vyHpxZdcWwPSzMXq multi-experiment creation."""
        model_folder = 'MULTIALLEN_vyHpxZdcWwPSzMXq'
        exp = {
            'experiment_name': ['MULTIALLEN_vyHpxZdcWwPSzMXq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vyHpxZdcWwPSzMXq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fPTifSrquFYdjBJg(self):
        """MULTIALLEN_fPTifSrquFYdjBJg multi-experiment creation."""
        model_folder = 'MULTIALLEN_fPTifSrquFYdjBJg'
        exp = {
            'experiment_name': ['MULTIALLEN_fPTifSrquFYdjBJg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fPTifSrquFYdjBJg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bwwmKNhGIkSnHaNG(self):
        """MULTIALLEN_bwwmKNhGIkSnHaNG multi-experiment creation."""
        model_folder = 'MULTIALLEN_bwwmKNhGIkSnHaNG'
        exp = {
            'experiment_name': ['MULTIALLEN_bwwmKNhGIkSnHaNG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bwwmKNhGIkSnHaNG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YxwxVbShITpakdnG(self):
        """MULTIALLEN_YxwxVbShITpakdnG multi-experiment creation."""
        model_folder = 'MULTIALLEN_YxwxVbShITpakdnG'
        exp = {
            'experiment_name': ['MULTIALLEN_YxwxVbShITpakdnG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YxwxVbShITpakdnG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_obdxTDEnLLISDAFK(self):
        """MULTIALLEN_obdxTDEnLLISDAFK multi-experiment creation."""
        model_folder = 'MULTIALLEN_obdxTDEnLLISDAFK'
        exp = {
            'experiment_name': ['MULTIALLEN_obdxTDEnLLISDAFK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_obdxTDEnLLISDAFK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UzTfdtOGNkdIyEjZ(self):
        """MULTIALLEN_UzTfdtOGNkdIyEjZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_UzTfdtOGNkdIyEjZ'
        exp = {
            'experiment_name': ['MULTIALLEN_UzTfdtOGNkdIyEjZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UzTfdtOGNkdIyEjZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YESereSUfoJEaQKE(self):
        """MULTIALLEN_YESereSUfoJEaQKE multi-experiment creation."""
        model_folder = 'MULTIALLEN_YESereSUfoJEaQKE'
        exp = {
            'experiment_name': ['MULTIALLEN_YESereSUfoJEaQKE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YESereSUfoJEaQKE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hPUqOdKYbzaCkihE(self):
        """MULTIALLEN_hPUqOdKYbzaCkihE multi-experiment creation."""
        model_folder = 'MULTIALLEN_hPUqOdKYbzaCkihE'
        exp = {
            'experiment_name': ['MULTIALLEN_hPUqOdKYbzaCkihE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hPUqOdKYbzaCkihE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DFjzrDgxCJYMpHgq(self):
        """MULTIALLEN_DFjzrDgxCJYMpHgq multi-experiment creation."""
        model_folder = 'MULTIALLEN_DFjzrDgxCJYMpHgq'
        exp = {
            'experiment_name': ['MULTIALLEN_DFjzrDgxCJYMpHgq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DFjzrDgxCJYMpHgq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qleShurOnlbsuiqh(self):
        """MULTIALLEN_qleShurOnlbsuiqh multi-experiment creation."""
        model_folder = 'MULTIALLEN_qleShurOnlbsuiqh'
        exp = {
            'experiment_name': ['MULTIALLEN_qleShurOnlbsuiqh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qleShurOnlbsuiqh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_keVZWRsJWFBBtpEn(self):
        """MULTIALLEN_keVZWRsJWFBBtpEn multi-experiment creation."""
        model_folder = 'MULTIALLEN_keVZWRsJWFBBtpEn'
        exp = {
            'experiment_name': ['MULTIALLEN_keVZWRsJWFBBtpEn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_keVZWRsJWFBBtpEn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XgnIeMwEydDQhpab(self):
        """MULTIALLEN_XgnIeMwEydDQhpab multi-experiment creation."""
        model_folder = 'MULTIALLEN_XgnIeMwEydDQhpab'
        exp = {
            'experiment_name': ['MULTIALLEN_XgnIeMwEydDQhpab'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XgnIeMwEydDQhpab']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OIiWLqNCfhxKRaut(self):
        """MULTIALLEN_OIiWLqNCfhxKRaut multi-experiment creation."""
        model_folder = 'MULTIALLEN_OIiWLqNCfhxKRaut'
        exp = {
            'experiment_name': ['MULTIALLEN_OIiWLqNCfhxKRaut'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OIiWLqNCfhxKRaut']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PDwedlmQqUoRXdjS(self):
        """MULTIALLEN_PDwedlmQqUoRXdjS multi-experiment creation."""
        model_folder = 'MULTIALLEN_PDwedlmQqUoRXdjS'
        exp = {
            'experiment_name': ['MULTIALLEN_PDwedlmQqUoRXdjS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PDwedlmQqUoRXdjS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vmaFBJvuDzpxoeHA(self):
        """MULTIALLEN_vmaFBJvuDzpxoeHA multi-experiment creation."""
        model_folder = 'MULTIALLEN_vmaFBJvuDzpxoeHA'
        exp = {
            'experiment_name': ['MULTIALLEN_vmaFBJvuDzpxoeHA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vmaFBJvuDzpxoeHA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ReIwHWOReOQJgFxz(self):
        """MULTIALLEN_ReIwHWOReOQJgFxz multi-experiment creation."""
        model_folder = 'MULTIALLEN_ReIwHWOReOQJgFxz'
        exp = {
            'experiment_name': ['MULTIALLEN_ReIwHWOReOQJgFxz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ReIwHWOReOQJgFxz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tJlPgcNHKesQQcwc(self):
        """MULTIALLEN_tJlPgcNHKesQQcwc multi-experiment creation."""
        model_folder = 'MULTIALLEN_tJlPgcNHKesQQcwc'
        exp = {
            'experiment_name': ['MULTIALLEN_tJlPgcNHKesQQcwc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tJlPgcNHKesQQcwc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VKOyInTuYQLQEmoW(self):
        """MULTIALLEN_VKOyInTuYQLQEmoW multi-experiment creation."""
        model_folder = 'MULTIALLEN_VKOyInTuYQLQEmoW'
        exp = {
            'experiment_name': ['MULTIALLEN_VKOyInTuYQLQEmoW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VKOyInTuYQLQEmoW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NynZbXCKZaZGoJoa(self):
        """MULTIALLEN_NynZbXCKZaZGoJoa multi-experiment creation."""
        model_folder = 'MULTIALLEN_NynZbXCKZaZGoJoa'
        exp = {
            'experiment_name': ['MULTIALLEN_NynZbXCKZaZGoJoa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NynZbXCKZaZGoJoa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oQYQlkyAzkDLUjaX(self):
        """MULTIALLEN_oQYQlkyAzkDLUjaX multi-experiment creation."""
        model_folder = 'MULTIALLEN_oQYQlkyAzkDLUjaX'
        exp = {
            'experiment_name': ['MULTIALLEN_oQYQlkyAzkDLUjaX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oQYQlkyAzkDLUjaX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xVihcYuErAKotEkc(self):
        """MULTIALLEN_xVihcYuErAKotEkc multi-experiment creation."""
        model_folder = 'MULTIALLEN_xVihcYuErAKotEkc'
        exp = {
            'experiment_name': ['MULTIALLEN_xVihcYuErAKotEkc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xVihcYuErAKotEkc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PNZGembArINScDXa(self):
        """MULTIALLEN_PNZGembArINScDXa multi-experiment creation."""
        model_folder = 'MULTIALLEN_PNZGembArINScDXa'
        exp = {
            'experiment_name': ['MULTIALLEN_PNZGembArINScDXa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PNZGembArINScDXa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MoPuLjTxFwHSUYTw(self):
        """MULTIALLEN_MoPuLjTxFwHSUYTw multi-experiment creation."""
        model_folder = 'MULTIALLEN_MoPuLjTxFwHSUYTw'
        exp = {
            'experiment_name': ['MULTIALLEN_MoPuLjTxFwHSUYTw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MoPuLjTxFwHSUYTw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp

