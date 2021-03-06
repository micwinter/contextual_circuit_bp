"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['_pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 5,
            'CRF_excitation': 5,
            'CRF_inhibition': 5,
            'SSN': 10,
            'SSF': 27.5
        },
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_vector_modulation'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 10,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',
                   'regularization_strength': 0.01
                },
                't_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                },
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                },
            }
        },
    }
]

output_structure = [
    {
        'layers': ['gather'],
        'aux': {
            'h': 25,
            'w': 25
        },  # Output size
        'names': ['gather'],
    },
    {
        'layers': ['fc'],
        'weights': [1],
        'names': ['fc1'],
    }
]
