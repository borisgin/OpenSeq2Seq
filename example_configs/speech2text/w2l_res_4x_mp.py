# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import Wave2LetterEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay


base_model = Speech2Text
data_prefix =  "data/" # "/raid/speech/" #

base_params = {
    "random_seed": 0,
    "use_horovod":  False, #
    #"max_steps" : 50,
    "num_epochs": 100, #50,

    "num_gpus":2, #8, #
    "batch_size_per_gpu": 32,
    "iter_size": 1,

    "save_summaries_steps": 1100,
    "print_loss_steps": 50,
    "print_samples_steps": 1100,
    "eval_steps": 2200,
    "save_checkpoint_steps": 44000,
    "logdir": "tmp_log/w2l",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.1, #0.01,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.002,
    },
    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": Wave2LetterEncoder,
    "encoder_params": {
        "convnet_layers": [
            # {
            #     "type": "conv1d", "repeat": 1,
            #     "kernel_size": [11], "stride": [2],
            #     "num_channels": 128, "padding": "SAME",
            # },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                # "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                # "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                # "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                # "dropout_keep_prob": 0.7,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                # "dropout_keep_prob": 0.7,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                # "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 2048, "padding": "SAME",
                # "dropout_keep_prob": 0.6,
            },
        ],
        "dropout_keep_prob": 0.6,
        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
#        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "activation_fn": tf.nn.relu,
        "data_format": "channels_last",
        "residual": True,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,
        # params for decoding the sequence with language model
        "beam_width": 512,
        "lm_weight": 2.0,
        "word_count_weight": 1.5,
        "valid_word_count_weight": 2.5,

        "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        "lm_binary_path": "language_model/lm.binary",
        "lm_trie_path": "language_model/trie",
        "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    },
    "loss": CTCLoss,
    "loss_params": {},
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        # "augmentation": {'time_stretch_ratio': 0.03,
        #              'noise_level_min': -90,
        #              'noise_level_max': -70},
        "dataset_files": [
            data_prefix+"librispeech/librivox-train-clean-100.csv",
            data_prefix+"librispeech/librivox-train-clean-360.csv",
            data_prefix+"librispeech/librivox-train-other-500.csv",
        ],
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            data_prefix+"librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            data_prefix+"librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}
