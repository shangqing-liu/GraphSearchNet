"""
Module to handle universal/general constants used across files.
"""
# GENERAL CONSTANTS:
VERY_SMALL_NUMBER = 1e-31
INF = 1e20
MINUS_INF = -1e20


_PAD_TOKEN = '#pad#'       # 0
_UNK_TOKEN = '<unk>'       # 3
_SOS_TOKEN = '<s>'         # 1
_EOS_TOKEN = '</s>'        # 2


# LOG FILES ##
_CONFIG_FILE = "config.json"
_SAVED_WEIGHTS_FILE = "params.saved"
_PREDICTION_FILE = "test_pred.txt"
_REFERENCE_FILE = "test_ref.txt"
