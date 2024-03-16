class Config(object):
    def __init__(self) -> None:
        # Parameters
        self.input_dim = 4  # Dimension of input data (GDP gap, inflation rates(x2), interest rates)
        self.output_dim = 3  # Dimension of output data (GDP gap, inflation rates(x2))
        self.input_shape = (self.input_dim,)
        self.intermediate_dim = 16
        self.latent_dim = 2
        self.batch_size = 32
        self.epochs = 1000
        self.dense_neurons = [256, 256, 128, 64, 32]

        # Split data
        self.train_split = 1.0
        self.val_split = self.train_split + .1 if self.train_split <= 0.9 else 0.0 # only for NNs
        self.test_split = 1 - self.train_split

        # If using GPUs, set this to True.
        self.use_gpu = False

        self.a = 5
        self.b = 10
        self.size = 100

        self.path_to_data = "/home/erskordi/projects/Autonomous_Fed/data/Data_in_Levels.xlsx"

        # suffixes
        self.suffix_prior = "_prior"
        self.suffix_posterior = "_posterior"

        # Sequence length (for LSTM)
        self.sequence_length = 1

        # filters for LSTM layer
        self.filters = [32, 64, 16]
        self.lstm_mlp_units = [16, 8]

        # Number of transformer blocks (for Transformer)
        self.num_transformer_blocks = 8
        # Number of heads (for Transformer)
        self.num_heads = 2
        # Head size (for Transformer)
        self.head_size = 128
        # Feed forward dimension (for Transformer)
        self.ff_dim = 2
        # Dropout rate (for Transformer)
        self.dropout = 0.
        # MLP units (for Transformer)
        self.mlp_units = [8, 4]
        # MLP dropout (for Transformer)
        self.mlp_dropout = 0.