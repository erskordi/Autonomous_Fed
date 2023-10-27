class Config(object):
    def __init__(self) -> None:
        # Parameters
        self.input_dim = 3  # Dimension of input data (GDP gap, inflation rates, interest rates)
        self.output_dim = 3  # Dimension of output data (GDP gap, inflation rates)
        self.input_shape = (self.input_dim,)
        self.intermediate_dim = 16
        self.latent_dim = 2
        self.batch_size = 16
        self.epochs = 50

        # If using GPUs, set this to True.
        self.use_gpu = False

        self.a = 5
        self.b = 10
        self.size = 100

        self.path_to_data = "/Users/erotokritosskordilis/Dropbox/ProvostAward2024/MonetaryPolicy_AI/train_data.csv"

        # suffixes
        self.suffix_prior = "_prior"
        self.suffix_posterior = "_posterior"