class GPTConfig:
    def __init__(
        self,
        vocab_size,
        max_seq_len=128,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        batch_size=32,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=500,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.eval_interval = eval_interval