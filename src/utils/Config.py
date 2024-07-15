import ast
import json
import os

import git


class Config(object):
    def __init__(self, filenames=None, kwargs=None):
        # Git commit version
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        self.git_commit = sha

        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.seed = 42
        self.device = "cuda"
        self.is_debug = False

        # Weak supervision for attribution training parameters
        self.ws_iteration = -1
        self.ws_attribution_training = False
        self.ws_filtering_threshold = 0.1
        self.ws_filtering_threshold_correctness = 0.1
        self.consistency_model = "google/t5_xxl_true_nli_mixture"
        self.consistency_threshold = 0.7
        self.dynamic_steps = False
        self.use_chat_template = False
        self.bootstrapping = False
        self.dynamic_stopping_tolerance = 0.001
        self.unlabeled_samples_cutoff = (
            1000  # How many unlabeled examples to be considered for weak supervision. Arbitarily set to 1000
        )
        self.sample_citation_replacements = False  # Whether to create additional candidates by replacing citations in generated answers. Default False since computationally expensive.

        # Focused learning parameters
        self.focused_learning = False
        self.importance_threshold = (
            0.5  # Set importance to all tokens to zero that have an importance below that threshold
        )
        self.importance_attribution_weights = 2  # How much citations tokens weights should be increased by
        self.focused_learning_mode = "full_masking"  # full_masking versus soft masking using scores themselves
        self.mask_all_but_citations = False
        self.length_bias_term_factor = 0.25
        self.shap_normalization = "min_max"

        # General Model Settings
        self.model_type = "decoder_only"
        self.load_weight = ""
        self.dropout = 0.1
        self.compute_strategy = "auto"
        self.normalized_scores = True
        self.answer_truncation = (
            "none"  # Whether to truncate input answer, generated, answer or both (train, inference, all)
        )

        # Reader Configs
        self.encoder_format = (
            "{query} title: {title} context: {text}"  # format string for reader's encoder preprocessing
        )
        self.text_maxlength = 3500  # maximum number of tokens in input text (question + passages).",
        self.use_gradient_checkpoint_reader = True  # use gradient checkpointing in the reader
        self.n_context = 3  # number of top k passages to pass to reader
        self.reader_model_origin = "Open-Orca/Mistral-7B-OpenOrca"
        self.in_context_learning = False  # Whether to use in_context learning in reader
        self.in_context_learning_samples = 1
        self.train_max_length = 512  # Max length of training answer

        # Generation setting
        self.decoder_prompt_format = (
            None  # format for decoder prompting, for instance "what is the answer to {query}:"
        )
        self.decoder_format = None  # format for decoder, model will be train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option. Runs on target sequence, instead on query.
        self.generation_max_length = 256
        self.generation_min_length = None
        self.generation_do_sample = (
            False  # Whehter to not use sampling when doing final generation (for weakly supervised still uses)
        )
        self.generation_num_beams = 1
        self.generation_length_penalty = 1.0
        self.target_maxlength = None  # Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1"
        self.temperature = 0.5  # Temperature for decoding
        self.top_p = 1.0  # Nucleus sampling top-p

        # Dataset Configs
        self.dataset = "asqa"
        self.stratified_sampling = False
        self.num_samples = 2
        self.batch_size = 1
        self.eval_batch_size = 1
        self.per_gpu_embedder_batch_size = 512  # Embedder's batch size per GPU.
        self.num_workers = 8

        # Template Settings
        self.template_setting_name = "default"
        self.attribution_representation = "num"

        # Trainer configs
        self.num_steps = 15_000  # 100_000
        self.grad_accum_factor = 4
        self.val_check_interval = 3000
        self.eval_before_training = False  # True
        self.save_model = True
        self.save_step_interval = 20_000

        # Optimization configs
        self.optimizer = "adamw"
        self.lr = 5e-5
        self.trainable_param_names = ".*"
        self.scheduler = "linear_decay_with_warmup"
        self.warmup_ratio = 0.06
        self.weight_decay = 0.3
        self.scale_parameter = True
        self.grad_clip_norm = 1

        # PEFT configs
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05

        # Retriever Setting
        self.retriever = "contriever"  # Specifies which retriever to use. Select from TODO
        self.train_retriever = False  # Whether to train the retriever
        self.use_file_passages = (
            True  # uses passages in "passages" field in train or eval jsonl files rather than retrieving passages'
        )
        self.use_gold_passages = (
            False  # Select from True False Only (True is first mode, where we use gold and fill up with retrieved)
        )
        self.use_gradient_checkpoint_retriever = True  # use gradient checkpointing in the retriever
        self.load_index_path = None  # path for loading the index, passage embeddings and passages
        self.query_side_retriever_training = True  # pass to enable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes.
        self.gold_score_mode = "ppmean"  # "retriever training method. `pdist` is the name used in the Atlas paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum`"
        self.retriever_format = "{title} {text}"  # format string for retriever's encoder preprocessing
        self.retriever_n_context = 1  # number of top k passages to use to train the retriever with
        self.temperature_gold = 0.01  # softmax temperature for target distribution for retriever distillation"
        self.temperature_score = 0.01  # softmax temperature for retriever"
        self.overwrite_data = False  # Whether to overwtite existing stored retrieval

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

        assert (
            not self.focused_learning or self.batch_size == 1
        ), "Found batch size: {}. Currently focused learning does not support batching, please set `batch_size=1`.".format(
            self.batch_size
        )

        assert not self.sample_citation_replacements or (self.batch_size == 1 and self.eval_batch_size == 1), "Currently citation replacements do not support batching, so please set `batch_size` and `eval_batch_size` to `1`."

        self.set_exp_dir()

    def update_kwargs(self, kwargs, eval=True):
        for k, v in kwargs.items():
            print(k, v)
            if eval:
                try:
                    if "+" in v:  # Spaces are replaced via symbol
                        v = v.replace("+", " ")
                    else:
                        v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def set_exp_dir(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        if self.exp_name is not None:
            self.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), self.exp_name)
        else:
            self.exp_dir = os.getenv("OUTPUT_PATH", default="exp_out")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if self.exp_dir is not None:
            self.dev_pred_file = os.path.join(self.exp_dir, "dev_pred.json")
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_pred_file = os.path.join(self.exp_dir, "test_pred.json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")
