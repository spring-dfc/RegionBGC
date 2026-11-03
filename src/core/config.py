"""Runtime configuration state shared by YAML-backed configurations."""

import torch


class BaseConfig:
    """Store lazily constructed training components and runtime options."""

    def __init__(self) -> None:
        self._model = None
        self._postprocessor = None
        self._criterion = None
        self._optimizer = None
        self._lr_scheduler = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._ema = None
        self._scaler = None

        self.resume = ""
        self.tuning = ""
        self.epoches = -1
        self.last_epoch = -1
        self.use_amp = False
        self.use_ema = False
        self.clip_max_norm = 0.0
        self.log_step = 10
        self._output_dir = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, root):
        self._output_dir = root
