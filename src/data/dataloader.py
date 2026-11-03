import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __init__(self, *args, **kwargs):
        """
        Thin wrapper around torch.utils.data.DataLoader with config-friendly safety:
        - If num_workers == 0, force persistent_workers=False and prefetch_factor=None
          (PyTorch raises if these are set in single-process mode).
        """
        num_workers = kwargs.get('num_workers', None)
        try:
            num_workers_i = int(num_workers) if num_workers is not None else 0
        except Exception:
            num_workers_i = 0

        if num_workers_i <= 0:
            # safe defaults for single-process loading
            if 'persistent_workers' in kwargs:
                kwargs['persistent_workers'] = False
            if 'prefetch_factor' in kwargs:
                kwargs['prefetch_factor'] = None
        else:
            # if user explicitly passes null/None, fall back to PyTorch default (=2)
            if 'prefetch_factor' in kwargs and kwargs.get('prefetch_factor', None) is None:
                kwargs['prefetch_factor'] = 2

        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string
