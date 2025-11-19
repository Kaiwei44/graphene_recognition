from typing import Dict, Union

from detectron2.config import CfgNode
from detectron2.utils.events import EventStorage, EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(
        self,
        config: Union[Dict, CfgNode] = {},  # noqa: B006
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        import wandb

        self._run = wandb.init(config=config, **kwargs) if not wandb.run else wandb.run
        self._run._label(repo="detectron2")

    def write(self):
        storage: EventStorage = get_event_storage()

        storage_dict = storage.latest()

        for i, (key, (value, logged_iteration)) in enumerate(storage_dict.items()):
            self._run.log(
                {key: value},
                step=logged_iteration,
                commit=True if i == len(storage_dict) - 1 else False,
            )

    def close(self):
        self._run.finish()
