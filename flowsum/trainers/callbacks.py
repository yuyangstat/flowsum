from typing import List

from transformers.integrations import TensorBoardCallback
from matplotlib.figure import Figure


class VisualTensorBoardCallback(TensorBoardCallback):
    """
    Add visualization to TensorBoardCallback.

    Notes:
        (1) For now, in order to keep track of all images generated along the training process,
            I dynamically update the tag. If there are better ways, may modify it.
    """

    def on_visualize(self, args, state, control, figures: List[Figure], **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            self.tb_writer.add_figure(
                tag=f"latent_dist_step{state.global_step}",
                figure=figures,
                global_step=state.global_step,
            )
            self.tb_writer.flush()
