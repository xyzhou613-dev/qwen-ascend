"""Export vLLM TP state before Ascend converts it to kernel-specific layouts."""

import os
from unittest.mock import patch

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm_ascend.worker.worker import NPUWorker


class ExportWorker(NPUWorker):
    def load_model(self):
        load_weights = DefaultModelLoader.load_weights

        def load_and_export(loader, model, model_config):
            load_weights(loader, model, model_config)
            ShardedStateLoader.save_model(
                model, os.environ["QWEN27_LQP_EXPORT_DIR"], max_size=2 * 1024**3
            )

        with patch.object(DefaultModelLoader, "load_weights", load_and_export):
            super().load_model()
