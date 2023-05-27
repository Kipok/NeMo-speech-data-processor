from typing import Dict

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

NA_INDICATOR="n/a"

class ChangePCFields(BaseParallelProcessor):

    def __init__(
        self,  **kwargs,
    ):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry: Dict):
        if data_entry["text_pc"] != NA_INDICATOR:
            data_entry["text"] = data_entry["text_pc"]
            data_entry["text_pc_origin"] = "original"

        else:
            data_entry["text"] = data_entry["text_pc_pred"]
            data_entry["text_pc_origin"] = "generated"

        # remove old fields
        del data_entry["text_pc"]
        del data_entry["text_pc_pred"]

        return [DataEntry(data=data_entry)]