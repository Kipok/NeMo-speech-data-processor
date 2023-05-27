from typing import Dict

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class RenameFields(BaseParallelProcessor):

    def __init__(
        self, rename_fields: Dict, **kwargs,
    ):
        super().__init__(**kwargs)
        self.rename_fields = rename_fields

    def process_dataset_entry(self, data_entry: Dict):
        for field_in, field_out in self.rename_fields.items():
            data_entry[field_out] = data_entry[field_in]
            del data_entry[field_in]

        return [DataEntry(data=data_entry)]