# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class NormalizeWithAudio(BaseProcessor):
    def __init__(
        self, tn_script_dir: str, language: str, cache_dir: Optional[str] = None, **kwargs,
    ):
        super().__init__(**kwargs)
        self.tn_script_dir = tn_script_dir
        self.language = language
        self.cache_dir = cache_dir

    def process(self):
        subprocess.run(
            f"python {os.path.join(self.tn_script_dir, 'normalize_with_audio.py')} "
            f" --audio_data={self.input_manifest_file}"
            f" --output_filename={self.output_manifest_file}"
            f" --cache_dir={self.cache_dir}"
            f" --language={self.language}"
            f" --input_case=cased",
            shell=True,
            check=True,
        )
