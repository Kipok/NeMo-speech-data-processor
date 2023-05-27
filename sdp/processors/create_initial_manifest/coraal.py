# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import sox
from sox import Transformer
from tqdm.contrib.concurrent import process_map

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive


# Full cleaning functions
def remove_markers(line, markers):
    # Remove any text within markers, e.g. 'We(BR) went' -> 'We went'
    # markers = list of pairs, e.g. ['()', '[]'] denoting breath or noise in transcripts
    for s, e in markers:
        line = re.sub(" ?\\" + s + "[^" + e + "]+\\" + e, "", line)
    return line


def clean_coraal(baseline_snippets):
    # Clean CORAAL human transcript
    # Restrict to CORAAL rows
    baseline_coraal = baseline_snippets

    # Replace original unmatched CORAAL transcript square brackets with squiggly bracket
    baseline_coraal.loc[:, 'clean_content'] = baseline_coraal.loc[:, 'content'].copy()
    baseline_coraal.loc[:, 'clean_content'] = baseline_coraal['clean_content'].str.replace('\[', '\{')
    baseline_coraal.loc[:, 'clean_content'] = baseline_coraal['clean_content'].str.replace('\]', '\}')

    def clean_within_coraal(text):

        # Relabel CORAAL words. For consideration: aks -> ask?
        split_words = text.split()
        split_words = [x if x != 'busses' else 'buses' for x in split_words]
        split_words = [x if x != 'aks' else 'ask' for x in split_words]
        split_words = [x if x != 'aksing' else 'asking' for x in split_words]
        split_words = [x if x != 'aksed' else 'asked' for x in split_words]
        text = ' '.join(split_words)

        # remove CORAAL unintelligible flags
        text = re.sub("\/(?i)unintelligible\/", '', ''.join(text))
        text = re.sub("\/(?i)inaudible\/", '', ''.join(text))
        text = re.sub('\/RD(.*?)\/', '', ''.join(text))
        text = re.sub('\/(\?)\1*\/', '', ''.join(text))

        # remove nonlinguistic markers
        text = remove_markers(text, ['<>', '()', '{}'])

        return text

    baseline_coraal['clean_content'] = baseline_coraal.apply(lambda x: clean_within_coraal(x['clean_content']), axis=1)

    return baseline_coraal


class CreateInitialManifestCORAAL(BaseParallelProcessor):
    def __init__(
        self,
        data_path: str,
        resampled_audio_dir: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = Path(data_path)
        self.resampled_audio_dir = resampled_audio_dir
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        # TODO: include prep step by using asr-disparities repo or copying processing over
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        df = pd.read_csv(self.data_path / "output.tsv", delimiter="\t")
        df = clean_coraal(df)
        return df.values

    def process_dataset_entry(self, data_entry):
        file_name, text = data_entry[-2], data_entry[-1]
        file_path = str(self.data_path / file_name[:3].lower() / "audio_segments" / file_name)
        transcript_text = text.strip().lower()  # TODO: remove lower to allow P&C

        output_wav_path = os.path.join(self.resampled_audio_dir, file_name)

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=file_path, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path,
            "duration": float(sox.file_info.duration(output_wav_path)),
            "text": transcript_text,
            "original_file": data_entry[0],
            "age": data_entry[4],
            "gender": data_entry[5],
        }

        return [DataEntry(data=data)]
