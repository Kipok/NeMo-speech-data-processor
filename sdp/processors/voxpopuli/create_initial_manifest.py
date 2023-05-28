# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import string
import subprocess
from pathlib import Path
from typing import Optional

import sox
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive

VOXPOPULI_URL = "https://github.com/facebookresearch/voxpopuli"


def is_same(orig_word, norm_word):
    # word is the same, except last symbol, which could indicate punctuation
    if orig_word[-1] in string.punctuation and orig_word[:-1].lower() == norm_word.lower():
        return True, 1
    # word is the same, except last symbol, which could indicate punctuation
    # (but by mistake it's been put in norm text)
    if norm_word[-1] in string.punctuation and norm_word[:-1].lower() == orig_word.lower():
        return True, 0
    # word is the same, but casing could be different
    if orig_word.lower() == norm_word.lower():
        return True, 1
    return False, None


def restore_pc(orig_words_list, norm_words_list):
    # copy so not to corrupt
    # merging any commas and dots between numbers right away to simplify logic below
    orig_text = list([re.sub(r'(\d)[\.,](\d)', r"\1\2", word) for word in orig_words_list])
    norm_text = list(norm_words_list)
    # to simplify logic below, so that we can assume last word always matches
    orig_text.append("end_text")
    norm_text.append("end_text")

    idx_orig = 0
    idx_norm = 0
    merged_text = []
    while idx_orig < len(orig_text) and idx_norm < len(norm_text):
        same, is_orig = is_same(orig_text[idx_orig], norm_text[idx_norm])
        if same:
            merged_text.append(orig_text[idx_orig] if is_orig else norm_text[idx_norm])
            idx_orig += 1
            idx_norm += 1
            continue

        # checking if first letter is a number, but the whole word is not - that happens
        # on typos like 37a which should really be 37 a. So fixing those
        # another case is for number + punctuation, like 2017, - handling separately
        # another case is for numbers separated by comma, like this "1,5". Those are spelled out
        # separately in normalized form, so just removing the comma here
        add_punct = ""
        if orig_text[idx_orig][0].isdigit() and not orig_text[idx_orig].isdigit():
            number, word = re.split('(\d+)', orig_text[idx_orig])[1:]
            orig_text[idx_orig] = number
            if word in string.punctuation:
                add_punct = word
            else:
                orig_text.insert(idx_orig + 1, word)

        # another annoying case is if typo ends with number like here "dell'11"
        # same logic, but need to go back to the first check, so doing "continue" below
        if orig_text[idx_orig][-1].isdigit() and not orig_text[idx_orig].isdigit():
            word, number = re.split('(\d+)', orig_text[idx_orig])[:-1]
            orig_text[idx_orig] = word
            orig_text.insert(idx_orig + 1, number)
            continue

        # word is different, but original is a number - take from normalized in this case until
        # get same word again (as number might be represented with multiple words)
        # also handling case for number + punctuation
        while orig_text[idx_orig].isdigit():
            idx_orig += 1

        while idx_norm < len(norm_text) and not is_same(orig_text[idx_orig], norm_text[idx_norm])[0]:
            merged_text.append(norm_text[idx_norm])
            idx_norm += 1

        # if there is any trailing punctuation from last digit, let's add it
        merged_text[-1] = merged_text[-1] + add_punct

    if idx_norm != len(norm_text):
        print(idx_orig, idx_norm, len(orig_text), len(norm_text), orig_text, norm_text, merged_text)
        raise RuntimeError("Something went wrong during merging")

    return " ".join(merged_text[:-1])  # removing end_text token


class CreateInitialManifestVoxpopuli(BaseParallelProcessor):
    """
    Downloads and unzips raw VoxPopuli data for the specified language,
    and creates an initial manifest using the transcripts provided in the
    raw data.

    Args:
        raw_data_dir: the directory where the downloaded data will be/is saved.
        language_id: the language of the data you wish to be downloaded.
        data_split: the data split for which the initial manifest will
            be created.
        resampled_audio_dir: the directory where the resampled (16kHz) wav
            files will be stored.
    """

    def __init__(
        self, raw_data_dir: str, language_id: str, data_split: str, resampled_audio_dir: str, **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language_id = language_id
        self.data_split = data_split
        self.resampled_audio_dir = resampled_audio_dir

    def prepare(self):
        """Downloading data (unless already done)"""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        if not (self.raw_data_dir / "transcribed_data").exists():
            # TODO: some kind of isolated environment?
            if not os.path.exists(self.raw_data_dir / 'voxpopuli'):
                logger.info("Downloading voxpopuli and installing requirements")
                subprocess.run(f"git clone {VOXPOPULI_URL} {self.raw_data_dir / 'voxpopuli'}", check=True, shell=True)
                subprocess.run(
                    f"pip install -r {self.raw_data_dir / 'voxpopuli' / 'requirements.txt'}", check=True, shell=True
                )
            if not os.path.exists(self.raw_data_dir / 'raw_audios'):
                logger.info("Downloading raw audios")
                subprocess.run(
                    f"cd {self.raw_data_dir / 'voxpopuli'} && python -m voxpopuli.download_audios --root {self.raw_data_dir} --subset asr",
                    check=True,
                    shell=True,
                )
            if not os.path.exists(self.raw_data_dir / 'transcribed_data' / self.language_id):
                logger.info("Segmenting and transcribing the data")
                subprocess.run(
                    f"cd {self.raw_data_dir / 'voxpopuli'} && python -m voxpopuli.get_asr_data  --root {self.raw_data_dir} --lang {self.language_id}",
                    check=True,
                    shell=True,
                )

    def read_manifest(self):
        with open(
            self.raw_data_dir / "transcribed_data" / self.language_id / f"asr_{self.data_split}.tsv",
            "rt",
            encoding="utf8",
        ) as fin:
            dataset_entries = fin.readlines()[1:]  # skip header line

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        if len(data_entry.split("\t")) != 8:
            raise RuntimeError(f"have more/less than 7 tabs in line {data_entry}")

        utt_id, raw_text, norm_text, spk_id, _, gender, is_gold_transcript, accent = data_entry.split("\t")
        if self.restore_pc:
            try:
                transcript_text = restore_pc(raw_text.split(), norm_text.split())
            except:
                logger.warning("Failed to restore punctuation! Skipping utterance")
                return []
        else:
            transcript_text = norm_text.strip()
        year = utt_id[:4]

        src_flac_path = os.path.join(self.raw_data_dir, "transcribed_data", self.language_id, year, utt_id + ".ogg")
        tgt_wav_path = os.path.join(self.resampled_audio_dir, utt_id + ".wav")

        if not os.path.exists(os.path.dirname(tgt_wav_path)):
            os.makedirs(os.path.dirname(tgt_wav_path), exist_ok=True)
        if not os.path.exists(tgt_wav_path):
            Transformer().build(src_flac_path, tgt_wav_path)

        data = {
            "audio_filepath": tgt_wav_path,
            "duration": float(sox.file_info.duration(tgt_wav_path)),
            "text": transcript_text,
            "provided_norm_text": norm_text,
            "raw_text": raw_text,
            "spk_id": spk_id,
            "gender": gender,
            "is_gold_transcript": is_gold_transcript,
            "accent": accent,
        }
        return [DataEntry(data=data)]
