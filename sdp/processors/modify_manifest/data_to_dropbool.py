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

import collections
import re
from typing import List

from sdp.logging import logger
from sdp.processors.base_processor import DataEntry
from sdp.processors.modify_manifest.modify_manifest import ModifyManifestTextProcessor
from sdp.utils.edit_spaces import remove_extra_spaces
from sdp.utils.get_diff import get_diff, get_diff_with_subs_grouped
from sdp.utils.metrics_computation import (
    get_cer,
    get_charrate,
    get_wer,
    get_wmr,
    get_wordrate,
)


class DropHighLowCharrate(ModifyManifestTextProcessor):
    """Drops utterances if their character rate is too low or too high.

    Character rate = ``(num of characters in self.text_key) / (duration of audio)``.
    A too-low or too-high character rate often implies that the ground
    truth transcription is inaccurate.

    Args:
        high_charrate_threshold (float): upper character rate threshold.
            If the character rate of an utterance is higher than this number,
            the utterance will be dropped.
        low_charrate_threshold (float): lower character rate threshold.
            If the character rate of an utterance is lower than this number,
            the utterance will be dropped.
    """

    def __init__(
        self,
        high_charrate_threshold: float,
        low_charrate_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.high_charrate_threshold = high_charrate_threshold
        self.low_charrate_threshold = low_charrate_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        """Drops utterances based on the provided thresholds."""
        charrate = get_charrate(remove_extra_spaces(data_entry[self.text_key]), data_entry["duration"])
        if charrate > self.high_charrate_threshold:
            return [DataEntry(data=None, metrics=(0, 1))]
        elif charrate < self.low_charrate_threshold:
            return [DataEntry(data=None, metrics=(1, 0))]

        return [DataEntry(data=data_entry, metrics=(0, 0))]

    def finalize(self, metrics):
        """Will report how many utterances were dropped for each threshold."""
        high_drop_counter = 0
        low_drop_counter = 0
        for dropped_low, dropped_high in metrics:
            low_drop_counter += dropped_low
            high_drop_counter += dropped_high
        logger.info(
            "Num of utterances that were dropped due to char rate > %f: %d",
            self.high_charrate_threshold,
            high_drop_counter,
        )

        logger.info(
            "Num of utterances that were dropped due to char rate < %f: %d",
            self.low_charrate_threshold,
            low_drop_counter,
        )
        super().finalize(metrics)


class DropHighLowWordrate(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if their word rate is
    too low or too high. Word rate = (num of words in self.text_key)/
    (duration of audio).
    A too-low or too-high word rate often implies that the ground
    truth text is inaccurate.

    Args:
        high_wordrate_threshold: a float for the upper word rate threshold.
            If the word rate of an utterance is higher than this number,
            the utterance will be dropped.
        low_wordrate_threshold: a float for the lower word rate threshold.
            If the word rate of an utterance is lower than this number,
            the utterance will be dropped.
    """

    def __init__(
        self,
        high_wordrate_threshold: float,
        low_wordrate_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.high_wordrate_threshold = high_wordrate_threshold
        self.low_wordrate_threshold = low_wordrate_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        wordrate = get_wordrate(data_entry[self.text_key], data_entry["duration"])
        if wordrate > self.high_wordrate_threshold:
            return [DataEntry(data=None, metrics=(0, 1))]
        elif wordrate < self.low_wordrate_threshold:
            return [DataEntry(data=None, metrics=(1, 0))]

        return [DataEntry(data=data_entry, metrics=(0, 0))]

    def finalize(self, metrics):
        high_drop_counter = 0
        low_drop_counter = 0
        for dropped_low, dropped_high in metrics:
            low_drop_counter += dropped_low
            high_drop_counter += dropped_high
        logger.info(
            "Num of utterances that were dropped due to word rate > %f: %d",
            self.high_wordrate_threshold,
            high_drop_counter,
        )
        logger.info(
            "Num of utterances that were dropped due to word rate < %f: %d",
            self.low_wordrate_threshold,
            low_drop_counter,
        )
        super().finalize(metrics)


class DropHighLowDuration(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if their audio duration
    (in seconds) is too low or too high.

    Args:
        high_duration_threshold: a float for the upper duration threshold (in seconds).
            If the duration of an utterance's audio is higher than this number,
            the utterance will be dropped.
        low_duration_threshold: a float for the lower duration threshold (in seconds).
            If the duration of an utterance's audio is lower than this number,
            the utterance will be dropped.
    """

    def __init__(
        self,
        high_duration_threshold: float,
        low_duration_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.high_duration_threshold = high_duration_threshold
        self.low_duration_threshold = low_duration_threshold
        self.high_drop_counter = 0
        self.low_drop_counter = 0

    def _process_dataset_entry(self, data_entry) -> List:
        duration = data_entry["duration"]
        if duration > self.high_duration_threshold:
            return [DataEntry(data=None, metrics=(0, 1))]
        elif duration < self.low_duration_threshold:
            return [DataEntry(data=None, metrics=(1, 0))]

        return [DataEntry(data=data_entry, metrics=(0, 0))]

    def finalize(self, metrics):
        high_drop_counter = 0
        low_drop_counter = 0
        for dropped_low, dropped_high in metrics:
            low_drop_counter += dropped_low
            high_drop_counter += dropped_high
        logger.info(
            "Num of utterances that were dropped due to duration > %f: %d",
            self.high_duration_threshold,
            high_drop_counter,
        )
        logger.info(
            "Num of utterances that were dropped due to duration < %f: %d",
            self.low_duration_threshold,
            low_drop_counter,
        )
        super().finalize(metrics)


class DropIfNoneOfRegexMatch(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if data[self.text_attribute] does
    not match any of regex_patterns.
    Args:
        regex_patterns: a list of strings. If data_entry[self.attribute] does not match any
            of the regex patterns in the list, that utterance will be dropped.
    """

    def __init__(
        self,
        regex_patterns: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_patterns = regex_patterns

    def _process_dataset_entry(self, data_entry) -> List:
        for regex_pattern in self.regex_patterns:
            if re.search(regex_pattern, data_entry[self.text_key]):
                break
        else:  # will only reach this if none of the regex match
            return [DataEntry(data=None, metrics=1)]

        # will reach this part of code if at least one of the regexes matches
        return [DataEntry(data=data_entry, metrics=0)]

    def finalize(self, metrics):
        total_counter = 0
        for value in metrics:
            if value:
                total_counter += value
        logger.info("Num of utterances that were dropped due to not containing any of the specified regex patterns")
        logger.info(f"{total_counter}")
        super().finalize(metrics)


class DropNonAlphabet(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if they contain characters that
    are not in our "alphabet".

    Args:
        alphabet: a string containing all of the characters in our alphabet.
            If an utterance contains at least one character that is not in the
            'alphabet', then that utterance will be dropped.
            Note: don't forget to include spaces in your alphabet, unless you
            want to make sure none of the utterances contain spaces.
    """

    def __init__(
        self,
        alphabet: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alphabet = alphabet

    def _process_dataset_entry(self, data_entry) -> List:
        drop_this_utt = False
        non_alphabet_counter = collections.defaultdict(int)
        for char in data_entry[self.text_key]:
            if char not in self.alphabet:
                drop_this_utt = True
                non_alphabet_counter[char] += 1
        if drop_this_utt:
            return [DataEntry(data=None, metrics=non_alphabet_counter)]
        return [DataEntry(data=data_entry, metrics=non_alphabet_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for char, value in counter.items():
                total_counter[char] += value
        logger.info("Num of non-alphabet characters")
        for char, count in total_counter.items():
            logger.info(f"{char}: {count}")
        super().finalize(metrics)


class DropASRErrorBeginningEnd(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if there is a sufficiently long
    ASR mismatch at the beginning or end of the utterance.

    Args:
        beginning_error_char_threshold: if there is an insertion or deletion at
            the beginning of the utterance that has more characters than this number,
            then the utterance will be dropped.
            If there is a substitution at the beginning of the utterance, then the
            utterance will be dropped if abs(len(deletion) - len(insertion)) >
            beginning_error_char_threshold.
        end_error_char_threshold: if there is an insertion or deletion at
            the end of the utterance that has more characters than this number,
            then the utterance will be dropped.
            If there is a substitution at the end of the utterance, then the
            utterance will be dropped if abs(len(deletion) - len(insertion)) >
            end_error_char_threshold.
    """

    def __init__(
        self,
        beginning_error_char_threshold: int,
        end_error_char_threshold: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beginning_error_char_threshold = beginning_error_char_threshold
        self.end_error_char_threshold = end_error_char_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]

        # remove spaces at start and end. Otherwise all utterances
        # will have no errors at the begining (because both self.text_key
        # and self.pred_text_key will begin with " ")
        orig_words = remove_extra_spaces(orig_words)
        pred_words = remove_extra_spaces(pred_words)

        diff = get_diff_with_subs_grouped(orig_words, pred_words)

        if len(diff) > 0:  # i.e. if there are differences between text and pred_text
            first_diff_entry = diff[0]
            if first_diff_entry[0] == 1 or first_diff_entry[0] == -1:  # i.e. diff is purely an insertion or deletion
                if len(first_diff_entry[1]) > self.beginning_error_char_threshold:
                    return [DataEntry(data=None, metrics=(1, 0))]
            elif first_diff_entry[0] != 0:  # i.e. diff should be a tuple representing substitution
                len_deletion = len(first_diff_entry[0][1])
                len_insertion = len(first_diff_entry[1][1])
                if abs(len_deletion - len_insertion) > self.beginning_error_char_threshold:
                    return [DataEntry(data=None, metrics=(1, 0))]

            last_diff_entry = diff[-1]
            if last_diff_entry[0] == 1 or last_diff_entry[0] == -1:  # i.e. diff is purely an insertion or deletion
                if len(last_diff_entry[1]) > self.end_error_char_threshold:
                    return [DataEntry(data=None, metrics=(0, 1))]
            elif last_diff_entry[0] != 0:  # i.e. diff should be a tuple representing substitution
                len_deletion = len(last_diff_entry[0][1])
                len_insertion = len(last_diff_entry[1][1])
                if abs(len_deletion - len_insertion) > self.end_error_char_threshold:
                    return [DataEntry(data=None, metrics=(0, 1))]

        return [DataEntry(data=data_entry, metrics=(0, 0))]

    def finalize(self, metrics):
        beginning_drop_counter = 0
        end_drop_counter = 0
        for dropped_beginning, dropped_end in metrics:
            beginning_drop_counter += dropped_beginning
            end_drop_counter += dropped_end
        logger.info(
            "Num of utterances that were dropped due to asr " "insertions/deletions at the beginning: %d",
            beginning_drop_counter,
        )
        logger.info(
            "Num of utterances that were dropped due to asr insertions/deletions at the end: %d",
            end_drop_counter,
        )
        super().finalize(metrics)


# TODO: needs unification with above class in some way
class DropASRError(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if there is a sufficiently long
    ASR mismatch anywhere in the utterance.

    Args:
        consecutive_words_threshold (int): will drop if there is a mismatch of
            at least this many words in a row.
    """

    def __init__(
        self,
        consecutive_words_threshold: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.consecutive_words_threshold = consecutive_words_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]
        diffs = get_diff(orig_words, pred_words)

        for diff_entry in diffs:
            if diff_entry[0] == 0:
                continue
            if len(diff_entry[1].split()) >= self.consecutive_words_threshold:
                return []

        return [DataEntry(data=data_entry)]


class DropHighCER(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if there is a sufficiently
    high CER between data[self.text_key] and data[self.pred_text_key].
    Note: we only drop the utterance if CER > threshold (ie strictly greater
    than) so that if we set the threshold to 0, we will not remove
    utterances with CER == 0.

    Args:
        cer_thershold: CER threshold above which the utterance will be dropped.
    """

    def __init__(
        self,
        cer_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cer_threshold = cer_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        cer = get_cer(
            remove_extra_spaces(data_entry[self.text_key]),
            remove_extra_spaces(data_entry[self.pred_text_key]),
        )
        if cer > self.cer_threshold:
            return [DataEntry(data=None, metrics=1)]
        else:
            return [DataEntry(data=data_entry, metrics=0)]

    def finalize(self, metrics):
        drop_counter = 0
        for dropped in metrics:
            drop_counter += dropped
        logger.info(
            "Num of utterances that were dropped due to CER > %d: %d",
            self.cer_threshold,
            drop_counter,
        )
        super().finalize(metrics)


class DropHighWER(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if there is a sufficiently
    high WER between data[self.text_key] and data[self.pred_text_key].
    Note: we only drop the utterance if CER > threshold (ie strictly greater
    than) so that if we set the threshold to 0, we will not remove
    utterances with WER == 0.

    Args:
        wer_thershold: WER threshold above which the utterance will be dropped.
    """

    def __init__(
        self,
        wer_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wer_threshold = wer_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        wer = get_wer(data_entry[self.text_key], data_entry[self.pred_text_key])
        if wer > self.wer_threshold:
            return [DataEntry(data=None, metrics=1)]
        else:
            return [DataEntry(data=data_entry, metrics=0)]

    def finalize(self, metrics):
        drop_counter = 0
        for dropped in metrics:
            drop_counter += dropped
        logger.info(
            "Num of utterances that were dropped due to WER > %d: %d",
            self.wer_threshold,
            drop_counter,
        )
        super().finalize(metrics)


class DropLowWordMatchRate(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if there is a sufficiently
    low WMR between data[self.text_key] and data[self.pred_text_key].
    Note: we only drop the utterance if WMR < threshold (ie strictly lower
    than) so that if we set the threshold to 100, we will not remove
    utterances with WMR == 100.

    Args:
        wmr_thershold: WMR threshold below which the utterance will be dropped.
    """

    def __init__(
        self,
        wmr_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wmr_threshold = wmr_threshold

    def _process_dataset_entry(self, data_entry) -> List:
        orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]
        orig_words = remove_extra_spaces(orig_words)
        pred_words = remove_extra_spaces(pred_words)
        wmr = get_wmr(orig_words, pred_words)
        if wmr < self.wmr_threshold:
            return [DataEntry(data=None, metrics=1)]
        else:
            return [DataEntry(data=data_entry, metrics=0)]

    def finalize(self, metrics):
        drop_counter = 0
        for dropped in metrics:
            drop_counter += dropped
        logger.info(
            "Num of utterances that were dropped due to WMR < %d: %d",
            self.wmr_threshold,
            drop_counter,
        )
        super().finalize(metrics)


class DropIfRegexMatch(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if data[self.text_key] matches
    a regex pattern.

    Args:
        regex_patterns: a list of strings. The list will be traversed in order.
            If data_entry.data[self.text_key] matches the regex, the entry will be dropped.
    """

    def __init__(
        self,
        regex_patterns: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regex_patterns = regex_patterns

    def _process_dataset_entry(self, data_entry) -> List:
        drop_counter = collections.defaultdict(int)
        for regex_pattern in self.regex_patterns:
            if re.search(regex_pattern, data_entry[self.text_key]):
                for match in re.finditer(regex_pattern, data_entry[self.text_key]):
                    drop_counter[regex_pattern] += 1
                return [DataEntry(data=None, metrics=drop_counter)]
        return [DataEntry(data=data_entry, metrics=drop_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for attribute, value in counter.items():
                total_counter[attribute] += value
        logger.info("Regex matches that were dropped in attribute")
        for attribute, matches in total_counter.items():
            logger.info(f"{attribute}, {matches}")
        super().finalize(metrics)


class DropOnAttribute(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if attribute is set to True/False.

    Args:
        key (str): which key to use for dropping utterances.
        drop_if_false (bool): whether to drop if value is False. Defaults
            to dropping if True.
    """

    # TODO: maybe need not to subclass from the base here, because text_key/pred_text_key are not used.
    #    But still want to leverage test-cases functionality. Probably need some redesign of API here.

    def __init__(
        self,
        key: str,
        drop_if_false: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key = key
        self.drop_if_false = drop_if_false

    def _process_dataset_entry(self, data_entry) -> List:
        if data_entry[self.key] is not self.drop_if_false:
            return [DataEntry(data=None, metrics=1)]
        return [DataEntry(data=data_entry, metrics=0)]

    def finalize(self, metrics):
        total_counter = 0
        for counter in metrics:
            total_counter += counter
        logger.info("Dropped %d utterances", total_counter)
        super().finalize(metrics)


class DropIfSubstringInInsertion(ModifyManifestTextProcessor):
    """
    Class for processor that drops utterances if a substring matches an insertion
    made between data[self.text_key] and data[self.pred_text_key].
    Note: we check for exact matches, so you need to be mindful of spaces, e.g.
    you may wish to do substrings_in_insertion = ["nemo ", ...] instead
    of substrings_in_insertion = ["nemo", ...]

    Args:
        substrings_in_insertion: a list of strings which might be inserted in predicted
            ASR text. If the insertion matches a string exactly, the utterance will
            be dropped.
    """

    def __init__(
        self,
        substrings_in_insertion: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.substrings_in_insertion = substrings_in_insertion

    def _process_dataset_entry(self, data_entry) -> List:
        for substring_in_insertion in self.substrings_in_insertion:
            if substring_in_insertion in data_entry[self.pred_text_key]:
                orig_words, pred_words = data_entry[self.text_key], data_entry[self.pred_text_key]
                diff = get_diff_with_subs_grouped(orig_words, pred_words)

                for diff_entry in diff:
                    if diff_entry[0] == 1:  # insertion in original string
                        if substring_in_insertion in diff_entry[1]:
                            return [DataEntry(data=None, metrics=diff_entry[1])]
        return [DataEntry(data=data_entry, metrics="")]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for diff_entry in metrics:
            if diff_entry:
                total_counter[diff_entry] += 1
        logger.info("Some of the insertions that cause the utterance to be dropped:")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))

        for insertion, count in total_counter_sorted.items():
            logger.info(f"{insertion}, {count}")
        super().finalize(metrics)
