import regex
from typing import List

def split_text_into_sentences(lang, text: str) -> List[str]:
    """
    Split text into sentences.
    Args:
        text: text
    Returns list of sentences
    """
    lower_case_unicode = ''
    upper_case_unicode = ''
    if lang == "ru":
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'

    # Read and split transcript by utterance (roughly, sentences)
    split_pattern = rf"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]+\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s(?![0-9]+[a-z]*\.)"

    sentences = regex.split(split_pattern, text)
    return sentences


if __name__ == "__main__":
    print(split_text_into_sentences('en', "so. hello. this is me testing, right? ah"))
    print(split_text_into_sentences('en', " so. ¿ hello. this is me testing, right? ah"))
    print(split_text_into_sentences('en', "so. ¿hello. this is me güe testing, á é í ó ú right ñ⟩? ah"))