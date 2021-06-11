
# From PySpellChecker
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_sentence(line):
    lines = line.strip().split(' ')
    new_line = ""
    similar_word = {}
    for l in lines:
        new_line += spell.correction(l) + " "
    # similar_word[l]=spell.candidates(l)
    return new_line