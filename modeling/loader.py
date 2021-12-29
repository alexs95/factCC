import unicodedata

import nltk
#nltk.download('punkt')
from tqdm import tqdm
import glob
import re
import os
import spacy
import neuralcoref

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


def read_text_file(text_file, mode="r"):
    lines = []
    with open(text_file, mode) as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def untokenize(sentence):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    step1 = sentence.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def load(cnndm, target_dir, coref=False, samples=None):
    cnndm = list(glob.glob(cnndm + '/**/*.story', recursive=True))
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)

    stories = []
    summaries = []
    ids = []
    count = 0

    for file in tqdm(glob.glob(target_dir + '/*.txt'), "Loading stories", total=samples):
        if count == samples:
            break

        storyid = os.path.basename(file).split("_")[0]
        target = os.path.basename(file).split("_")[1][:-4]

        # Add detokenized summary
        summary_file = os.path.join(target_dir, "{}_{}.txt".format(storyid, target))
        summary = read_text_file(summary_file, "rU")
        summary = [untokenize(l) for l in summary]
        summary = [unicodedata.normalize('NFKD', l) for l in summary]
        if coref:
            summary = nlp(' '.join(summary))._.coref_resolved
            summary = nltk.sent_tokenize(summary)
        summaries.append(summary)

        # Add story from original dataset
        article_file = next(f for f in cnndm if storyid in f)
        story, _ = get_art_abs(article_file)
        story = [unicodedata.normalize('NFKD', l) for l in story]
        if coref:
            story = nlp(' '.join(story))._.coref_resolved
            story = nltk.sent_tokenize(story)
        stories.append(story)

        ids.append(storyid)
        count += 1

    return ids, stories, summaries