import os
import glob
from tqdm import tqdm

from factCC.modeling.utils import InputExample

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
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
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    return article_lines, highlights


def load(stories_dir, sample=None):
    stories = []
    summaries = []
    ids = []
    count = 0
    for file in tqdm(glob.glob(stories_dir + '**/*.story', recursive=True), "Loading stories", total=sample):
        if count == sample:
            break
        story, summary = get_art_abs(file)
        stories.append(story)
        summaries.append(summary)
        ids.append(os.path.basename(file))
        count += 1

    return ids, stories, summaries

    # nlp = spacy.load('en')
    # neuralcoref.add_to_pipe(nlp)
    # story = nlp(story)
    # summary = nlp(summary)
