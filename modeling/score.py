from pytorch_transformers import (BertForSequenceClassification, BertTokenizer)
from utils import convert_examples_to_features, InputExample
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
import torch.nn.functional as F
from fastparquet import write
from pathlib import Path
from loader import load
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import uuid
import os


MODEL_NAME = "bert-base-uncased"


class FactCC:
    def __init__(self, checkpoint, path, batch_size, max_seq_length, gpu, method="sentence"):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.method = method

        # Configure GPU
        self.gpu = None
        if gpu:
            self.gpu = torch.cuda.current_device()

        # Configure paths
        self.path = path
        self.features_dir = os.path.join(self.path, "features")
        self.features_path = os.path.join(self.features_dir, "features.pkl")
        self.dataset_dir = os.path.join(self.path, "dataset")
        self.scores_dir = os.path.join(self.path, "scores")
        self._create_dirs()

        # Load model
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = None

    def _create_dirs(self):
        Path(self.features_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
        Path(self.scores_dir).mkdir(parents=True, exist_ok=True)

    def _create_examples(self, ids, stories, summaries):
        examples = []
        if ids is None:
            ids = range(len(stories))

        for storyid, story, summary in zip(ids, stories, summaries):
            for claimid, claim in enumerate(summary):
                label = "CORRECT"  # redundant, just so it works with the factCC code
                if self.method == "sentence":
                    for sentid, sentence in enumerate(story):
                        text_a = sentence.strip()
                        text_b = claim.strip()
                        examples.append(
                            InputExample(storyid=storyid, claimid=claimid, sentid=sentid, text_a=text_a, text_b=text_b, label=label)
                        )
                else:
                    text_a = " ".join(s.strip() for s in story)
                    text_b = claim.strip()
                    examples.append(
                        InputExample(storyid=storyid, claimid=claimid, sentid=0, text_a=text_a, text_b=text_b, label=label)
                    )

        return examples

    def _to_tensor_dataset(self, features):
        storyids = [f.storyid for f in features]
        claimids = [f.claimid for f in features]
        sentids = [f.sentid for f in features]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_ext_mask = torch.tensor([f.extraction_mask for f in features], dtype=torch.float)
        all_ext_start_ids = torch.tensor([f.extraction_start_ids for f in features], dtype=torch.long)
        all_ext_end_ids = torch.tensor([f.extraction_end_ids for f in features], dtype=torch.long)
        all_aug_mask = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float)
        all_aug_start_ids = torch.tensor([f.augmentation_start_ids for f in features], dtype=torch.long)
        all_aug_end_ids = torch.tensor([f.augmentation_end_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
            all_ext_mask, all_ext_start_ids, all_ext_end_ids,
            all_aug_mask, all_aug_start_ids, all_aug_end_ids,
        )

        return dataset, storyids, claimids, sentids

    def _save_dataset(self, examples, format):
        df = pd.DataFrame({
            "storyid": (e.storyid for e in examples),
            "sentid": (e.sentid for e in examples),
            "claimid": (e.claimid for e in examples),
            "sent": (e.text_a for e in examples),
            "claim": (e.text_b for e in examples)
        })
        if format == 'csv':
            df.to_csv(
                os.path.join(self.dataset_dir, "data.csv.gz"),
                index=False,
                compression="gzip",
                float_format='%.5f'
            )
        elif format == 'parquet':
            write(
                data=df,
                row_group_offsets=5000000,
                file_scheme="hive",
                filename=self.dataset_dir,
                compression='SNAPPY'
            )

    def _save_batch_scores(self, batch_scores):
        batch_scores.to_csv(
            os.path.join(self.scores_dir, "{}.csv.gz".format(uuid.uuid4())),
            index=False,
            compression="gzip",
            float_format='%.5f'
        )

    def preprocess(self, ids, stories, summaries, dataset_format='csv'):
        examples = self._create_examples(ids, stories, summaries)
        features = convert_examples_to_features(
            examples=examples,
            label_list=["CORRECT", "INCORRECT"],  # redundant, just so it works with the factCC code
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            output_mode="classification",
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )

        self._save_dataset(examples, dataset_format)
        torch.save(features, self.features_path)

    def run(self):
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint)
        self.model.to(self.gpu)

        # Load and batch features
        features = torch.load(self.features_path)
        dataset, storyids, claimids, sentids = self._to_tensor_dataset(features)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        # Evaluate batches
        count = 0
        scores = None
        for batch in tqdm(dataloader, "Running batches"):
            self.model.eval()
            batch = tuple(example.to(self.gpu) for example in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                }
                outputs = self.model(**inputs)
                logits = outputs[1]
                logits = F.softmax(logits, dim=1)
                preds = logits.detach().cpu().numpy()

                # Calculate and save batch scores
                batch_scores = pd.DataFrame({
                    "storyid": storyids[count:count + len(preds)],
                    "claimid": claimids[count:count+len(preds)],
                    "sentid": sentids[count:count+len(preds)],
                    "score": preds[:, 0]
                })
                self._save_batch_scores(batch_scores)

                # Append to overall scores
                batch_scores = batch_scores.groupby(['storyid', 'claimid']).max()
                if scores is None:
                    scores = batch_scores
                else:
                    scores = scores.append(batch_scores)

                count += len(preds)

        return scores.groupby(['storyid', 'claimid']).max()['score'].mean()


def parse_args():
    base = Path(__file__).parent.parent.resolve()
    evaluation = os.path.join(base, "evaluation", "1000_sample")
    checkpoint = os.path.join(base, "checkpoint/factcc-checkpoint")
    data = os.path.join(base, "../data/cnndm/")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    parser.add_argument('--data', type=str, help='Path to cnn/dm dataset.', default=data)
    parser.add_argument('--evaluation', type=str, help='Path to evaluation directory.', default=evaluation)
    parser.add_argument('--checkpoint', type=str, help='Path to factcc checkpoint directory.', default=checkpoint)
    parser.add_argument('--mode', type=str, help='Evaluate or preprocess.', choices=["preprocess", "evaluate"])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    factCC = FactCC(
        checkpoint=args.checkpoint,
        path=args.evaluation,
        gpu=args.gpu,
        batch_size=512,
        max_seq_length=12,
        method="sentence"
    )
    if args.mode == "preprocess":
        ids, stories, summaries = load(args.data, 1000)
        factCC.preprocess(ids, stories, summaries, dataset_format='parquet')
    else:
        # Need to add a loading bar for loading the data.
        score = factCC.run()
        print(score)