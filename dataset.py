import json
import torch
from torch.utils.data import Dataset
from collections import Counter
from vocab import build_token_vocab, build_tag_vocab

class ATISDataset(Dataset):
    def __init__(self, data_path, token2id, tag2id, split_type='question-split', mode='train', max_len=128):
        """
        Load the ATIS dataset.
        Args:
            data_path (str): Path to the atis.json file.
            token2id (dict): Mapping from tokens to their IDs.
            tag2id (dict): Mapping from tags to their IDs.
            split_type (str): 'question-split' or 'query-split'.
            mode (str): 'train', 'dev', or 'test'.
            max_len (int): Maximum sequence length.
        """
        self.token2id = token2id
        self.tag2id = tag2id
        self.max_len = max_len

        # Load the raw json data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        # Process the data to create samples
        self.samples = self.build_samples(raw_data, split_type, mode)

    def build_samples(self, raw_data, split_type, mode):
        """
        Build the sample list according to split type and mode.
        Each sample contains: tokenized text, token labels, template id, variable values, raw SQL.
        """
        samples = []

        for item in raw_data:
            if 'sentences' not in item:
                continue

            for sent in item['sentences']:
                if sent[split_type] != mode:
                    continue

                tokens = sent['text'].split()  # simple whitespace tokenization
                variables = sent['variables']

                # Prepare tagging labels for each token
                tags = self.create_tags(tokens, variables)

                # Pick the shortest SQL template from sql list
                sql_templates = item['sql']
                sql_template = min(sql_templates, key=lambda x: (len(x), x))

                samples.append({
                    'tokens': tokens,
                    'tags': tags,
                    'template': sql_template,
                    'variables': variables
                })

        return samples

    def create_tags(self, tokens, variables):
        """
        Create tag list for each token.
        Default tag is 'O' (outside).
        If a token matches a variable value, tag it with the variable name.
        """
        tags = ['O'] * len(tokens)

        # Simple matching: exact match token to variable value
        for var_name, var_value in variables.items():
            var_tokens = var_value.split()
            for i in range(len(tokens) - len(var_tokens) + 1):
                if tokens[i:i+len(var_tokens)] == var_tokens:
                    for j in range(len(var_tokens)):
                        tags[i+j] = var_name

        return tags

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get one sample.
        Return input_ids, tag_ids, template_text, variables.
        """
        sample = self.samples[idx]
        tokens = sample['tokens']
        tags = sample['tags']
        template = sample['template']
        variables = sample['variables']

        # Encode tokens and tags
        input_ids = [self.token2id.get(t, self.token2id['<UNK>']) for t in tokens]
        tag_ids = [self.tag2id.get(tag, self.tag2id['O']) for tag in tags]

        # Pad or truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        else:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.token2id['<PAD>']] * pad_len
            tag_ids += [self.tag2id['O']] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'tag_ids': torch.tensor(tag_ids, dtype=torch.long),
            'template': template,  # return template text for template prediction task
            'variables': variables, # keep variables for later use in SQL reconstruction
            'tokens': tokens        # optionally keep original tokens
        }



"""
if __name__ == "__main__":
    # Example usage and test
    token2id = build_token_vocab("atis.json")
    tag2id = build_tag_vocab("atis.json")

    dataset = ATISDataset(
        data_path="atis.json",
        token2id=token2id,
        tag2id=tag2id,
        split_type='question-split',
        mode='train',
        max_len=32
    )

    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print("Sample input_ids:", sample['input_ids'])
    print("Sample tag_ids:", sample['tag_ids'])
    print("Sample template:", sample['template'])
    print("Sample variables:", sample['variables'])
    print("Sample tokens:", sample['tokens'])
"""