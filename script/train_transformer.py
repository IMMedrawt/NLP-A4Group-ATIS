import json
import os
import re
import argparse
from collections import Counter
from sklearn.metrics import accuracy_score
from models import TransformerTextClassifier

def normalize_sql(sql):
    return re.sub(r'\s+', ' ', sql.strip())

def load_data(split_dir, split_type):
    data = {}
    for mode in ['train', 'dev', 'test']:
        filename = f"{mode}_{split_type}.json"
        path = os.path.join(split_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data[mode] = json.load(f)
    return data

def prepare_data(data):
    texts = [item['question'] for item in data]
    labels = [normalize_sql(item['sql']) for item in data]
    return texts, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['geography', 'atis'], help="Dataset name")
    parser.add_argument('--split', type=str, required=True, choices=['question_split', 'query_split'], help="Split type")
    args = parser.parse_args()

    dataset_name = args.dataset
    split_type = args.split

    input_dir = f'../processed_data/{dataset_name}'
    output_dir = f'../outputs/{dataset_name}_transformer_{split_type}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading dataset: {dataset_name}, split: {split_type}")
    dataset = load_data(input_dir, split_type)
    X_train_texts, y_train = prepare_data(dataset['train'])
    X_test_texts, y_test = prepare_data(dataset['test'])

    label_counts = Counter(y_train)
    print(f"[INFO] label_counts: {len(label_counts)}")
    print("[INFO] Top 5 :", label_counts.most_common(5))

    if dataset_name == 'geography' and split_type == 'question_split':
        min_freq = 3
        valid_sqls = {sql for sql, count in label_counts.items() if count >= min_freq}
        if not valid_sqls:
            print(f"[WARN] No valid SQL templates found with min_freq={min_freq}. Lowering threshold to 1.")
            min_freq = 1
            valid_sqls = {sql for sql, count in label_counts.items() if count >= min_freq}
        if not valid_sqls:
            raise RuntimeError(f"[FATAL] Still no valid SQL templates after relaxing min_freq.")
        X_train_filtered = [x for x, y in zip(X_train_texts, y_train) if y in valid_sqls]
        y_train_filtered = [y for y in y_train if y in valid_sqls]
        print(f"[INFO] Using min_freq={min_freq}, remaining train examples: {len(X_train_filtered)}")
    else:
        X_train_filtered = X_train_texts
        y_train_filtered = y_train
        valid_sqls = set(y_train_filtered)
        print(f"[INFO] No frequency filtering applied. Total train examples: {len(X_train_filtered)}")

    if not X_train_filtered:
        print(f"[WARN] No training examples found in {dataset_name} {split_type}. Skipping training.")
        return

    clf = TransformerTextClassifier()
    clf.train(X_train_filtered, y_train_filtered)

    y_pred = clf.predict(X_test_texts)
    y_pred = [normalize_sql(p) for p in y_pred]
    y_test = [normalize_sql(g) for g in y_test]

    y_test_eval = [g if g in valid_sqls else '__invalid__' for g in y_test]
    y_pred_eval = [p if p in valid_sqls else '__invalid__' for p in y_pred]
    eval_pairs = [(p, g) for p, g in zip(y_pred_eval, y_test_eval) if g != '__invalid__']
    correct = sum(p == g for p, g in eval_pairs)
    acc = correct / len(eval_pairs) if eval_pairs else 0.0

    print(f"[Transformer Classifier] Accuracy on {split_type} (filtered classes): {acc:.4f}")

    output_file = os.path.join(output_dir, f'results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy (valid only): {acc:.4f}\n")
        for q, pred, gold in zip(X_test_texts, y_pred, y_test):
            f.write(f"Q: {q}\nPRED: {pred}\nGOLD: {gold}\n\n")

if __name__ == '__main__':
    main()
