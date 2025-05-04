import json
import os
import re

def load_atis_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def substitute_variables(text, variables):
    for name, value in variables.items():
        text = text.replace(name, value)
    return text

def select_shortest_sql(sql_list):
    return sorted(sql_list, key=lambda x: (len(x), x))[0]

def process_dataset(data, split_key):
    result = {'train': [], 'dev': [], 'test': []}

    for item in data:
        try:
            sql = select_shortest_sql(item['sql'])
        except IndexError:
            continue  # skip items without valid sql list

        if split_key == 'query-split':
            split = item.get('query-split')
            if split not in result:
                continue
            for sentence in item['sentences']:
                variables = sentence.get('variables', {})
                question = substitute_variables(sentence['text'], variables)
                sql_filled = substitute_variables(sql, variables)
                result[split].append({
                    'question': question,
                    'sql': sql_filled
                })

        elif split_key == 'question-split':
            for sentence in item['sentences']:
                if 'question-split' not in sentence:
                    continue
                split = sentence['question-split']
                if split not in result:
                    continue
                variables = sentence.get('variables', {})
                question = substitute_variables(sentence['text'], variables)
                sql_filled = substitute_variables(sql, variables)
                result[split].append({
                    'question': question,
                    'sql': sql_filled
                })

    return result

def save_processed_data(data, output_dir, split_type):
    os.makedirs(output_dir, exist_ok=True)
    for split, samples in data.items():
        output_path = os.path.join(output_dir, f'{split}_{split_type}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
    print(f"[✔] Data saved to: {output_dir} ({split_type})")

def main():
    input_file = '../data/atis.json'
    output_dir = '../processed_data/atis'

    data = load_atis_data(input_file)

    # Question Split
    question_split_data = process_dataset(data, 'question-split')
    save_processed_data(question_split_data, output_dir, 'question_split')

    # Query Split
    query_split_data = process_dataset(data, 'query-split')
    save_processed_data(query_split_data, output_dir, 'query_split')

if __name__ == '__main__':
    main()
