import json
import os
import re

def load_geography_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def substitute_variables(text, variables):
    for var_name, var_value in variables.items():
        text = text.replace(var_name, var_value)
    return text

def select_shortest_sql(sql_list):
    sorted_sql = sorted(sql_list, key=lambda x: (len(x), x))
    return sorted_sql[0]

def process_dataset(data, split_type):
    processed_data = {'train': [], 'dev': [], 'test': []}

    for item in data:
        sql_query = select_shortest_sql(item['sql'])
        for sentence in item['sentences']:
            if split_type not in sentence:
                continue
            split = sentence[split_type]
            question = substitute_variables(sentence['text'], sentence['variables'])
            sql = substitute_variables(sql_query, sentence['variables'])
            processed_data[split].append({'question': question, 'sql': sql})

    return processed_data

def save_processed_data(processed_data, output_dir, split_type):
    os.makedirs(output_dir, exist_ok=True)
    for split, examples in processed_data.items():
        output_path = os.path.join(output_dir, f'{split}_{split_type}.json')
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(examples, file, indent=2)
    print(f"Data successfully saved in directory: {output_dir}")

def main():
    geography_filepath = '../data/geography.json'
    output_directory = '../processed_data/geography'

    data = load_geography_data(geography_filepath)

    # Question Split processing
    question_split_data = process_dataset(data, 'question-split')
    save_processed_data(question_split_data, output_directory, 'question_split')

    # Query Split processing
    query_split_data = process_dataset(data, 'query-split')
    if any(len(v) for v in query_split_data.values()):
        save_processed_data(query_split_data, output_directory, 'query_split')
    else:
        print("Warning: 'query-split' data does not exist in geography.json, skipping query-split processing.")

if __name__ == '__main__':
    main()
