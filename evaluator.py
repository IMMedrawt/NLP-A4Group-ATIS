import re

# Replace placeholders with predicted values
def fill_variables(template, variables):
    sql = template
    for var, val in variables.items():
        sql = sql.replace(f'"{var}"', f'"{val}"')
        sql = sql.replace(var, f'"{val}"')
    return sql

# Normalize SQL string for comparison
def normalize_sql(sql):
    sql = sql.strip().lower()
    return re.sub(r'\s+', ' ', sql)

# Compute exact match accuracy for predicted SQLs
def evaluate_sql(predictions, references):
    correct = 0
    for pred, refs in zip(predictions, references):
        norm_pred = normalize_sql(pred)
        norm_refs = [normalize_sql(ref) for ref in refs]
        if norm_pred in norm_refs:
            correct += 1
    return correct / len(predictions) if predictions else 0.0
