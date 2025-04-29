# evaluator.py

import re

def fill_variables(template, variables):
    """
    Replace placeholders in SQL template with predicted variables.
    Args:
        template (str): SQL template with placeholders like airport_code0
        variables (dict): Mapping from placeholder to actual value, e.g., {"airport_code0": "MKE"}
    Returns:
        sql_filled (str): Completed SQL string
    """
    sql_filled = template
    for var, val in variables.items():
        # Replace all occurrences of the variable name with its value
        sql_filled = sql_filled.replace(f"\"{var}\"", f"\"{val}\"")
        sql_filled = sql_filled.replace(var, f"\"{val}\"")  # in case not inside quotes
    return sql_filled

def normalize_sql(sql):
    """
    Normalize SQL string for comparison:
    - Remove extra whitespace
    - Lowercase
    """
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.lower()
    return sql

def evaluate_sql(predictions, references):
    """
    Compare predicted SQL and reference SQLs.
    Args:
        predictions (list): list of predicted SQL strings
        references (list): list of list of gold SQL strings (multiple correct versions)
    Returns:
        accuracy (float): exact match accuracy
    """
    correct = 0
    total = len(predictions)

    for pred, ref_list in zip(predictions, references):
        pred_norm = normalize_sql(pred)
        ref_norms = [normalize_sql(ref) for ref in ref_list]
        if pred_norm in ref_norms:
            correct += 1

    return correct / total