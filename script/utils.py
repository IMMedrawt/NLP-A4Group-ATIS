import re

def normalize_sql(sql):
    return re.sub(r'\s+', ' ', sql.strip())