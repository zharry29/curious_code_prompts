import pandas as pd


def main():
    # load dataset
    data = pd.read_csv('./performance_4000.csv')
    column_drop = [
        'GPT3-D + code (4000 tokens)',
        'Codex + code (vanilla) (4000 tokens)',
        'Codex + code (var identifier) (4000 tokens)', 
        'Codex + code (var identifier + comments) (4000 tokens)', 
        'Codex + code (class + var identifier+comments) (4000 tokens)'
    ]
    
    for col_name in column_drop:
        data = data.drop(col_name, axis=1)

    for row in data.iterrows():
        cur_task = row[1]['Task']
        print(cur_task)
        performances = row[1].iloc[1:]
        cur_rank = performances.sort_values(ascending=False)
        print(cur_rank)
        print('\n')


if __name__ == '__main__':
    main()
