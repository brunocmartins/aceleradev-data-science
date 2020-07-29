import pandas as pd

# Import challenge data
df = pd.read_csv('desafio1.csv')

# Create 'pontuacao_credito' mode by state
mode = df.groupby('estado_residencia')['pontuacao_credito'].apply(lambda x: x.mode()[0]).reset_index()
# Create dataframe with mean, median and standart deviation of 'pontuacao_credito' by state
pivot_table = df.pivot_table(values='pontuacao_credito', index='estado_residencia',
                       aggfunc=['mean', 'median', 'std']).round(2)
# Rename columns from multiindex columns to standard index columns
pivot_table.columns = ['media', 'mediana', 'desvio_padrao']

# Merge 'mode' and 'pivot_table' dataframes
solution = mode.merge(pivot_table, on='estado_residencia', how='left')
# Rename columns to its respective function
solution.rename({'pontuacao_credito': 'moda'},
                 axis=1, inplace=True)

# Export data to json file
solution.set_index('estado_residencia').to_json('submission.json', orient='index')