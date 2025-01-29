import pandas as pd
from sklearn.utils import shuffle

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('spanish'))

def get_min_num_class(df, class_val, num_data, name_col):
  df = df[df[name_col] == class_val]
  df = shuffle(df)
  return df[:num_data]

def remove_stop_words(text):
  without_stop_words = [word for word in text.split() if word.lower() not in STOP_WORDS]
  without_stop_words = ' '.join(without_stop_words)
  return without_stop_words

def preprocessing_dataframe(df_revisado, remove_stop_words):
    df = df_revisado.copy()
    #Transforma las etiquetas a 0..len(NUM_LABELS) porque el modelo accede mediante el indice
    df['Score_G'] = df_revisado['Score_G'].apply(lambda x: x-1)

    min_data_class = list(df['Score_G'].value_counts(ascending=True))[0]
    df_revisado_eq = pd.DataFrame()
    #Obtiene el mismo número de ejemplos para cada clase
    for num in df['Score_G'].unique():
        df_num = get_min_num_class(df, num, min_data_class, 'Score_G')
        df_revisado_eq = pd.concat([df_revisado_eq, df_num], axis=0)

    df_revisado_eq = shuffle(df_revisado_eq)

    if remove_stop_words:
       #Elimina las stop words
       df_revisado_eq['Review'] = df_revisado_eq['Review'].apply(remove_stop_words)
    
    return df_revisado_eq
       



