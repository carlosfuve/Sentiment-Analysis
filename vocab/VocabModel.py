import string, re
from collections import defaultdict


class MyVocabModel():
    def __init__(self, corpus):
        self.signos_punt_keep = ',.?!"()+-:;%*€'
        self.signos_punt_rem = re.sub(f"[{re.escape(self.signos_punt_keep)}]",'',string.punctuation) + '•…\ufeff“”»—–’º°'
        self.chinese_pat = r'[\u4e00-\u9fff]+'
        self.emoji_pat = re.compile("["
                            u"\U0001F600-\U0001F6FF" 
                            u"\U0001F300-\U0001F5FF"  
                            u"\U0001F700-\U0001F77F"
                            u"\U0001F900-\U0001F9FF"
                                    "]+", flags=re.UNICODE)


        #Transforma los textos a listas de textos separados por espacios
        corpus = corpus.apply(self.transform_txt) 

        self.words_freq = self.get_freq_words(corpus)
        self.vocab, self.splits = self.get_vocab_ini()
    
    def get_freq_words(self,corpus):
        word_freqs = defaultdict(int)
        #Calcula el número de veces que aparece una letra en el corpus
        for text in corpus:
            for word in text:
                word_freqs[word] += 1
        return word_freqs
    
    def get_vocab_ini(self):
        alphabet = []
        #Genera todos los caracteres posibles del corpus, de longitud 1
        for word in self.words_freq.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
  
            alphabet.sort()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

        #Genera un diccionario palabra: partición en tokens del corpus
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.words_freq.keys()
        }

        return vocab, splits 

    def transform_txt(self, frase): 
        frase = re.sub(self.chinese_pat,'',frase) #Elimina los caracteres chinos
        frase = re.sub(self.emoji_pat, '', frase) #Elimina los emoticonos
        frase = frase.lower()
        frase = re.sub(f"[{re.escape(self.signos_punt_rem)}]",'',frase) #Elimina los signos de puntuación que no me interesan
        frase_punt_space = re.sub(f"[{re.escape(self.signos_punt_keep)}]", r' \g<0> ', frase) #Añade espacio delante y atrás a los signos de puntuación
        frase_no_double_space = re.sub(r'\s+', ' ', frase_punt_space) #Elimina los dobles espacios
        return frase_no_double_space.strip().split(" ")
    
    def new_token(self,a,b):
        return a+re.sub("##",'',b)
    
    def calc_pair_score(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        for pal, freq in self.words_freq.items():
            split = self.splits[pal]

            if len(split) == 1:
                #Si la palabra es una letra no es posible generar un pair
                letter_freqs[split[0]] += freq 
                continue
        
            for i in range(len(split)-1):
                #Genera una subpalabra secuencial
                pair = (split[i], split[i + 1])

                #El acceder por split[i] ya diferencia entre a o ##a
                letter_freqs[split[i]] += freq
                #La frencia es el número de apariciones de la subpalabra en el corpus, la misma que la frecuencia de la palabra
                pair_freqs[pair] += freq 
        
            letter_freqs[split[-1]] += freq

        #Calcula la probabilidad conjunta de la unión de los elementos pair[0] y pair[1]
        scores = defaultdict(int)
        for pair, freq in pair_freqs.items():
            scores[pair] = freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
    
        return scores
    
    def recal_splits(self,pair_max):
        a,b = pair_max
        #Genera los nuevos splits con el nuevo token del diccionario
        for pal in self.words_freq:
            split = self.splits[pal]
            if len(split) > 1:
                i = 0
                while i < len(split) - 1:
                    #Si coincide con el nuevo token, modifica el split
                    if split[i] == a and split[i+1] == b:
                        split = split[:i] + [self.new_token(a,b)] + split[i+2:]
                    else:
                        i += 1
            self.splits[pal] = split


    def create_vocab(self, max_len_vocab: int):
        while len(self.vocab) < max_len_vocab:
            pair_score = self.calc_pair_score()
            if len(pair_score) == 0:
                #Si todo el vocabulario son tokens, no hay mas particiones
                break

            pair_max = sorted(pair_score.items(), key=lambda x: x[1], reverse=True)[0][0]
            self.recal_splits(pair_max)
            self.vocab.append(self.new_token(pair_max[0], pair_max[1]))

        return self.vocab

