import string, re
from collections import defaultdict


class MyVocabModel():
    def __init__(self, corpus):
        self.signos_punt_keep = ',.?!"()+-:;%*€'
        self.signos_punt_rem = re.sub(f"[{re.escape(self.signos_punt_keep)}]",'',string.punctuation) + '•…\ufeff“”»—–’º°'

        corpus = corpus.apply(self.transform_txt) #Transforma los textos a listas de textos separados por espacios

        self.words_freq = self.get_freq_words(corpus)
        self.vocab, self.splits = self.get_vocab_ini()
    
    def get_freq_words(self,corpus):
        word_freqs = defaultdict(int)
        for text in corpus:
            for word in text:
                word_freqs[word] += 1
        return word_freqs
    
    def get_freq_tokens(self): #NO SE USA
        tokendict = defaultdict(int)
        for token in self.vocab:
            for word, split in self.splits.items():
                for word_token in split:
                    if token == word_token:
                        tokendict[token] += self.words_freq[word]
        return tokendict
    
    def get_vocab_ini(self): #Vocabulario inicial con todos los caracteres del corpus
        alphabet = []
        for word in self.words_freq.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
  
            alphabet.sort()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
  
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.words_freq.keys()
        }

        return vocab, splits 

    def transform_txt(self,frase): 
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
                letter_freqs[split[0]] += freq 
                continue
        
            for i in range(len(split)-1):
                pair = (split[i], split[i + 1])
                #El acceder por split[i] ya diferencia entre a o ##a, ESTA BIEN EL LETTER_FREQ
                letter_freqs[split[i]] += freq

                #También esta bien, por la misma regla que antes 
                #Da el mismo resultado que buscarlo por fuerza bruta en todo el corpus
                pair_freqs[pair] += freq 
        
            letter_freqs[split[-1]] += freq
    
        scores = defaultdict(int)
        for pair, freq in pair_freqs.items():
            scores[pair] = freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
    
        return scores
    
    def recal_splits(self,pair_max):
        a,b = pair_max
        for pal in self.words_freq:
            split = self.splits[pal]
            if len(split) > 1:
                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i+1] == b:
                        split = split[:i] + [self.new_token(a,b)] + split[i+2:]
                    else:
                        i += 1
            self.splits[pal] = split


    def create_vocab(self, max_len_vocab: int):
        while len(self.vocab) < max_len_vocab:
            pair_score = self.calc_pair_score()
            if len(pair_score) == 0:
                #Si no todo el vocabulario son todo tokens, no hay mas particiones
                break
            pair_max = sorted(pair_score.items(), key=lambda x: x[1], reverse=True)[0][0]
            self.recal_splits(pair_max)
            self.vocab.append(self.new_token(pair_max[0], pair_max[1]))

        return self.vocab

