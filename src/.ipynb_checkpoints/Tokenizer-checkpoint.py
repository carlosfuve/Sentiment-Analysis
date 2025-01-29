import re, string
import torch


class MyTokenizer():
    def __init__(self, vocab_path):
        self.vocab_dict = self.load_vocab(vocab_path=vocab_path)
        self.reverse_vocab = {v: k for k, v in self.vocab_dict.items()}

        self.signos_punt_keep = ',.?!"()+-:;%*€'
        self.signos_punt_rem = re.sub(f"[{re.escape(self.signos_punt_keep)}]",'',string.punctuation) + '•…\ufeff“”»—–’º°'

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r") as archivo:
            vocab = [word.strip() for word in archivo.readlines()]

        return dict(zip(vocab, range(len(vocab))))
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = []
        for token in tokens:
            input_ids.append(self.vocab_dict[token])
        return input_ids
    
    def convert_ids_to_tokens(self,input_ids):
        tokens = []
        for id in input_ids: #input_ids = torch.tensor([...])
            tokens.append(self.reverse_vocab[id.item()])
        return tokens
    
    def transform_txt(self, frase): 
        frase = frase.lower()
        frase = re.sub(f"[{re.escape(self.signos_punt_rem)}]",'',frase)
        frase_punt_space = re.sub(f"[{re.escape(self.signos_punt_keep)}]", r' \g<0> ', frase) 
        frase_no_double_space = re.sub(r'\s+', ' ', frase_punt_space) 
        return frase_no_double_space.strip().split(" ")
    
    def tokenize_word(self,word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab_dict: #Busca la subpalabra más larga del diccionario
                i -= 1
            if i == 0: #En el caso que no se encuentre devuelve el token UNK
                return ["[UNK]"]

            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0: 
                word = f"##{word}" #Esto añade ## para buscar los prefijos
        return tokens
    
    def tokenize(self, raw_txt):
        sep_txt = self.transform_txt(raw_txt)
        txt_encoded = []
        for word in sep_txt:
            txt_encoded += self.tokenize_word(word)
        return txt_encoded

    def encode_plus(self, raw_txt, add_special_tokens, max_length, return_attention_mask=True, return_tensors='pt',  
                padding='max_length', truncation=True, return_token_type_ids=False):    
        result = {}
        txt_encoded = self.tokenize(raw_txt)

        if add_special_tokens:
            txt_encoded = ['[CLS]'] + txt_encoded + ['[SEP]']
    
        if return_attention_mask:
            att_mask_list = [1]*len(txt_encoded) + [0]*(max_length-len(txt_encoded))
            att_mask_list = att_mask_list[:max_length]
            result['attention_mask'] = torch.tensor([att_mask_list])

        if len(txt_encoded) > max_length:
            txt_encoded = txt_encoded[:max_length] #TRUNCAR
        else:
            txt_encoded = txt_encoded + ['[PAD]']*(max_length-len(txt_encoded)) #PADDING

        result['input_ids'] = torch.tensor([self.convert_tokens_to_ids(txt_encoded)]) #TOKENS TO NUMBERS

        return result