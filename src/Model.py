import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


PRETRAINED_NAME_MODEL = "bert-base-uncased" #Entrenado con frases en inglés. El uncase usa las palabras en lowercase, el case deja el texto en forma original.
PRETRAINED_NAME_MODEL = "google-bert/bert-base-multilingual-cased" #Entrenado con frases en diferenetes lenguajes
PRETRAINED_NAME_MODEL = "dccuchile/bert-base-spanish-wwm-cased" #Entrenado con frases en español

class Bert_FineTuned(nn.Module):
    def __init__(self, name_model, freeze_w, num_labels, dropout):
        super().__init__()

        self.bert = BertModel.from_pretrained(name_model)
        #Decide si modificar o no los pesos por defecto del modelo preentrenado
        for name, param in self.bert.named_parameters():
            param.requires_grad = not freeze_w
            #El modelo en español carga los pesos del pooler de manera aleatoria por lo que hay que entrenarlas
            if "pooler" in name:  
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        
        n_384 = self.bert.config.hidden_size // 2
        n_192 = n_384 // 2
        n_96 = n_192 // 2
        
         # [768, 5]
        self.linear_384 = nn.Linear(self.bert.config.hidden_size, n_384)
        self.linear_192 = nn.Linear(n_384, n_192)
        self.linear_96 = nn.Linear(n_192, n_96)
        self.output_layer = nn.Linear(n_96,num_labels) # [768, 5]
   
    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids, attention_mask) #[batch_size, max_len, embed_dim]
        #out_bert.pooler_output es la clasificación de [CLS], salida del mecanismo de atención.
        #Donde todos los tokens contienen información de todos, da igual que columna cojer.
        out_dropout = self.dropout(out_bert.pooler_output) #[batch_size, embed_dim]
        logits = self.linear_384(out_dropout)
        logits = self.linear_192(logits)
        logits = self.linear_96(logits)
        logits = self.output_layer(logits) #[batch_size, num_labels]
        return logits

#--------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, att_heads):
        super().__init__()
        assert embed_dim % att_heads == 0

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.weights_out = nn.Linear(embed_dim, embed_dim)

        self.d_k = embed_dim // att_heads
        self.att_heads = att_heads
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        
        #[batch_size, max_len, embed_dim] -> [batch_size, att_heads, max_len, d_k]
        query = self.q_linear(q).view(batch_size, self.att_heads, -1, self.d_k) 
        key = self.k_linear(k).view(batch_size, self.att_heads, -1, self.d_k)
        value = self.v_linear(v).view(batch_size, self.att_heads, -1, self.d_k)

        result = self.scale_dot_product_att(query, key, value, mask) #[batch_size, att_heads, max_len, d_k]

        result = result.transpose(1,2).contiguous().view(batch_size, -1, self.d_k*self.att_heads) 
        output = self.weights_out(result)
        return output #[batch_size, max_len, embed_dim]
    
    def scale_dot_product_att(self, query, key, value, mask): 
        #matmult([batch_size, att_heads, max_len, d_k] * [batch_size, att_heads, d_k, max_len])
        #[batch_size, att_heads, max_len, max_len]
        scores = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k)) 
        if mask is not None:    
            mask = mask.unsqueeze(1).unsqueeze(2) #[batch_size,max_len] -> [batch_size, 1, 1, max_len]
            #Elimina los ceros para evitar divisiones por cero
            #Se cambia por un número pequeño para que la softmax le de un valor bajo
            scores = scores.masked_fill(mask==0,-1e9)
        attn = F.softmax(scores, dim=-1) #[batch_size, att_heads, max_len, max_len]
        #matmult([batch_size, att_heads, max_len, max_len] * [batch_size, att_heads, max_len, d_k])
        return torch.matmul(attn, value) 

class MyBertLayer(nn.Module): # "Encoder"
    def __init__(self, att_heads, embed_dim, d_ff, dropout):
        super().__init__()
        self.att = MultiHeadAttention(embed_dim, att_heads)
        self.ff = FeedForward(embed_dim, d_ff, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        out_att = self.att(x, x, x, mask) 
        add_norm_1 = self.norm1(x + self.drop1(out_att))
        ff_out = self.ff(add_norm_1)
        output = self.norm2(add_norm_1 + self.drop2(ff_out))
        return output    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, device):
        super().__init__()        
        self.pos_embed_matrix = torch.zeros(max_len, embed_dim, device=device)
        position = torch.arange(max_len).unsqueeze(1)
        #El -log nos asegura que decrementará a medida que avanza las posiciones, devuelve 1/10000^(-2i/embed_dim)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0))/embed_dim))
        #Aplica la operación a las posiciones pares
        self.pos_embed_matrix[:, 0::2] = torch.sin(position * div_term)
        #Aplica la operación a las posiciones impares
        self.pos_embed_matrix[:, 1::2] = torch.cos(position * div_term)

        #[1, max_len, embed_dim]
        self.pos_embed_matrix = self.pos_embed_matrix.unsqueeze(0)
        
    def forward(self, x):
        #Se aplica la suma a todos los elementos del batch
        return x + self.pos_embed_matrix[:, :x.size(1), :] 

class MyBert(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, att_heads, d_ff, dropout, num_labels, device, N=1):
        super().__init__()
        self.sqrt_embed_dim = torch.sqrt(torch.tensor(embed_dim))
        self.token_embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=0)
        self.pos_encod = PositionalEncoding(embed_dim, max_len, device)
        
        self.encoder_layers = nn.ModuleList([MyBertLayer(att_heads, embed_dim, d_ff, dropout) for _ in range(N)])
        self.layerNorm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        
        n_256 = embed_dim // 2
        n_128 = n_256 // 2
        n_64 = n_128 // 2
        
        self.linear_256 = nn.Linear(embed_dim, n_256)
        self.linear_128 = nn.Linear(n_256, n_128)
        self.linear_64 = nn.Linear(n_128, n_64)
        self.last_layer = nn.Linear(n_64, num_labels)
    
    def forward(self, tokens_ids, att_mask):
        #Obtener los Embeddings y añadrile el Positional Embedding
        tok_embed = self.token_embed(tokens_ids) * self.sqrt_embed_dim 
        output = self.pos_encod(tok_embed) #[batch_size, max_len, embed_dim]

        #Pasar por las capas del enconder
        for layer in self.encoder_layers:
           output = layer(output, att_mask)
        output = self.layerNorm(output)
        
        #output[:, 0, :] == bert pooler output, representación del [CLS]
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear_256(output)
        output = self.linear_128(output)
        output = self.linear_64(output)
        return self.last_layer(output)
      

        









      