import torch.nn as nn
from transformers import BertModel


PRETRAINED_NAME_MODEL = "bert-base-uncased" #El modelo ha sido entrenado con frases en inglés, el uncase usa las palabras en lowercase. El case deja el texto en forma original.
PRETRAINED_NAME_MODEL = "google-bert/bert-base-multilingual-cased" #El modelo ha sido entrenado con frases en diferenetes lenguajes
PRETRAINED_NAME_MODEL = "dccuchile/bert-base-spanish-wwm-cased" #El modelo ha sido entrenado con frases en español

class Bert_FineTuned(nn.Module):
    def __init__(self, name_model, freeze_w, num_labels, dropout):
        super().__init__()

        self.bert = BertModel.from_pretrained(name_model)

        for _, param in self.bert.named_parameters():
            param.requires_grad = not freeze_w

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.bert.config.hidden_size,num_labels) # [768, 5]
   
    def forward(self, input_ids, attention_mask):
        out_bert = self.bert(input_ids, attention_mask) #[batch_size, max_len, embed_dim]
        #out_bert.pooler_output es la clasificación de [CLS], salida del mecanismo de atención.
        #Donde todos los tokens contienen información de todos, da igual que columna cojer.
        out_dropout = self.dropout(out_bert.pooler_output) #[batch_size, embed_dim]
        logits = self.output_layer(out_dropout) #[batch_size, num_labels]
        return logits

      

        









      