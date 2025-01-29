import torch
import torch.nn.functional as F

def model_train(model, data, loss_fn, optimizer, n_examples, device):
  #Para avisar el modelo que debe ponerse en modo entrenamiento, el dropout funciona diferente en train que en eval
  model.train()
  running_loss_train = 0.0
  correct_pred = 0

  for d in data:
    inputs_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    
    #Pone los gradientes a 0 del modelo
    optimizer.zero_grad()

    #Logits
    output_model = model(inputs_ids, attention_mask)
    
    #Compara los logits y el target
    #Aplica una softmax y luego un logaritmo a la probabilidad de la clase target
    loss = loss_fn(output_model,targets)

    #Calculo del gradiente
    #PyTorch rastrea todos los elementos involucrados con el cálculo del loss, mediante el grafo dinámico
    #En los tensores con requires_grad=True almacena el calculo del loss en el parametro .grad de cada capa
    #x.grad += dloss/dx -> Derivada
    loss.backward()

    #Backpropagation
    #El optimizer ya contiene los tensores del modelo que necesitan actualizar su gradiente que ha almacenado el loss.backward
    #x += -lr * x.grad
    optimizer.step()

    running_loss_train += loss.item()

    preds = torch.argmax(output_model, dim = 1)
    #Se ejecuta en cpu porque necesito el resultado en la cpu para ejecutar codigo despues
    correct_pred += torch.sum(preds == targets).cpu() 

  #Como el running_loss ya se divide entre el número de batch, se divide entre el número de iteraciones que hace
  return running_loss_train/len(data), correct_pred/ n_examples


def model_eval(model, data, loss_fn, n_examples, device):
  model.eval()
  running_loss_val = 0.0
  correct_pred = 0
  lista_dif = []  
    
  with torch.no_grad():
    for d in data:
      inputs_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      output_model = model(inputs_ids, attention_mask)

      loss = loss_fn(output_model,targets)
      running_loss_val += loss.item()

      preds = torch.argmax(output_model, dim = 1)
      correct_pred += torch.sum(preds == targets).cpu()

        
      num_dif = torch.abs(targets - preds).tolist()
        
      output_model = F.softmax(output_model,dim=1)
      out_targ = output_model[torch.arange(targets.size(0)), targets]
      out_preds = output_model[torch.arange(preds.size(0)), preds]
      prob_dif = torch.abs(out_targ - out_preds).tolist()
        
      lista_dif += [val for val in zip(num_dif, prob_dif)]

  return running_loss_val / len(data), correct_pred/n_examples, lista_dif
