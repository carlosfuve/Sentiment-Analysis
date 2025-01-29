class EarlyStopping:
  def __init__(self, min_delta=0.001, patience=10, restore_best_weights=True):
    self.min_delta = min_delta
    self.patience = patience
    self.restore_best_weights = restore_best_weights
    self.epochs_without_improvement = 0
    self.best_score = None
    self.best_weights = None

  def __call__(self, val_loss, model):

    if self.best_score is None:
        self.best_score = val_loss
        self.best_weights = model.state_dict()
        return False
    
    #Es mejor monitorear el val_loss, porque el train_loss puede aprender patrones especificos
    #Si el val_loss de la epoch actual es menor, se actualizan los pesos 
    if val_loss < (self.best_score - self.min_delta):
      self.best_score = val_loss
      self.best_weights = model.state_dict()
      self.epochs_without_improvement = 0
    else:
      self.epochs_without_improvement += 1


    if self.epochs_without_improvement >= self.patience:
      #Se cargan los mejores pesos al modelo
      if self.restore_best_weights:
        model.load_state_dict(self.best_weights)
      return True

    return False