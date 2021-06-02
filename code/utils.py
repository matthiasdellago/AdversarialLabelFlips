import torch
import numpy as np
from livelossplot import PlotLosses

def compute_confusion_matrix(fmodel, attack, attack_kwargs, dataloader, device,
                             save_path=None):
        
    confusion_matrix = np.zeros((10,10), dtype=int)
    for images, labels in dataloader:
        # Perform Tensor device conversion
        images, labels = images.to(device), labels.to(device) 

        # Attack
        _, clipped_advs, success = attack(fmodel, images, labels, 
                                          **attack_kwargs)
        
        predicted_labels = fmodel(clipped_advs).argmax(dim=1).cpu().numpy()
        source_labels = labels.data.cpu().numpy()
        
        np.add.at(confusion_matrix, (source_labels, predicted_labels), 1)
    return confusion_matrix

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs,
                save_path=None, scheduler=None):
    liveloss = PlotLosses() # Live training plot generic API
    model = model.to(device) # Moves and/or casts the parameters and buffers to device.
    best_val_acc = 0
    
    for epoch in range(num_epochs): # Number of passes through the entire training & validation datasets
        logs = {}
        for phase in ['train', 'validation']: # First train, then validate
            # Switch between training and test eval mode depending on phase.
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0 # keep track of loss
            running_corrects = 0 # count of carrectly classified images
    
            for images, labels in dataloaders[phase]:
                images = images.to(device) # Perform Tensor device conversion
                labels = labels.to(device)
    
                outputs = model(images) # forward pass through network
                loss = criterion(outputs, labels) # Calculate loss
    
                if phase == 'train':
                    optimizer.zero_grad() # Set all previously calculated gradients to 0
                    loss.backward() # Calculate gradients
                    optimizer.step() # Step on the weights using those gradient w -=  gradient(w) * lr
    
                preds = torch.argmax(outputs, dim=1) # Get model's predictions
                running_loss += loss.detach() * images.size(0) # multiply mean loss by the number of elements
                running_corrects += torch.sum(preds == labels.data) # add number of correct predictions to total
    
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # get the "mean" loss for the epoch
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset) # Get proportion of correct predictions
            
            # Logging
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
    
            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
            
            if phase == 'validation' and epoch_acc>best_val_acc:
                best_val_acc = epoch_acc
                if save_path is not None:
                    print("New best validation accuracy")
                    accstr = str(np.round(epoch_acc.item(),4))
                    accstr += (6-len(accstr))*'0'
                    name = f"val_acc={accstr},epoch={epoch}.pt"
                    torch.save(model, save_path + name)
                    print("Model saved")
                    print()
        if scheduler is not None:
            scheduler.step()
        liveloss.update(logs) # Update logs
        liveloss.send() # draw, display stuff