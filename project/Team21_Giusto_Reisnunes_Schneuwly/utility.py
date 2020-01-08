'''
Image analysis and pattern recognition (EE-451)

@author: Christelle Schneuwly, Raphaël Reis, Gianni Giusto
'''
#---------------------------------- IMPORT -----------------------------------#
import torch
import numpy as np
#-----------------------------------------------------------------------------#


#---------------------------- ERRORS CLASSIFICATION --------------------------#

def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#-----------------------------------------------------------------------------#

#--------------------- ERRORS CLASSIFICATION + LOCALIZATION ------------------#

def compute_nb_errors_2(model, data_input, data_target):

    nb_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))

        output_classification = output[:, 0:2]
        
        _, predicted_classes = torch.max(output_classification.data, 1)
        
        for k in range(mini_batch_size):
            if train_target.data[b + k][0] != predicted_classes[k]: 
                nb_errors = nb_errors + 1

    return nb_errors

#-----------------------------------------------------------------------------#


#---------------------------- TRAIN CLASSIFICATION ---------------------------#

def train_model(model, train_input, train_target, crit, mini_batch_size, monitor_params):
    '''
    crit: criterion. MSELoss() or CrossEntropyLoss().
    Cross entropy usually preferred for classification tasks.
    train_target: labels equired to compute the error.
    '''
    criterion = crit
    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr = 1e-3)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    nb_epochs = 10
    
    loss_storage = []
    error_storage = []
    accuracy_storage = []
    
    
    for e in range(nb_epochs):
        sum_loss = 0
        sum_error = 0
        sum_acc = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            ### Compute class from the output ###
            _, predicted_classes = torch.max(output.data, 1)
            
            ### Compute loss ###
            loss = criterion(output, train_target.view(train_target.size(0)).narrow(0, b, mini_batch_size))
            
            ### Compute train error ###
            nb_errors = 0
            for k in range(mini_batch_size):
                if train_target.data[b + k] != predicted_classes[k]:
                    nb_errors = nb_errors + 1
            
            sum_loss += loss.item() # compute loss for each mini batch for 1 epoch
            sum_error += nb_errors
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Accuracy computed from the number of errors
        sum_acc = (train_input.size(0) - sum_error) / train_input.size(0)
        
        loss_storage.append(sum_loss)
        error_storage.append(sum_error)
        accuracy_storage.append(sum_acc)
                
        print('[epoch {:d}] loss: {:0.2f} error: {} accuracy: {:0.4f}'.format(e+1, sum_loss, sum_error, sum_acc))
        
    
    if monitor_params:
        return loss_storage, error_storage, accuracy_storage

#-----------------------------------------------------------------------------#


#---------------------- TRAIN CLASSIFICATION + LOCALIZATION ------------------#

def train_model_2(model, train_input, train_target, mini_batch_size, monitor_params):
    '''
    train_label: [1]
    train_target: [1, 0, minc, minr, w, h]
    '''
   
    criterion_classification = nn.CrossEntropyLoss()
    criterion_boxes = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    nb_epochs = 200
    
    loss_storage = []
    error_storage = []
    accuracy_storage = []
    
    
    for e in range(nb_epochs):
        sum_loss_classification = 0
        sum_error = 0
        #sum_acc = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            # output: [1, 0, minc, minr, w, h]
            output_classification = output[:, 0:2]
            output_boxes = output[:, 2:6]
            
            # true labels
            #train_target_classification = train_target[0] #first index is the class
            
            # CLASSIFICATION
            ### Compute class from the output ###
            _, predicted_classes = torch.max(output_classification.data, 1)
            
            ### Compute loss ###
            #loss = criterion(output, h_train_target.narrow(0, b, mini_batch_size)) # if using MSE
            loss_classification = criterion_classification(output_classification, 
                                            train_target[:, 0].view(train_target.size(0)).narrow(0, b, mini_batch_size))
            
            ### Compute train error ###
            nb_errors = 0
            for k in range(mini_batch_size):
                if train_target.data[b + k][0] != predicted_classes[k]: #or train_label[b+k]
                    nb_errors = nb_errors + 1
            
            sum_loss_classification += loss_classification.item() # compute loss for each mini batch for 1 epoch
            sum_error += nb_errors
            #sum_acc += acc # ok
            
            # BOXES
            loss_boxes = criterion_boxes(output_boxes, 
                                         train_target[:, 2:6].narrow(0, b, mini_batch_size).float())
            
            loss = 0.7*loss_classification + 0.3*loss_boxes
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Accuracy computed from the number of errors
        sum_acc = (train_input.size(0) - sum_error) / train_input.size(0)
        
        loss_storage.append(sum_loss_classification)
        error_storage.append(sum_error)
        accuracy_storage.append(sum_acc)
                
        print('[epoch {:d}] loss: {:0.2f} error: {} accuracy: {:0.4f}'.format(e+1, sum_loss_classification, sum_error, sum_acc))
        
    
    if monitor_params:
        return loss_storage, error_storage, accuracy_storage

#-----------------------------------------------------------------------------#


#------------------------------- MISCELLANEOUS -------------------------------#

def one_hot(labels, nb_labels):
    '''
    input labels can be either np array or tensors
    output is a torch.Tensor
    '''
    if (type(labels) == np.ndarray):
        h_labels = (np.arange(nb_labels) == labels[:, None]).astype(np.float32)

    elif (type(labels) == torch.Tensor):
        h_labels = labels.numpy()
        h_labels = (np.arange(nb_labels) == h_labels[:, None]).astype(np.float32)
        
    else:
        raise ValueError('The input type must be either numpy.ndarray or torch.Tensor')
    
    return torch.Tensor(h_labels) 

def train_test_split_imgs(x, y, ratio, seed=0):
    '''
    Split the data in train, test arrays
    Inputs:
        - x: features vector = (nb_imgs, nb_rows, nb_cols)
        - y: labels
        - ratio: percetange of the data used for testing (between [0, 1[)
    Outputs:
        - x_train, x_test, y_train, y_test
    '''
    np.random.seed(seed)
    nb_samples = x.shape[0]
    idx_split = int(np.floor(ratio * nb_samples))
    
    my_permutation = np.random.permutation(nb_samples)
    x = x[my_permutation, :, :]
    y = y[my_permutation]
    
    x_train, x_test = x[:idx_split, :, :], x[idx_split:, :, :]
    y_train, y_test = y[:idx_split], y[idx_split:]
    
    return x_train, x_test, y_train, y_test

#-----------------------------------------------------------------------------#
