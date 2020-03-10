'''Train model function in PyTorch.

Training your deep learning model

Reference:
[1] No References
'''
from tqdm import tqdm


# Training
def train(net, device, trainloader, optimizer, criterion, epoch):
    global train_losses
    global train_acc
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    #total = 0
    processed = 0
    pbar = tqdm(trainloader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        # get samples
        inputs, targets = inputs.to(device), targets.to(device)

        # Init
        optimizer.zero_grad()

        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        outputs = net(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        processed += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)