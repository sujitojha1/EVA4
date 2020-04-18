'''Train model function in PyTorch.

Training your deep learning model

Reference:
[1] No References
'''
from tqdm import tqdm

class train:

    def __init__(self):

        self.train_losses = []
        self.train_acc    = []

    # Training
    def execute(self,net, device, trainloader, optimizer, criterion,epoch,scheduler):

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
            print(inputs.shape)
            print(targets.shape)

            # Init
            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            scheduler.step()
            lr_val = scheduler.get_last_lr()[0]

            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Loss={loss.item():0.4f} Batch_id={batch_idx} LR={lr_val:0.4f} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            