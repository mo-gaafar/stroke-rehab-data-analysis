
import torch
import matplotlib.pyplot as plt

device = (
    "cuda" # "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

class SilverBullet(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        batch_size = shape[0]
        n_channels = shape[1]
        n_features = 5
        self.n_hidden = 4*2*n_features
        self.n_layers = 1
        is_bidirectional = False

        self.spatial = torch.nn.Conv2d(1,n_features, (shape[1], 7))
        self.temporal1 = torch.nn.Conv2d(n_features,n_features*2, (1, 7))
        self.temporal2 = torch.nn.Conv2d(n_features*2,n_features*4, (1, 7))
        # self.temporal3 = torch.nn.Conv2d(n_features*4,n_features*8, (1, 7))
        self.relu = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(2)
        self.temporal_pool = torch.nn.MaxPool2d((1,8))

        self.flat_regular = torch.nn.Flatten()

        # self.rnn = torch.nn.RNN(n_features, self.n_hidden, num_layers=self.n_layers, batch_first=True, dropout=0.1)
        # self.h0 = torch.randn((is_bidirectional*1+1)*self.n_layers, batch_size, self.n_hidden)

        self.lstm = torch.nn.LSTM(n_features*4, self.n_hidden, num_layers=self.n_layers, batch_first=True, dropout=0.25/self.n_layers)
        # self.h0 = 

        self.dropout = torch.nn.Dropout(0.7)

        self.mlp = torch.nn.Sequential(
            torch.nn.LazyLinear(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2),
        )
    
    def forward(self, X):
        # batch, chan, sample -> batch, 1, chan, sample
        X = self.spatial(X.unsqueeze(1))
        X = self.relu(X)
        X = self.temporal1(X)
        X = self.relu(X)
        X = self.temporal_pool(X)
        X = self.temporal2(X)
        X = self.relu(X)
        X = self.temporal_pool(X)
        # X = self.temporal3(X)
        # X = self.relu(X)
        # X = self.temporal_pool(X)
        # print(X.shape)
        X = X.permute(0,3,1,2)
        # print(X.shape)
        X = self.flat(X)
        # print(X.shape)
        # batch, sample, feat

        #  self.h0[:,:X.shape[0],:]
        # X = self.rnn(X)
        # X = X[1].permute((1,0,2)).squeeze()
        # h0 = torch.randn(self.n_layers, X.shape[0],self.n_hidden)
        # c0 = torch.randn(self.n_layers, X.shape[0],self.n_hidden)
        # initial_state = (h0, c0)
        initial_state = None
        # print(X.shape)
        X = self.lstm(X, initial_state)[0]#.permute((1,0,2)).squeeze()
        # print(X[0].shape)
        # X = X[0][:,]
        # # print(X.shape)
        # # X = 
        # print(X.shape)
        X = self.flat_regular(X)
        X = self.dropout(X)
        X = self.mlp(X)
        # [...,-1:].squeeze()
        # print(X)
        return X




def _train_loop(dataloader, model, loss_fun, optimizer):
    num_batches = len(dataloader)
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_accuracy = 0.0

    for i, (inputs, labels) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += torch.sum(labels == torch.argmax(outputs.detach(), dim=1))/labels.shape[0]
        # current = self.batch_size * i + len(inputs)
    return (running_loss/num_batches, running_accuracy.cpu()/num_batches)

    # pred = model(X_train[i*batch_size%X_train.shape[0]:(i*batch_size%X_train.shape[0]+batch_size)].to(device))
    # target = y_train[i*batch_size%X_train.shape[0]: (i*batch_size%X_train.shape[0]+batch_size)].detach().to(device)
    # # plt.plot(X_train[1][12])
    # loss = criterion(pred,target)
    # loss.backward()
    # optimizer.step()
    # print(f"Train loss: {loss.item()}")
    # model.eval()
    # pred = m(X_test[:min(X_test.shape[0],100)].to(device))
    # pred_classes = torch.argmax(torch.nn.functional.softmax(pred, dim=-1), dim=-1)
    # # print(((pred_classes==y_test[:min(y_test.shape[0],100)])/pred_classes.shape[0]).sum())
    # accuracies.append( ((pred_classes==y_test[:min(y_test.shape[0],100)])/pred_classes.shape[0]).sum().item())
    # # print(pred[:5])
    # loss = criterion(pred,y_test[:min(y_test.shape[0],100)].to(device))
    # losses.append(loss.item())
    # plt.plot(range(len(losses)),losses)
    # plt.plot(accuracies)
    # # plt.show()
    # i += 1

def _test_loop(dataloader, model, loss_fun):
    num_batches = len(dataloader)
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        running_loss += loss.item()
        running_accuracy += torch.sum(labels == torch.argmax(outputs.detach(), dim=1))/labels.shape[0]
    return (running_loss/num_batches, running_accuracy.cpu()/num_batches)

def train_sb(X_train, y_train, X_test, y_test):
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    # X_test = (X_test - X_test.mean(dim=-1).unsqueeze(-1)) / X_test.std(dim=-1).unsqueeze(-1)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    # X_train = (X_train - X_train.mean(dim=-1).unsqueeze(-1)) / X_train.std(dim=-1).unsqueeze(-1)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    m = SilverBullet(X_train.shape).to(device)
    gamma = 1/0.001
    optimizer = torch.optim.Adam(m.parameters(), lr=5e-4/gamma)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=gamma)

    n_epochs = 100

    trainloader = torch.utils.data.DataLoader(list(zip(X_train,y_train)), batch_size=5)
    testloader = torch.utils.data.DataLoader(list(zip(X_test,y_test)), batch_size=100, shuffle=True)

    train_loss, train_accuracy = _train_loop(trainloader, m, criterion, optimizer)

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        num_batches = len(trainloader)
        train_loss, train_accuracy = _train_loop(trainloader, m, criterion,optimizer)
        test_loss, test_accuracy = _test_loop(testloader, m, criterion)
        scheduler.step()

        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f'[epoch:{epoch + 1}/{n_epochs}, overall datum: {num_batches*(epoch+1):7d}]')
        print(f'train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}')
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}')

        # floating_avg = floating_avg * (window_len-1)/(window_len) + test_accuracy * 1/(window_len)
        # floating_avgs.append(floating_avg)
        # scheduler.step(floating_avg)
        # print(f"Difference from best: {floating_avg-best_score:.7f}")
        # last_learning_rate=scheduler.get_last_lr()[-1]
        # print(f"Last learning rate by scheduler: {last_learning_rate:.3e}")
        # if floating_avg > (best_score + min_increase_abs):
        # # if test_loss < best_score:
        # #     best_score = test_loss
        #     best_score = floating_avg
        #     epochs_since_best = 0
        # else:
        #     epochs_since_best += 1
        #     if epochs_since_best >= leeway_epochs:
        #         # stop, if not improoving for `leeway_epochs` epochs
        #         print(f"No improvement for {leeway_epochs} epochs. Stopping the training.")
        #         break

        # # Non-blocking non-focus-stealing incremental plot of loss and accuracy
        # # part 2
        # ax = plt.subplot(221)
        # ax.plot(train_losses)
        # ax = plt.subplot(222)
        # ax.plot(test_losses)
        # ax = plt.subplot(223)
        # ax.plot(train_accuracies)
        # ax = plt.subplot(224)
        # ax.plot(test_accuracies)
        # ax.plot(floating_avgs)
        # # plt.show(block=False)
        # # plt.pause(0.001)
        # fig.canvas.draw_idle()
        # fig.canvas.start_event_loop(0.001)

    # Final plot, this time blocking, to replace the previous non-blocking plot
    ax = plt.subplot(221)
    ax.plot(train_losses)
    ax = plt.subplot(222)
    ax.plot(test_losses)
    ax = plt.subplot(223)
    ax.plot(train_accuracies)
    ax = plt.subplot(224)
    ax.plot(test_accuracies)
    # Plot on the screen
    plt.show()