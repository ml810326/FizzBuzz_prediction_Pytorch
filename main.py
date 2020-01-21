import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

BATCH = 6
DIGITS = 13
EPOCH = 300
DATASET_SIZE = 3000
CLASSES = ['FizzBuzz', 'Fizz', 'Buzz', '']

def encoder(num):
    return list(map(lambda x: int(x), ('{:0' + str(DIGITS) + 'b}').format(num))) 

def fizz_buzz(num):
    if num % 15 == 0:
        return 0 # 'FizzBuzz'
    elif num % 3 == 0:
        return 1 # 'Fizz'
    elif num % 5 == 0:
        return 2 # 'Buzz'
    else:
        return 3 # num

def make_data(num_of_data, batch_size):
    xs = []
    ys = []
    for _ in range(num_of_data):
        x = random.randint(0, 2**DIGITS-1)
        xs += [encoder(x)]
        ys += [fizz_buzz(x)]

    data = []
    for b in range(num_of_data//batch_size):
        xxs = xs[b*batch_size:(b+1)*batch_size]
        yys = ys[b*batch_size:(b+1)*batch_size]
        data += [(xxs, yys)]
        
    return data

class FizzBuzz(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FizzBuzz, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, 1024),
            nn.ReLU(), # Activation function
            nn.Linear(1024, 1024),
            nn.ReLU(), # Activation function
            nn.Linear(1024, out_channel)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def training(model, optimizer, training_data):
    model.train()
    for data, label in training_data:
        data = Variable(torch.FloatTensor(data))
        label = Variable(torch.LongTensor(label))
        optimizer.zero_grad() # Clear gradient
        out = model(data) # predict by model
        classification_loss = F.cross_entropy(out, label) # Cross entropy loss
        classification_loss.backward() # Calculate gradient
        optimizer.step() # Update model parameters

def testing(model, data):
    model.eval()
    loss = []
    correct = 0

    for x, y in data:
        x = Variable(torch.FloatTensor(x))
        y = Variable(torch.LongTensor(y))

        out = model(x)
        loss += [F.cross_entropy(out, y).data]
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()

    avg_loss = sum(loss) / len(loss)

    return {
        'accuracy': 100.0 * correct/(len(loss)*BATCH),
        'avg_loss': avg_loss
    }

def interactive_test(model):
    while True:
        num = input()
        if num == 'q':
            print('Bye~')
            return
        if int(num) >= 2**DIGITS:
            print('Please enter number smaller than {}'.format(2**DIGITS))
            continue

        ans = fizz_buzz(int(num))
        x = Variable(torch.FloatTensor([encoder(int(num))]))

        predict = model(x).data.max(1, keepdim=True)[1]
        print('Predict: {}, Real_Answer: {}'.format(CLASSES[predict[0][0]], CLASSES[ans]))

if __name__ == '__main__':
    m = FizzBuzz(DIGITS, 4)

    optimizer = optim.SGD(m.parameters(), lr=0.02, momentum=0.9)

    print('[INFO] Generating {} training datas'.format(DATASET_SIZE))
    training_data = make_data(DATASET_SIZE, BATCH)
    testing_data = make_data(100, BATCH)

    print('==== Start Training ====')
    for epoch in range(1, EPOCH + 1):
        training(m, optimizer, training_data)
        res = testing(m, testing_data)
        print('Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.2f}%'.format(
                epoch,
                EPOCH,
                res['avg_loss'],
                res['accuracy'],
            ), end='\r')

    print('\n==== Inertactive Test ====')
    print('[INFO] Enter a digit smaller than {}. ("q" to quit)'.format(2**DIGITS))
    interactive_test(m)
