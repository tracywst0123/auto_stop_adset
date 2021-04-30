from numpy import array, hstack
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cbook
from nd_process_data import return_train_data, return_eval_data
from sklearn.metrics import f1_score, precision_score, recall_score
from nd_models import MV_LSTM


def eval_pair(model, x, y_real):
    x_eval = torch.tensor(x, dtype=torch.float32)
    model.init_hidden(x_eval.size(0))
    y_result = list(cbook.flatten(model(x_eval).tolist()))
    y_compute = [1 if i > 0.85 else 0 for i in y_result]
    y_real = list(map(int, y_real))

    tp = [1 if (i == 1 and j == 1) else 0 for i, j in zip(y_compute, y_real)].count(1)
    fp = [1 if (i == 0 and j == 1) else 0 for i, j in zip(y_compute, y_real)].count(1)
    tn = [1 if (i == 1 and j == 0) else 0 for i, j in zip(y_compute, y_real)].count(1)

    if tp == 0:
        return 0, 0, 0
    else:
        precision = tp/(tp + fp)  # 关了的多少是该关的
        recall = tp/(tp + tn)     # 多少该关的关了
        f1 = 2 * (precision * recall)/(precision + recall)
        return precision, recall, f1


def train_evaluate(X, y, test_x, test_y, mv_net):
    mv_net.eval()
    train_pre, train_rec, train_f1 = eval_pair(mv_net, X, y)
    test_pre, test_rec, test_f1 = eval_pair(mv_net, test_x, test_y)
    return (train_pre, train_rec, train_f1), (test_pre, test_rec, test_f1)


def load_evaluate(path='./lstm_baseline.pth'):
    x, y = return_eval_data()

    n_features = 8
    n_timesteps = 5
    mv_net = MV_LSTM(n_features, n_timesteps)
    mv_net.load_state_dict(torch.load(path))
    mv_net.eval()

    precision, recall, f1 = eval_pair(mv_net, x, y)
    print('precision: ' + str(precision))
    print('recall:    ' + str(recall))
    print('f1:        ' + str(f1))


def run_train(model_save_path='./lstm_baseline.pth'):
    X, y, test_x, test_y = return_train_data()

    n_features = 8
    n_timesteps = 5

    # create NN
    mv_net = MV_LSTM(n_features, n_timesteps)

    print(mv_net)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-2)

    train_episodes = 50
    batch_size = 80

    losses = list()

    train_scores = list()
    test_scores = list()
    for t in range(train_episodes):
        for b in range(0, len(X), batch_size):
            mv_net.train()
            inpt = X[b:b + batch_size, :, :]
            target = y[b:b + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)

            optimizer.zero_grad()
            mv_net.init_hidden(x_batch.size(0))
            output = mv_net(x_batch)
            loss = criterion(output.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if b % (batch_size*20) == 0:
                print('epoches:  ' + str(t) + '  batches: ' + str(b/batch_size) + '  loss: ' + str(loss.item()))

        train_score, test_score = train_evaluate(X, y, test_x, test_y, mv_net)
        print('epoches:  ' + str(t) +
              '   train precision:  ' + str(train_score[0]) +
              '   train recall:  ' + str(train_score[1]) +
              '   train f1:  ' + str(train_score[2]) +
              '   test precision:  ' + str(test_score[0]) +
              '   test recall:  ' + str(test_score[1]) +
              '   test f1:  ' + str(test_score[2]))

        train_scores.append(train_score)
        test_scores.append(test_score)

    for param_tensor in mv_net.state_dict():
        print(param_tensor, "\t", mv_net.state_dict()[param_tensor].size())

    torch.save(mv_net.state_dict(), model_save_path)

    train_score_pd = pd.DataFrame(train_scores, columns=['precision', 'recall', 'f1'])
    test_score_pd = pd.DataFrame(test_scores, columns=['precision', 'recall', 'f1'])

    train_score_pd.to_csv('./train_score.csv')
    test_score_pd.to_csv('./test_score.csv')

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    load_evaluate()