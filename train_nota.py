import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as utils
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from sklearn.metrics import f1_score,accuracy_score


device = "cuda:1"
def prepare_data(pkl,batch_size,num):
    with open(pkl,"rb") as f:
        nota_distribution = pickle.load(f)
        other_distribution = pickle.load(f)
    train_x =torch.stack([i[:num] for i in nota_distribution+other_distribution])
    train_y = torch.cat([torch.zeros((len(nota_distribution),1)),torch.ones((len(other_distribution),1))]).to(device)
    my_dataset = utils.TensorDataset(train_x,train_y)
    my_dataloader = utils.DataLoader(my_dataset,shuffle=True,batch_size=batch_size)
    return my_dataloader

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim).to(device)
        self.linear = nn.Linear(input_dim, output_dim).to(device)
    def forward(self, x):
        normalized = self.norm(x)
        out = self.linear(normalized)
        return out

def test(args,test_loader):
    input_dim = args.num_ft
    output_dim = 1
    activation = args.activation
    model = LogisticRegressionModel(input_dim, output_dim)
    checkpoint = torch.load("{}_logreg_{}".format(args.data_path,input_dim))
    model.load_state_dict(checkpoint['state_dict'])
    r_total = r_correct= 0
    all_pred = np.empty(0)
    all_label = np.empty(0)
    all_score = np.empty(0)
    for logits, labels in test_loader:
        logits = Variable(logits.to(device))
        outputs = model(logits)
        if activation == "soft":
            predicted = torch.max(outputs, 1)[1].float().to("cpu").detach().numpy()
            labels = labels.squeeze(1).to("cpu").detach().numpy()
        elif activation == "sig":
            predicted = torch.sigmoid(outputs).round().squeeze(1).to("cpu").detach().numpy()
            score = torch.sigmoid(outputs).squeeze(1).to("cpu").detach().numpy()
            labels = labels.squeeze(1).to("cpu").detach().numpy()
        for i in range(len(labels)):
            if labels[i] == 1:
                r_total += 1
                if logits.argmax(dim=1)[i]==0 and predicted[i]==1:
                    r_correct += 1
        all_pred = np.append(all_pred,predicted)
        all_label = np.append(all_label,labels)
        all_score = np.append(all_score,score)
    f1_nota = f1_score(all_label,all_pred,pos_label=0)
    f1_gt = f1_score(all_label,all_pred,pos_label=1)
    acc = accuracy_score(all_label,all_pred)
    recall = r_correct/r_total
    with open("{}_logreg_{}_results.pkl".format(args.data_path,input_dim),"wb") as f:
        pickle.dump(all_score,f)
        pickle.dump(all_label,f)
    print("@{} R:{:.4f}, N:{:.4f}, N F1:{:.4f}, G F1: {:.4f}".format(input_dim,recall,acc,f1_nota,f1_gt))

def train(args,train_loader,test_loader=None):
    input_dim = args.num_ft
    activation = args.activation
    if activation == "soft":
        print("using softmax")
        output_dim = 2
        criterion = nn.CrossEntropyLoss()
    elif activation == "sig":
        print("using sigmoid")
        output_dim = 1
        criterion = nn.BCEWithLogitsLoss()
    model = LogisticRegressionModel(input_dim, output_dim)
    if args.optimizer == "sgd":
        print("using sgd optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        print("using Adam optimizer")
        optimizer = torch.optim.Adam(lr = args.lr,weight_decay=args.l2_norm,params=model.parameters())

    for epoch in range(args.num_epochs):
        cumu_loss = 0
        for logits, labels in train_loader:
            logits = Variable(logits[:,:input_dim].to(device),requires_grad=True)
            if activation == "soft":
                labels = Variable(labels.long()).squeeze(1)
            elif activation == "sig":
                labels = Variable(labels)
            
            optimizer.zero_grad()
            outputs = model(logits)
            loss = criterion(outputs, labels)
            cumu_loss += loss
            loss.backward()
            optimizer.step()

        average_loss = cumu_loss/len(train_loader)
        correct = 0
        total = 0
        if (epoch+1)%20 == 0:
            r_total = r_correct= 0
            all_pred = np.empty(0)
            all_label = np.empty(0)
            for logits, labels in test_loader:
                logits = Variable(logits.to(device))
                outputs = model(logits)
                if activation == "soft":
                    predicted = torch.max(outputs, 1)[1].float().to("cpu").detach().numpy()
                    labels = labels.squeeze(1).to("cpu").detach().numpy()
                elif activation == "sig":
                    predicted = torch.sigmoid(outputs).round().squeeze(1).to("cpu").detach().numpy()
                    labels = labels.squeeze(1).to("cpu").detach().numpy()
                for i in range(len(labels)):
                    if labels[i] == 1 and logits.argmax(dim=1)[i]==0 and predicted[i]==1:
                        r_correct += 1
                all_pred = np.append(all_pred,predicted)
                all_label = np.append(all_label,labels)
            f1_nota = f1_score(all_label,all_pred,pos_label=0)
            f1_gt = f1_score(all_label,all_pred,pos_label=1)
            acc = accuracy_score(all_label,all_pred)
            recall = r_correct/sum(all_label)
            print("@:{}".format(input_dim))
            print('R:{:.4f}'.format(recall))
            print('N:{:.4f}'.format(acc))
            print('NF1:{:.4f}'.format(f1_nota))
            print('GF1:{:.4f}'.format(f1_gt))
            print('average F1:{:.4f}'.format((f1_nota + f1_gt) / 2))
            torch.save({'state_dict': model.state_dict()},"{}_logreg_{}".format(args.data_path,input_dim))



def parse_arguments():
    parser = argparse.ArgumentParser(description='Logistic Regression that does binary classification')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_norm', type=float, default=0.00001)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--data_path', type=str, default='ubuntu_uncertain/logits_distri_')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--activation', type=str, default='sig',help='sig or soft')
    parser.add_argument('--optimizer', type=str, default='sgd',help='sgd or adam')
    parser.add_argument('--num_ft', type=int, default=10)
    parser.add_argument('--test_only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_loader = prepare_data(args.data_path+".pkl",args.batch_size,args.num_ft)
    test_loader = prepare_data(args.data_path+"test.pkl",args.batch_size,args.num_ft)
    if args.test_only:
        test(args,test_loader)
    else:
        train(args,train_loader,test_loader=test_loader)

if __name__ == "__main__":
    main()
