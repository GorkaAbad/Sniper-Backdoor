import argparse
import torch
from participants import Client, Server
from utils import get_dataset, trainer, get_entire_dataset, backdoor_train, backdoor_evaluate
import copy
import os
import numpy as np
from models import build_model
from torch import optim, nn

parser = argparse.ArgumentParser(description='Client wise backdoor')
parser.add_argument('--n_clients', type=int, default=10,
                    help='number of clients')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--dataname', type=str, default='mnist',
                    choices=['mnist', 'emnist', 'fmnist', 'cifar10', 'cifar100'])
parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--n_local_epochs', type=int,
                    default=1, help='number of local epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--dir', type=str, default='results', help='directory')
parser.add_argument('--iid', action='store_true', default=False, help='iid')
parser.add_argument('--trainset_size', type=int,
                    default=1000, help='holdout dataset size')
parser.add_argument('--warm', action='store_true',
                    default=False, help='Warm-up for Non-IID')
args = parser.parse_args()


def main():

    test_clients = []
    test_server = []

    for _ in range(1):

        torch.manual_seed(_)
        np.random.seed(_)

        list_trainloader, list_testloader, n_classes = get_dataset(
            args.n_clients, args.dataname, args.iid, args.batch_size, args.trainset_size)

        clients = []
        for train, test in zip(list_trainloader, list_testloader):
            clients.append(Client(trainloader=train, testloader=test,
                                  lr=args.lr, momentum=args.momentum,
                                  dataname=args.dataname, n_classes=n_classes,
                                  local_epochs=args.n_local_epochs))

        server = Server(
            clients=clients, dataname=args.dataname, n_classes=n_classes,
            testloader=copy.deepcopy(list_testloader[0]))

        if args.warm:
            # If we are in the warm up model we train the model for few epochs in the 5% of the dataset
            trainloader, testloader, n_classes = get_entire_dataset(
                size=args.trainset_size, split=0.05)
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            model = build_model(
                n_classes=n_classes, dataname=args.dataname).to(device)
            optimizer = optim.SGD(
                model.parameters(), lr=0.01, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            for _ in range(15):
                backdoor_train(model, trainloader,
                               optimizer, criterion, device)
                _, acc = backdoor_evaluate(
                    model, testloader, criterion, device)
                print('Accuracy: ', acc)

            server.model = model
            for client in clients:
                client.model = model

        server.fedavg()

        for client in clients:
            test_clients.append(client.list_test_acc)
            client.scheduler = optim.lr_scheduler.StepLR(
                client.optimizer, step_size=(client.local_epochs*args.n_epochs) // 3, gamma=0.1)

        server_model = trainer(clients, server, args.n_epochs)

        test_server.append(server.list_test_acc)

        for idx, client in enumerate(clients):
            client.save_model(idx, args, args.dir)

        torch.save({'model': server_model.state_dict(),
                    'loss': server.list_test_loss,
                    'acc': server.list_test_acc},
                   os.path.join(args.dir, f'{args.dataname}_server_results.pt'))

        torch.save({'acc_clients': test_clients,
                    'acc_server': test_server}, os.path.join(args.dir, f'{args.dataname}_average_results.pt'))


if __name__ == '__main__':
    main()
