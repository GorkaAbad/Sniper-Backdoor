from fileinput import hook_compressed
from typing import OrderedDict
from models import build_model
from torch import optim, nn
from tqdm import tqdm
import torch
import os
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG


class Participant:
    def __init__(self, testloader, dataname='mnist', n_classes=10):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(
            n_classes=n_classes, dataname=dataname).to(self.device)
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.data = dataname

        self.list_test_loss = []
        self.list_test_acc = []

    def evaluate(self):
        '''
        Evaluate the model on the test set

        Parameters:
            model (torch.nn.Module): model
            test_loader (torch.utils.data.DataLoader): test loader
            criterion (torch.nn.modules.loss._Loss): loss function
            device (torch.device): device

        Returns:
            test_loss (float): test loss
            test_acc (float): test accuracy
        '''
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for (data, target) in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output,
                                            torch.argmax(target, dim=1))
                _, pred = output.max(1)
                correct += pred.eq(torch.argmax(target,
                                                dim=1)).sum().item()

        test_loss /= len(self.testloader)
        test_acc = 100 * correct / len(self.testloader.dataset.data)

        self.list_test_loss.append(test_loss)
        self.list_test_acc.append(test_acc)

        return test_loss, test_acc


class Client(Participant):
    def __init__(self, trainloader, testloader, dataname='mnist', n_classes=10, lr=0.01, momentum=0.9, local_epochs=1):
        super().__init__(testloader, dataname=dataname, n_classes=n_classes)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum)
        self.scheduler = None
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.list_train_loss = []
        self.list_train_acc = []
        self.latent_space = []
        self.models_record = []

    def record_model(self):
        self.models_record.append(self.model.state_dict())

    def save_model(self, idx, args, path='results'):
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{args.dataname}_client_{idx}_results.pt')

        torch.save({'train_loss': self.list_train_loss,
                    'train_acc': self.list_train_acc,
                    'test_loss': self.list_test_loss,
                    'test_acc': self.list_test_acc,
                    'model_records': self.models_record,
                    'latent_space': self.latent_space,
                    'args': args},
                   path)

    def train(self):

        self.model.train()
        for local_epoch in range(self.local_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for (data, target) in tqdm(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, torch.argmax(target, dim=1))

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(torch.argmax(target,
                                        dim=1)).sum().item()

        train_loss = running_loss / len(self.trainloader)
        train_acc = 100 * correct / len(self.trainloader.dataset.data)

        self.list_train_loss.append(train_loss)
        self.list_train_acc.append(train_acc)

        return train_loss, train_acc


class Server(Participant):
    def __init__(self, clients, dataname='mnist', n_classes=10, testloader=None):
        super().__init__(testloader, dataname=dataname, n_classes=n_classes)
        self.clients = clients

    def fedavg(self):

        model_weights = [client.model.state_dict().values()
                         for client in self.clients]
        num_training_samples = [len(client.trainloader)
                                for client in self.clients]

        assert len(model_weights) == len(num_training_samples)
        new_weights = []
        total_training_samples = sum(num_training_samples)

        for layers in zip(*model_weights):
            weighted_layers = torch.stack(
                [torch.mul(l, w) for l, w in zip(layers, num_training_samples)])
            averaged_layers = torch.div(
                torch.sum(weighted_layers, dim=0), total_training_samples)
            new_weights.append(averaged_layers)

        self.model.load_state_dict(OrderedDict(zip(
            self.model.state_dict().keys(), new_weights)))

        for client in self.clients:
            client.model.load_state_dict(
                OrderedDict(zip(self.model.state_dict().keys(), new_weights)))

    def extract_latent_space(self):
        data, targets = next(iter(self.testloader))
        test_img = data[0][None].to(self.device)

        my_output = None

        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_.detach()

        for client in self.clients:
            client.model.eval()
            if type(client.model) == ResNet:
                hook = client.model.layer4.register_forward_hook(my_hook)
            elif type(client.model) == VGG:
                hook = client.model.features[-1].register_forward_hook(my_hook)
            else:
                hook = client.model.conv3.register_forward_hook(my_hook)
            client.model(test_img)
            client.latent_space.append(torch.flatten(my_output))
            hook.remove()
