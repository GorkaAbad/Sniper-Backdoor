import argparse
import os
import torch
from models import Discriminator, Generator
from utils import weights_init, train_gan, get_dataset_gan, get_noise, normalize, CustomDataset
from models import build_model
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np

parser = argparse.ArgumentParser(description='GAN')
parser.add_argument('--latent_size', type=int, default=64, help='latent size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--noise_dim', type=int, default=100, help='noise dim')
parser.add_argument('--gen_dim', type=int, default=64, help='gen dim')
parser.add_argument('--disc_dim', type=int, default=64, help='disc dim')
parser.add_argument('--n_channels', type=int, default=1, help='n channels')
parser.add_argument('--img_size', type=int, default=28, help='img size')
parser.add_argument('--n_epochs', type=int, default=200, help='n epochs')
parser.add_argument('--n_clients', type=int, default=10, help='n clients')
parser.add_argument('--lr', type=float, default=0.0002, help='lr')
parser.add_argument('--source_epoch', type=int, default=0,
                    help='source epoch to create the discriminator')
parser.add_argument('--dir', type=str, default='results',
                    help='source directory')
parser.add_argument('--dataname', type=str, default='mnist',
                    choices=['mnist', 'emnist', 'fmnist'], help='dataname')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--iid', action='store_true', default=False, help='iid')
parser.add_argument('--trainset_size', type=int,
                    default=1000, help='holdout dataset size')
parser.add_argument('--pretrained', action='store_true',
                    default=False, help='load a pretrained model')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    trainloader, n_classes = get_dataset_gan(
        args.dataname, args.batch_size, args.trainset_size)

    list_discrimiator = []
    for idx in range(args.n_clients):
        # Load the model at the desired epoch
        path = os.path.join(
            args.dir, f'{args.dataname}_client_{idx}_results.pt')
        resources = torch.load(path)
        model = resources['model_records'][args.source_epoch]

        # Create the Discriminator
        discriminator = Discriminator(args.disc_dim, args.n_channels)
        discriminator.apply(weights_init)

        model.pop('out.weight')
        model.pop('out.bias')

        model['conv4.weight'] = discriminator.state_dict()['conv4.weight']
        discriminator.load_state_dict(model)

        list_discrimiator.append(discriminator)

    # Get the last model
    path = os.path.join(
        args.dir, f'{args.dataname}_server_results.pt')
    resources = torch.load(path)
    w_model = resources['model']
    model = build_model(n_classes, args.pretrained)
    model.load_state_dict(w_model)
    model.to(device)

    for idx, D in enumerate(list_discrimiator):
        print(f'Training GAN for Client {idx}')
        G = Generator(args.noise_dim, args.gen_dim, args.n_channels)
        G.apply(weights_init)
        G = G.to(device)
        D = D.to(device)

        d_optim = torch.optim.Adam(D.parameters(), lr=args.lr)
        g_optim = torch.optim.Adam(G.parameters(), lr=args.lr)
        criterion = torch.nn.BCELoss()

        path = os.path.join(args.dir, f'gan_{args.dataname}')
        if not os.path.exists(path):
            os.makedirs(path)

        train_gan(G, D, criterion, d_optim, g_optim, trainloader,
                  args.n_epochs, args.batch_size, args.noise_dim, path, args.n_channels,
                  args.img_size, idx, device)

        # Generate the dataset of the clients
        z = get_noise(5000, args.noise_dim, device)
        z = Variable(z)

        fake_images = G(z)
        fake_images = normalize(fake_images)
        predictions = model(fake_images)

        predictions = torch.max(predictions, 1)[1]
        fake_dataset = CustomDataset(
            fake_images, predictions, n_classes=n_classes)

        fake_directory = os.path.join(
            args.dir, f'fake_datasets_{args.dataname}')
        if not os.path.exists(fake_directory):
            os.makedirs(fake_directory)
        fake_path = os.path.join(fake_directory, f'fake_dataset_{idx}.pt')
        torch.save(fake_dataset, fake_path)


if __name__ == '__main__':
    main()
