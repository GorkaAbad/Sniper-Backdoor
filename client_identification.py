import argparse
import torch
from snn import *
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

parser = argparse.ArgumentParser(description='Client identification')
parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=20, help='batch size')
parser.add_argument('--alpha', type=float, default=0.02, help='alpha')
parser.add_argument('--n_clients', type=int, default=10,
                    help='number of clients')
parser.add_argument('--dir', type=str, default='results', help='directory')
parser.add_argument('--dir_shadow', type=str,
                    default='shadow', help='directory')
parser.add_argument('--dataname', type=str, default='mnist',
                    choices=['mnist', 'emnist', 'fmnist'])
parser.add_argument('--seed', type=int, default=42, help='seed')
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_clean = []
    for idx in range(args.n_clients):
        path = os.path.join(args.dir_shadow,
                            f'{args.dataname}_client_{idx}_results.pt')
        latent_space = torch.load(path)['latent_space']

        for idx, latent in enumerate(latent_space):
            latent_space[idx] = latent.cpu().detach().numpy()

        dataset_clean.append(latent_space)

    reshaped = np.array(dataset_clean).reshape((np.array(dataset_clean).shape[0] *
                                                np.array(
                                                    dataset_clean).shape[1],
                                                np.array(dataset_clean).shape[2]))

    # Generate dummy labels
    label_vector = []
    for i in range(0, np.array(dataset_clean).shape[0]):
        for j in range(0, np.array(dataset_clean).shape[1]):
            label_vector.append(i)

    dataset = np.concatenate(
        [np.array(label_vector).reshape(-1, 1), reshaped], axis=1)

    # np.save(os.path.join(args.dir, 'clients_latent_space'), final_set)

    emb_size = args.n_clients

    embedding_model = second_embedding_model(emb_size, dataset.shape[1] - 1)

    siamese_net = create_SNN_oneinput(embedding_model, dataset.shape[1] - 1)

    plot_model(
        siamese_net, show_shapes=True, expand_nested=True)

    optimiser_obj = Adam(learning_rate=args.lr)
    siamese_net.compile(loss=triplet_loss_adapted_from_tf,
                        optimizer=optimiser_obj, metrics=['acc'])

    dummy_train = np.zeros((len(dataset[:, 1:]), emb_size + 1))

    x_data = {'input_image': dataset[:, 1:],
              'input_label': dataset[:, :1].astype(np.uint8)}

    siamese_net.fit(x=x_data,
                    y=dummy_train,
                    # validation_data=(x_test, dummy_test),
                    batch_size=args.batch_size, epochs=args.epochs)

    print('[INFO]: Saving model')
    siamese_net.save(os.path.join(
        args.dir_shadow, f'{args.dataname}_fl_siamese.h5'))
    print('[INFO]: Done')

    if args.dataname == 'mnist':
        n_classes = 10
    elif args.dataname == 'emnist':
        n_classes = 26
    elif args.dataname == 'fmnist':
        n_classes = 10

    euclidean_net = two_input_model_composer(
        siamese_net.layers[2], 2304, 1, n_classes)
    plot_model(euclidean_net, show_shapes=True, expand_nested=True)

    dataset_clean = []
    for idx in range(args.n_clients):
        path = os.path.join(args.dir,
                            f'{args.dataname}_client_{idx}_results.pt')
        latent_space = torch.load(path)['latent_space']

        for idx, latent in enumerate(latent_space):
            latent_space[idx] = latent.cpu().detach().numpy()

        dataset_clean.append(latent_space)

    lis_idx = np.zeros((args.n_clients, args.epochs), dtype=int)

    # We begin with the anchors as t=0
    for t in range(args.epochs - 1):
        # Get the anchors at round t
        anchors = np.array(dataset_clean)[:, t, :]
        # Duplicate per number of clients
        anchors_repeated = np.repeat(anchors, args.n_clients, axis=0)

        # Comparators are the updates one epoch ahead
        comparators = np.array(dataset_clean)[:, t + 1, :]

        # Calculate the distances between t and t+1 per client
        i = 0
        last = args.n_clients
        for _ in range(len(anchors)):
            predictions = euclidean_net.predict({'input_anchor': anchors_repeated[i:last, :],
                                                 'input_comparator': comparators})

            # Get the index with the lowe distance
            min_idx = np.argmin(predictions)
            lis_idx[_][t] = int(min_idx)

            i = last
            last += args.n_clients

    for idx, client in enumerate(lis_idx):
        print(client)
        print(f'{idx} model belongs to: {np.argmax(np.bincount(client))}')


if __name__ == '__main__':
    main()
