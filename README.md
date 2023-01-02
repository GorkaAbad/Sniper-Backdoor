# Sniper Backdoor
Code for the paper "Sniper Backdoor: Single Client Targeted Backdoor Attack in Federated Learning". SaTML'23.

## Training the network and extracting the latent space:

### IID 
```bash
python main.py --lr 0.001 --dataname cifar100 --n_clients 10 --iid --dir <path> --n_epochs 23 --pretrained
```

### Non-IID
```bash
python main.py --lr 0.001 --dataname cifar100 --n_clients 10 --iid --dir <path> --n_epochs 23 --pretrained
```

## Generating synthetic data:

### IID
```bash
pyhon synthetic_data.py --dataname cifar100 --dir <path> --n_clients 10 --iid
```

### Non-IID
```bash
pyhon synthetic_data.py --dataname cifar100 --dir <path> --n_clients 10
```

## Shadow training:

### IID
```bash
python shadow_network.py --dataname cifar100 --n_epochs 23 --pretrained --dir <path> --fake_dir <results path> --iid
```

### IID
```bash
python shadow_network.py --dataname cifar100 --n_epochs 23 --pretrained --dir <path> --fake_dir <results path>
```

## Clients identification:

```bash
python client_identification.py --epochs 100 --n_clients 10 --dir <path> --dir_shadow <shadow network path> --dataname cifar100
```

## Inject the Backdoor:
If we want to inject a dynamic backdoor we should use trojanzoo or other third party code. For static backdoors use:

```bash
python backdoor.py --dataname mnist --n_clients 5 --client_id <id> --source_label 1 --target_label 7
```
