#!/bin/bash


# Please generate datasets first


# Local

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnet_512 -did 1 -algo Local -sfn resnet_512 > Local-Cifar10_2-resnet_512.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnet_512 -did 2 -algo Local -sfn resnet_512 > Local-Cifar10-0.1-resnet_512.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnet_512 -did 3 -algo Local -sfn resnet_512 > Local-Cifar100_10-resnet_512.out 2>&1 &

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnets -did 3 -algo Local -sfn resnets > Local-Cifar10_2-resnets.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnets -did 6 -algo Local -sfn resnets > Local-Cifar10-0.1-resnets.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnets -did 0 -algo Local -sfn resnets > Local-Cifar100_10-resnets.out 2>&1 &

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnet18_cnn -did 1 -algo Local -sfn resnet18_cnn > Local-Cifar10_2-resnet18_cnn.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnet18_cnn -did 6 -algo Local -sfn resnet18_cnn > Local-Cifar10-0.1-resnet18_cnn.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnet18_cnn -did 0 -algo Local -sfn resnet18_cnn > Local-Cifar100_10-resnet18_cnn.out 2>&1 &

# FedProto

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnet_512 -did 6 -algo FedProto -lam 10 -sfn resnet_512 > FedProto-Cifar10_2-resnet_512.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnet_512 -did 6 -algo FedProto -lam 10 -sfn resnet_512 > FedProto-Cifar10-0.1-resnet_512.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnet_512 -did 0 -algo FedProto -lam 10 -sfn resnet_512 > FedProto-Cifar100_10-resnet_512.out 2>&1 &

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnets -did 2 -algo FedProto -lam 10 -sfn resnets > FedProto-Cifar10_2-resnets.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnets -did 3 -algo FedProto -lam 10 -sfn resnets > FedProto-Cifar10-0.1-resnets.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnets -did 4 -algo FedProto -lam 10 -sfn resnets > FedProto-Cifar100_10-resnets.out 2>&1 &

# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10_2 -m resnet18_cnn -did 2 -algo FedProto -lam 10 -sfn resnet18_cnn > FedProto-Cifar10_2-resnet18_cnn.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 10 -data Cifar10-0.1 -m resnet18_cnn -did 5 -algo FedProto -lam 10 -sfn resnet18_cnn > FedProto-Cifar10-0.1-resnet18_cnn.out 2>&1 &
# nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100_10 -m resnet18_cnn -did 4 -algo FedProto -lam 10 -sfn resnet18_cnn > FedProto-Cifar100_10-resnet18_cnn.out 2>&1 &
