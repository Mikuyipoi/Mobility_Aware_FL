import argparse
import torch
import numpy
def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'fashion',
        help = 'name of the dataset: mnist, cifar, fashion'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--iid',
        type=str,
        default='edge_niid',
        help='edge_niid, iid,niid'
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 64,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=100,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_epoch',
        type=int,
        default=5,
        help='number of local update (tau_l)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=2,
        help = 'number of edge aggregation (tau_e)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.001,
        help = 'learning rate of the SGD when trained on client'
    )

    parser.add_argument(
        '--num_clients',
        type = int,
        default = 40,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 4,
        help= 'number of edges'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default =numpy.random.randint(0,1000),
        help = 'random seed '
    )


    parser.add_argument(
        '--policy',
        type=str,
        default='policy',
        help='random,policy,max'
    )

    parser.add_argument(
        '--size',
        type=int,
        default=500,
        help='side length'
    )
    parser.add_argument(
        '--v',
        type=int,
        default=20,
        help='velocity of clients'
    )
    parser.add_argument(
        '--time',
        type=int,
        default=1,
        help='time of edge aggregation'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1e-27,
        help='energy consumption coefficient'
    )
    parser.add_argument(
        '--gamma',
        type=int,
        default=10000,
        help='CPU cycles computing a sample'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=3.98e-15,
        help='noise: -114dbm/mhz(3.98e-15w)'
    )
    parser.add_argument(
        '--power',
        type=float,
        default=0.2,
        help='max power: 0.2w'
    )
    parser.add_argument(
        '--f_min',
        type=float,
        default=1e8,
        help='min frequency: 10 mhz'
    )
    parser.add_argument(
        '--f_max',
        type=float,
        default=1e9,
        help='max frequency: 100 mhz'
    )
    parser.add_argument(
        '--bandwidth',
        type=float,
        default=5e4,
        help='bandwidth of RB: 1 mhz'
    )
    parser.add_argument(
        '--channel_num',
        type=int,
        default=4,
        help='channel number per edge'
    )
    parser.add_argument(
        '--lyp_v',
        type=float,
        default=100,
        help='lyp para'
    )
    args = parser.parse_args()
    args.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dataset=='cifar':
        args.upload_size=4506784
    if args.dataset=='fashion':
        args.upload_size=349440
    args.noise=args.noise*args.bandwidth/1e6
    return args
