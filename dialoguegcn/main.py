import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import seed_everything, get_IEMOCAP_loaders, train_or_eval_graph_model
from model import DialogueGCNModel

seed = 100

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=True, help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--nodal-attention', action='store_true', default=True,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # train
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    D_m = 100   # Dimensions of utterance embeddings
    D_g = 150   # Dimensions of graph
    D_p = 150
    D_e = 100   # Dimensions of lstm outputs
    D_h = 100
    D_a = 100
    graph_h = 100

    seed_everything()
    model = DialogueGCNModel(args.base_model,
                             D_m,
                             D_e,
                             graph_h,
                             n_speakers=2,
                             max_seq_len=110,
                             window_past=args.windowp,
                             window_future=args.windowf,
                             n_classes=n_classes,
                             listener_state=args.active_listener,
                             context_attention=args.attention,
                             dropout=args.dropout,
                             nodal_attention=args.nodal_attention,
                             no_cuda=args.no_cuda)
    loss_weights = torch.FloatTensor([1 / 0.086747,
                                      1 / 0.144406,
                                      1 / 0.227883,
                                      1 / 0.160585,
                                      1 / 0.127711,
                                      1 / 0.252668])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    print(20 * '=', "Preprocessing data", 20 * '=')
    print("\t* building iterators")
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)
    print(20 * '=', "Training data", 20 * '=')
    loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore = []
    for e in range(n_epochs):
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                             loss_function,
                                                                                             train_loader,
                                                                                             e,
                                                                                             cuda,
                                                                                             optimizer,
                                                                                             True)
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model,
                                                                                                           loss_function,
                                                                                                           test_loader,
                                                                                                           e,
                                                                                                           cuda)
        all_fscore.append(test_fscore)
    print('\t-> Finished, Best F1 Score:', max(all_fscore))
