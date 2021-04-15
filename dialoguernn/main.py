import numpy as np
import torch
import torch.optim as optim
import argparse
from utils import get_IEMOCAP_loaders, train_or_eval_model
from model import MaskedNLLLoss, BiModel
# f_out = open('out.log', 'w')
# sys.stdout = f_out
np.random.seed(1234)

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
parser.add_argument('--attention', default='general', help='Attention type')
args = parser.parse_args()
args.cuda = torch.cuda.is_available() and not args.no_cuda
args.cuda = False
if args.cuda:
    print('Running on GPU')
else:
    print('Running on CPU')
print(args)

# train
batch_size = args.batch_size
n_classes = 6
cuda = args.cuda
n_epochs = args.epochs
D_m = 100  # dimension of utterance embeddings
D_g = 500  # dimension of global states
D_p = 500  # dimension of party states
D_e = 300  # dimension of emotion states
D_h = 300  # dimension of linear hidden states
model = BiModel(D_m,
                D_g,
                D_p,
                D_e,
                D_h,
                n_classes=n_classes,
                listener_state=args.active_listener,
                context_attention=args.attention,
                dropout_rec=args.rec_dropout,
                dropout=args.dropout)
if cuda:
    model.cuda()
loss_weights = torch.FloatTensor([1 / 0.086747,
                                  1 / 0.144406,
                                  1 / 0.227883,
                                  1 / 0.160585,
                                  1 / 0.127711,
                                  1 / 0.252668])
if args.class_weight:
    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
else:
    loss_function = MaskedNLLLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.l2)
print(20 * '=', "Preprocessing data", 20 * '=')
print("\t* building iterators")
train_loader, valid_loader, test_loader = get_IEMOCAP_loaders('IEMOCAP_features_raw.pkl',
                                                              valid=0.0,
                                                              batch_size=batch_size,
                                                              num_workers=0)
print(20 * '=', "Training data", 20 * '=')
best_loss, best_label, best_pred, best_mask = None, None, None, None
for e in range(n_epochs):
    train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model,
                                                                          loss_function,
                                                                          train_loader,
                                                                          e,
                                                                          optimizer,
                                                                          True)
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model,
                                                                                                         loss_function,
                                                                                                         test_loader,
                                                                                                         e)
    if best_loss is None or best_loss > test_loss:
        best_loss, best_accuracy, best_fscore, best_label, best_pred, best_mask, best_attn = \
            test_loss, test_acc, test_fscore, test_label, test_pred, test_mask, attentions
print("\t-> best_loss: {:.4f}, bst_accuracy: {:.4f}, best_fscore: {:.4f}".format(best_loss, best_accuracy, best_fscore))
