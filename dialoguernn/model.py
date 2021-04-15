import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector

        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) * mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m + D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel  # Tensor(batch, D_p)

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)  # batch, 1, D_p
        q0_sel = self._select_parties(q0, qm_idx)  # (batch, D_p)

        g_ = self.g_cell(torch.cat([U, q0_sel], dim=1),
                         torch.zeros(U.size()[0], self.D_g).type(U.type()) if g_hist.size()[0] == 0 else
                         g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
        U_c_ = torch.cat([U, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)  # batch, party, D_m+D_g
        qs_ = self.p_cell(U_c_.contiguous().view(-1, self.D_m + self.D_g),
                          q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1). \
                expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            U_ss_ = torch.cat([U_, ss_], 1)
            ql_ = self.l_cell(U_ss_, q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)  # batch, party, 1
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0] == 0 \
            else e0
        e_ = self.e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)

        return g_, q_, e_, alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                                             listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        # initialize
        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                         self.D_p).type(U.type())  # batch, party, D_p
        e_ = torch.zeros(0).type(U.type())  # batch, D_e
        e = e_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, alpha  # seq_len, batch, D_e


class BiModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_a=100, dropout_rec=0.5,
                 dropout=0.5):
        super(BiModel, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.linear = nn.Linear(2 * D_e, 2 * D_h)
        self.smax_fc = nn.Linear(2 * D_h, n_classes)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)  # batch, seq_len, dim
        mask_sum = torch.sum(mask, dim=1).int()
        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, att2=True):
        """
        U(utterance embeddings) -> seq_len, batch, D_m
        qmask(speaker masks) -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)  # concatenate forward and backward
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                # att_em (batch, D_m), alpha_ (batch, 1, seq)
                att_emotions.append(att_em.unsqueeze(0))  # (1, batch, D_m)
                alpha.append(alpha_[:, 0, :])  # (batch, seq)
            att_emotions = torch.cat(att_emotions, dim=0)  # (seq, batch, D_m)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)  # seq_len, batch, n_classes
        if att2:
            return log_prob, alpha, alpha_f, alpha_b
        else:
            return log_prob, [], alpha_f, alpha_b

