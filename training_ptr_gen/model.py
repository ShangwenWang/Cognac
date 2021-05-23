from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random
import numpy as np
use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


from .frequence import frequence
def constructAttention(stmts):
    np_stmts = stmts.cpu().numpy()
    type_attn = None
    for x in np_stmts:
        if type_attn is None:
            type_attn = np.vectorize(lambda t: frequence[t])(x)
        else:
            type_attn = np.vstack((type_attn, np.vectorize(lambda t: frequence[t])(x)))
    type_attn = torch.from_numpy(type_attn)

    # type_attn = F.softmax(type_attn, dim=1)  # Softmax or not??

    normalization_factor = type_attn.sum(1, keepdim=True)
    type_dist = type_attn / normalization_factor
    if stmts.device.type == 'cuda':
        type_dist = type_dist.cuda()
    return type_dist




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.stmt_embedding = nn.Embedding(config.stmt_size, config.stmt_emb_dim)
        init_wt_normal(self.embedding.weight)
        init_wt_normal(self.stmt_embedding.weight)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.stmt_lstm = nn.LSTM(config.stmt_emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        init_lstm_wt(self.stmt_lstm)
        if 0.0 < config.dropout_prob < 1.0:
            self.input_drop = nn.Dropout(p=config.dropout_prob)
        else:
            self.input_drop = None
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.stmt_W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
    # seq_lens should be in descending order

    def forward(self, input, stmts, seq_lens):
        embedded = self.embedding(input)  # batch_size * len * embedding len
        # embedded_stmts = self.stmt_embedding(stmts) # test plan 5 original
        embedded_stmts = constructAttention(stmts)  # test plan 5

        # embedded = torch.cat((embedded, embedded_stmts), dim=2)
        # seq_lens : batch_size * actual length
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        # stmt_packed = pack_padded_sequence(embedded_stmts, seq_lens, batch_first=True) # test plan 4 original
        output, hidden = self.lstm(packed)
        # stmt_output, _ = self.stmt_lstm(stmt_packed)# test plan 4 original
        # stmt_encoder_outputs, _ = pad_packed_sequence(stmt_output, batch_first=True) # test plan 4 original
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n (bi-directional lstm)
        encoder_outputs = encoder_outputs.contiguous()
        # stmt_encoder_outputs =stmt_encoder_outputs.contiguous() # test plan 4 original

        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim (3200*512)
        # stmt_feature = stmt_encoder_outputs.view(-1, 2*config.hidden_dim) # test plan 4 original
        encoder_feature = self.W_h(encoder_feature)  # a linear, outsize:3200*512
        if self.input_drop is not None:
            encoder_feature = self.input_drop(encoder_feature)
        # stmt_feature = self.stmt_W_h(stmt_feature) # test plan 4 original
        stmt_feature = embedded_stmts  # test plan 4 / test plan 5
        # stmt_feature = None  # test plan 4 / test plan 5 ori
        return encoder_outputs, encoder_feature, stmt_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim. h saves the last h_state of each layer
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))  # linear relu (512 -> 256)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        # self.stmt_v = nn.Linear(config.hidden_dim * 2, 1, bias=False) # test plan 3
        self.stmt_v = nn.Linear(config.stmt_emb_dim, 1, bias=False)  # test plan 4

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, stmt_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())  # batchsize, max_len, dim
        # s_t_hat: h_decoder + c_decoder
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim # test plan 1 (original)
        # att_features = encoder_feature + dec_fea_expanded + stmt_feature  # test plan 1
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k


        # scores = scores + self.stmt_v(torch.tanh(stmt_feature)).view(-1, t_k)  # test plan 2
        # stmt_scores = self.stmt_v(torch.tanh(stmt_feature)).view(-1, t_k)  # test plan 3
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask  # B x t_k
        # stmt_dist_ = F.softmax(stmt_scores, dim=1) *enc_padding_mask  # test plan 3
        # attn_dist_ = attn_dist_ + stmt_dist_  # test plan 3

        # stmt_dist_ = F.softmax(self.stmt_v(torch.tanh(stmt_feature.view(b*t_k, -1))).view(-1, t_k), dim=1)*enc_padding_mask   # test plan 4
        # normalization_factor_stmt = stmt_dist_.sum(1, keepdim=True)  # test plan 4
        # stmt_dist = stmt_dist_ / normalization_factor_stmt  # test plan 4


        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        # attn_dist += stmt_dist  # test plan 4
        stmt_feature = stmt_feature*enc_padding_mask  # test plan 5
        attn_dist += stmt_feature  # test plan 5
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Attention_ori(nn.Module):
    def __init__(self):
        super(Attention_ori, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        # self.stmt_v = nn.Linear(config.hidden_dim * 2, 1, bias=False) # test plan 3
        self.stmt_v = nn.Linear(config.stmt_emb_dim, 1, bias=False)  # test plan 4

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, stmt_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())  # batchsize, max_len, dim
        # s_t_hat: h_decoder + c_decoder
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim # test plan 1 (original)
        # att_features = encoder_feature + dec_fea_expanded + stmt_feature  # test plan 1
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k


        # scores = scores + self.stmt_v(torch.tanh(stmt_feature)).view(-1, t_k)  # test plan 2
        # stmt_scores = self.stmt_v(torch.tanh(stmt_feature)).view(-1, t_k)  # test plan 3
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask  # B x t_k
        # stmt_dist_ = F.softmax(stmt_scores, dim=1) *enc_padding_mask  # test plan 3
        # attn_dist_ = attn_dist_ + stmt_dist_  # test plan 3

        # stmt_dist_ = F.softmax(self.stmt_v(torch.tanh(stmt_feature.view(b*t_k, -1))).view(-1, t_k), dim=1)*enc_padding_mask   # test plan 4
        # normalization_factor_stmt = stmt_dist_.sum(1, keepdim=True)  # test plan 4
        # stmt_dist = stmt_dist_ / normalization_factor_stmt  # test plan 4


        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        # attn_dist += stmt_dist  # test plan 4

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.attention_network = Attention_ori()  # original
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        # self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim + config.stmt_emb_dim)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        # self.lstm = nn.LSTM(config.emb_dim + config.stmt_emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            # self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim + config.stmt_emb_dim, 1)
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        if 0.0 < config.dropout_prob < 1.0:
            self.output_drop = nn.Dropout(p=config.dropout_prob)
        else:
            self.output_drop = None

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, stmt_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        # ground truth,  reduced hidden state,encoder_outputs, encoder_feature, enc_padding_mask,
        # c state 1 (zeros)
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, stmt_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))  # what's x ??
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, stmt_feature,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim
        if self.output_drop is not None:
            output = self.output_drop(output)
        #output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
