import torch
import numpy as np

import torch.nn as nn

from utils import batch_by_size
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from models import EmerGNN
from sklearn.metrics import roc_auc_score, average_precision_score
import loss as loss_func
import wandb


class BaseModel(object):
    def __init__(self, eval_ent, eval_rel, args, entity_vocab=None, relation_vocab=None):
        self.model = EmerGNN(eval_ent, eval_rel, args)
        if args.load_model:
            adversarial_suffix = '_adv' if args.adversarial else ''
            state_dict = torch.load(args.dataset + '_saved_model'+adversarial_suffix+'.pt')
            self.model.load_state_dict(state_dict)
        self.model.cuda()

        self.eval_ent = eval_ent
        self.eval_rel = eval_rel
        self.all_rel = args.all_rel
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max')

        self.bce_loss = nn.BCELoss()

        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab

        if args.adversarial:
            self.optimizer_ad = Adam(self.model.ad_net.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
            self.scheduler_ad = ReduceLROnPlateau(self.optimizer_ad, mode='max')
            self.model.random_layer.cuda()
            self.model.ad_net.cuda()

    def train(self, train_pos, train_neg, train_pos1, train_neg1, KG):
        if self.args.adversarial:
            pos_head, pos_tail, pos_label = torch.LongTensor(train_pos[:,0]).cuda(), torch.LongTensor(train_pos[:,1]).cuda(), torch.FloatTensor(train_pos[:,2:]).cuda()
            neg_head, neg_tail, neg_label = torch.LongTensor(train_neg[:,0]).cuda(), torch.LongTensor(train_neg[:,1]).cuda(), torch.FloatTensor(train_neg[:,2:]).cuda()
            pos_head1, pos_tail1, pos_label1 = torch.LongTensor(train_pos1[:,0]).cuda(), torch.LongTensor(train_pos1[:,1]).cuda(), torch.FloatTensor(train_pos1[:,2:]).cuda()
            neg_head1, neg_tail1, neg_label1 = torch.LongTensor(train_neg1[:,0]).cuda(), torch.LongTensor(train_neg1[:,1]).cuda(), torch.FloatTensor(train_neg1[:,2:]).cuda()

            n_train = max(len(pos_head), len(pos_head1))
            n_batch = self.args.n_batch
            repeat_factor = -(-n_train // min(len(pos_head), len(pos_head1)))
            pos_head, pos_tail, pos_label = pos_head.repeat(repeat_factor)[:n_train], pos_tail.repeat(repeat_factor)[:n_train], pos_label.repeat(repeat_factor, 1)[:n_train]
            neg_head, neg_tail, neg_label = neg_head.repeat(repeat_factor)[:n_train], neg_tail.repeat(repeat_factor)[:n_train], neg_label.repeat(repeat_factor, 1)[:n_train]
            pos_head1, pos_tail1, pos_label1 = pos_head1.repeat(repeat_factor)[:n_train], pos_tail1.repeat(repeat_factor)[:n_train], pos_label1.repeat(repeat_factor, 1)[:n_train]
            neg_head1, neg_tail1, neg_label1 = neg_head1.repeat(repeat_factor)[:n_train], neg_tail1.repeat(repeat_factor)[:n_train], neg_label1.repeat(repeat_factor, 1)[:n_train]
            perm = torch.randperm(n_train).cuda()
            pos_head1, pos_tail1, pos_label1 = pos_head1[perm], pos_tail1[perm], pos_label1[perm]
            neg_head1, neg_tail1, neg_label1 = neg_head1[perm], neg_tail1[perm], neg_label1[perm]

            self.model.train()
            for p_h, p_t, p_r, n_h, n_t, n_r, p_h1, p_t1, p_r1, n_h1, n_t1, n_r1 in tqdm(batch_by_size(n_batch, pos_head, pos_tail, pos_label, neg_head, neg_tail, neg_label, 
                    pos_head1, pos_tail1, pos_label1, neg_head1, neg_tail1, neg_label1, n_sample=n_train), 
                    ncols=100, leave=False, total=len(pos_head)//n_batch+int(len(pos_head)%n_batch>0)):
                self.model.zero_grad()
                self.optimizer.zero_grad()
                self.optimizer_ad.zero_grad()
                p_ht_embed = self.model.enc_ht(p_h, p_t, KG)
                n_ht_embed = self.model.enc_ht(n_h, n_t, KG)
                p_ht_embed_adv = self.model.enc_ht(p_h1, p_t1, KG)
                p_scores = torch.sigmoid(self.model.enc_r(p_ht_embed))
                n_scores = torch.sigmoid(self.model.enc_r(n_ht_embed))
                p_scores_adv = torch.sigmoid(self.model.enc_r(p_ht_embed_adv))
                p_scores_out = torch.flatten(torch.cat([1 - p_scores.unsqueeze(2), p_scores.unsqueeze(2)], dim=2).permute(1, 0, 2), start_dim=0, end_dim=1)
                p_scores_out_adv = torch.flatten(torch.cat([1 - p_scores_adv.unsqueeze(2), p_scores_adv.unsqueeze(2)], dim=2).permute(1, 0, 2), start_dim=0, end_dim=1)
                scores_comb = torch.cat([p_scores_out, p_scores_out_adv], dim=0)
                final_layer_comb = torch.cat([p_ht_embed.repeat(209, 1), p_ht_embed_adv.repeat(209, 1)], 0)

                p_r = p_r.float()
                p_scores = p_scores[p_r>0]
                n_scores = n_scores[n_r>0]
                scores = torch.cat([p_scores, n_scores], dim=0)
                labels = torch.cat([torch.ones(len(p_scores)), torch.zeros(len(n_scores))], dim=0).cuda()
                loss = self.bce_loss(scores, labels)
                ad_loss = loss_func.CDAN([final_layer_comb, scores_comb], self.model.ad_net, None, None, self.model.random_layer)*self.args.adversarial_weight
                print(f" | Classification Loss: {loss.item()} | Adversarial Loss: {ad_loss.item()} | Total loss: {loss.item() + ad_loss.item()}")
                wandb.log({"train_loss": loss.item()})
                wandb.log({"train_ad_loss": ad_loss.item()})
                loss += ad_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer_ad.step()
        else:
            pos_head, pos_tail, pos_label = torch.LongTensor(train_pos[:,0]).cuda(), torch.LongTensor(train_pos[:,1]).cuda(), torch.FloatTensor(train_pos[:,2:]).cuda()
            neg_head, neg_tail, neg_label = torch.LongTensor(train_neg[:,0]).cuda(), torch.LongTensor(train_neg[:,1]).cuda(), torch.FloatTensor(train_neg[:,2:]).cuda()
            n_train = len(pos_head)
            n_batch = self.args.n_batch

            self.model.train()
            for p_h, p_t, p_r, n_h, n_t, n_r in tqdm(batch_by_size(n_batch, pos_head, pos_tail, pos_label, neg_head, neg_tail, neg_label, n_sample=n_train), 
                    ncols=100, leave=False, total=len(pos_head)//n_batch+int(len(pos_head)%n_batch>0)):
                self.model.zero_grad()
                p_scores = torch.sigmoid(self.model.enc_r(self.model.enc_ht(p_h, p_t, KG)))
                n_scores = torch.sigmoid(self.model.enc_r(self.model.enc_ht(n_h, n_t, KG)))
                p_r = p_r.float()

                p_scores = p_scores[p_r>0]
                n_scores = n_scores[n_r>0]
                scores = torch.cat([p_scores, n_scores], dim=0)
                labels = torch.cat([torch.ones(len(p_scores)), torch.zeros(len(n_scores))], dim=0).cuda()
                loss = self.bce_loss(scores, labels) 
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_pos, test_neg, KG):
        pos_head, pos_tail, pos_label = test_pos[:,0], test_pos[:,1], test_pos[:,2:]
        neg_head, neg_tail, neg_label = test_neg[:,0], test_neg[:,1], test_neg[:,2:]
        batch_size = self.args.test_batch_size
        num_batch = len(pos_head) // batch_size + int(len(pos_head)%batch_size>0)

        self.model.eval()
        pos_scores = []
        neg_scores = []
        pred_class = {}
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, len(pos_head))
            p_h= pos_head[start:end]
            p_t= pos_tail[start:end]
            p_scores = self.model.enc_r(self.model.enc_ht(p_h, p_t, KG))
            p_scores = torch.sigmoid(p_scores)

            n_h= neg_head[start:end]
            n_t= neg_tail[start:end]
            n_scores = self.model.enc_r(self.model.enc_ht(n_h, n_t, KG))
            n_scores = torch.sigmoid(n_scores)
            pos_scores.append(p_scores.cpu().data.numpy())
            neg_scores.append(n_scores.cpu().data.numpy())

        labels = pos_label.cpu().data.numpy()
        pos_scores = np.concatenate(pos_scores)
        neg_scores = np.concatenate(neg_scores)
        for r in range(self.eval_rel):
            index = labels[:,r] > 0
            pred_class[r] = {'score': list(pos_scores[index,r]) + list(neg_scores[index,r]), 
                    'preds': list((pos_scores[index,r] > 0.5).astype('int')) + list((neg_scores[index,r]>0.5).astype('int')),
                    'label': [1] * np.sum(index) + [0] * np.sum(index)}

        roc_auc = []
        prc_auc = []
        ap = []
        for r in range(self.eval_rel):
            label = pred_class[r]['label']
            score = pred_class[r]['score']
            if len(label)==0:
                continue
            sort_label = np.array(sorted(zip(score, label), reverse=True))
            roc_auc.append(roc_auc_score(label, score))
            prc_auc.append(average_precision_score(label, score))
            k = int(len(label)//2)
            apk = np.sum(sort_label[:k,1])
            ap.append(apk/k)
        return np.mean(roc_auc), np.mean(prc_auc), np.mean(ap)

    def test_single(self, triplet, KG):
        heads = triplet[0].unsqueeze(0)
        tails = triplet[1].unsqueeze(0)
        ht_embed = self.model.enc_ht(heads, tails, KG)
        scores = self.model.enc_r(ht_embed)
        rela_scores = torch.sigmoid(scores).data.cpu().numpy()

        pred = (rela_scores > 0.5).astype('float')
        return pred[0]

    def visualize(self, triplet, KG, head_batch=True):
        h, t, r = triplet[0], triplet[1], triplet[2:]
        paths, weights = self.model.visualize_forward(h.unsqueeze(0), t.unsqueeze(0), r.unsqueeze(0), KG, 5, head_batch)
        outputs = []
        rel_weights = [0] * (self.all_rel - self.eval_rel)
        rel_freq = [0] * self.all_rel
        for path, weight in zip(paths, weights):
            out_str = '%4f\t' % weight
            for i in range(len(path)):
                h, t, r = path[i]
                h_name = self.entity_vocab[h]
                t_name = self.entity_vocab[t]
                if r == 2*self.all_rel - self.eval_rel:
                    r_name = 'idd'
                elif r < self.all_rel:
                    r_name = self.relation_vocab[r]
                    rel_freq[r] += 1
                    if r >= self.eval_rel:
                        rel_weights[r-self.eval_rel] += 1
                else:
                    r_id = r - self.all_rel + self.eval_rel
                    r_name = self.relation_vocab[r_id] + '_inv'
                    rel_freq[r_id] += 1
                    rel_weights[r-self.all_rel] += 1
               
                if i == 0:
                    out_str += '< %s, %6s, %18s' % (h_name, r_name, t_name)
                else:
                    out_str += ', %6s, %18s' % (r_name, t_name)
            out_str += ' >\n'
            outputs.append(out_str)
        return outputs, np.array(rel_weights), np.array(rel_freq)

    def save_model(self, out_str=''):
        adversarial_suffix = '_adv' if self.args.adversarial else ''
        torch.save(self.model.state_dict(), self.args.dataset+'_saved_model'+adversarial_suffix+'.pt')
        print(out_str, 'model saved')

    def load_model(self, model_name='_saved_model.pt'):
        self.model.load_state_dict(torch.load(self.args.dataset+model_name))
