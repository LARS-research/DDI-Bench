import gc
import pickle
import torch
import random
import time
import logging
import math
import json
import datetime
import numpy as np
import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(current_path)
import argparse
import torch.distributed as dist
from transformers import AutoTokenizer
from datetime import date
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tools.drugbank_dataset_rl import drugbank_dataset_rl
from tools.twosides_dataset_rl import twosides_dataset_rl
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from sklearn import metrics
from models.roberta_cls import roberta_cls
from tools.drugbank_dataset_ppo import drugbank_dataset_ppo
# from evaluate import evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
from models.twosides_cls import  twosides_cls
# from apex import amp

from models.CDAN import *

MODEL_CLASSES = {
    'roberta_cls': roberta_cls,
    'twosides_cls': twosides_cls
}

DATASET_MODEL_CLASS = {
    'drugbank': drugbank_dataset_rl,
    'twosides': twosides_dataset_rl
}



def train(args,train_dataloader,model,optimizer,writer,logger=None,global_step=0,adversarial=0,adversarial_list=None):
    t0 = time.time()
    avg_loss, avg_acc = [],[]

    if adversarial:
        random_layer, ad_net, optimizer_ad,train_dataloader_adv, device, adversarial_weight  = adversarial_list

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    model.zero_grad()

    if adversarial:
        data_loader = zip(train_dataloader, train_dataloader_adv)
    else:
        data_loader = train_dataloader

    for step, batch_all in enumerate(data_loader):
        model.train()
        if adversarial:
            batch, adv_batch = tuple(t.to(args.device) for t in batch_all[0]), tuple(t.to(args.device) for t in batch_all[1])
        else:
            batch = tuple(t.to(args.device) for t in batch_all)

        if adversarial:
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output
                return hook

            model.llm.classifier.dense.register_forward_hook(get_activation('dense'))

        output = model(*(tuple(batch)))

        if adversarial:
            final_layer = activation['dense']
            softmax_output = nn.Softmax(dim=1)(torch.cat([output[1],output[2]]))
            # adv_data_batch = next(iter(train_dataloader_adv))
            # adv_batch = tuple(t.to(args.device) for t in adv_data_batch)
            output_adv = model(*(tuple(adv_batch)))
            softmax_output_adv = nn.Softmax(dim=1)(torch.cat([output_adv[1],output_adv[2]]))
            final_layer_adv = activation['dense']
            final_layer_comb = torch.cat((final_layer, final_layer_adv), 0)
            output_comb = torch.cat((softmax_output, softmax_output_adv), 0)
            ad_loss = CDAN([final_layer_comb, output_comb], ad_net, device, None, None, random_layer) * adversarial_weight
            loss = ad_loss + output[0]
        else:
            loss = output[0]

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            if adversarial:
                ad_loss = ad_loss / args.gradient_accumulation_steps

        loss.backward()
        loss = loss.item()
        avg_loss.append(loss)
        if adversarial:
            optimizer_ad.step()
            ad_net.zero_grad()

        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
            optimizer.step()
            model.zero_grad()
            global_step += 1
        
        if global_step % 10==0 and args.local_rank in [-1,0]:
                writer.add_scalar('loss', loss, global_step)
                # writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
        if (step+1) % args.log_step == 0 and args.local_rank in [-1,0]:
            elapsed = format_time(time.time() - t0)
            if adversarial:
                logger.info('Batch {:>5,} of {:>5,}.Loss: {:} | Adversarial Loss: {:} Elapsed:{:}.'
                .format(step+1, len(train_dataloader), format(loss, '.4f'), format(ad_loss, '.4f'), elapsed))
            else:
                logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Elapsed:{:}.'
                .format(step+1, len(train_dataloader), format(loss, '.4f'), elapsed))

        # break
        
    avg_loss = np.array(avg_loss).mean()
    return avg_loss,global_step


def evaluate(test_dataloader,model,args,logger):
    avg_acc = []
    model.eval()   

    y_pred = []
    y_true = []
    pos_scores_list = []
    neg_scores_list = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            # if len(batch)==5:
            pos_input_ids,pos_attention_mask,labels,neg_input_ids,neg_input_ids = batch
            #     neg_scores = model(neg_input_ids,neg_input_ids,labels)[1]
            #     neg_scores = torch.sigmoid(neg_scores)
            #     neg_scores_list.append(neg_scores)
            # else:
            #     pos_input_ids,pos_attention_mask,labels = batch

            _,pos_logits,neg_logits = model(*tuple(batch))
            
            pos_scores_list.append(pos_logits)
            neg_scores_list.append(neg_logits)
            labels_list.append(labels)
    pos_scores = np.concatenate([item.cpu() for item in pos_scores_list])
    labels = np.concatenate([item.cpu() for item in labels_list])
    if len(batch)==5: ### len(batch)==5 is true
        pred_class = {}
        neg_scores = np.concatenate([item.cpu() for item in neg_scores_list])
        for r in range(209):
            index = labels[:,r] > 0 
            pred_class[r] = {'score': list(pos_scores[index,r]) + list(neg_scores[index,r]), 
                    'preds': list((pos_scores[index,r] > 0.5).astype('int')) + list((neg_scores[index,r]>0.5).astype('int')),
                    'label': [1] * np.sum(index) + [0] * np.sum(index)}

        roc_auc = []
        prc_auc = []
        ap = []
        for r in range(209):
            label = pred_class[r]['label']
            score = pred_class[r]['score']
            sort_label = np.array(sorted(zip(score, label), reverse=True))
            if len(label) == 0:
                roc_auc.append(0)
                prc_auc.append(0)
                ap.append(0)
            else:
                roc_auc.append(roc_auc_score(label, score))
                prc_auc.append(average_precision_score(label, score))
                k = int(len(label)//2)
                apk = np.sum(sort_label[:k,1])
                ap.append(apk/k)

    pred_class_pos = {}
    for r in range(209):
        pred_class_pos[r] = {'score': list(pos_scores[:,r]), 
                # 'preds': list((pos_scores[r] > 0.5).astype('int')),
                'label': labels[:,r]}

    roc_auc_pos = []
    prc_auc_pos = []
    ap_pos = []
    for r in range(209):
        label = pred_class_pos[r]['label']
        score = pred_class_pos[r]['score']
        sort_label = np.array(sorted(zip(score, label), reverse=True))
        roc_auc_pos.append(roc_auc_score(label, score))
        prc_auc_pos.append(average_precision_score(label, score))

        def calculate_ap_at_k(y_true, y_scores, k):
            
            sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
            
            
            sorted_indices = sorted_indices[:k]
            
            
            num_relevant = 0
            precision_sum = 0.0
            
            for i, index in enumerate(sorted_indices):
                if y_true[index] == 1:
                    num_relevant += 1
                    precision = num_relevant / (i + 1)  
                    precision_sum += precision
            
            
            ap = precision_sum / num_relevant if num_relevant > 0 else 0.0
            
            return ap
        k = int(len(label)//2)
        ap_pos.append(calculate_ap_at_k(label,score,k))
    
    if len(batch)==5:
        return np.mean(roc_auc), np.mean(prc_auc), np.mean(ap),np.mean(roc_auc_pos), np.mean(prc_auc_pos), np.mean(ap_pos)
    else:
        return np.mean(roc_auc_pos), np.mean(prc_auc_pos), np.mean(ap_pos),0,0,0

            # correct = torch.eq(torch.max(scores, dim=1)[1], labels.flatten()).float()         
            # acc = correct.sum().item() / len(correct)

            # y_pred.extend(torch.max(scores, dim=1)[1].tolist())
            # y_true.extend(labels.flatten().tolist())
            # avg_acc.append(acc)
            # # break
        
    # gather_objects = [None for _ in range(torch.distributed.get_world_size())]
    # gather_objects[args.local_rank] = [y_pred,y_true]
    # Note: Process group initialization omitted on each rank.

    acc = metrics.accuracy_score(y_true,y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    # print('len = {} acc = {} f1 = {}, kappa = {} local_rank = {}'.format(len(y_true),acc,f1,kappa,torch.distributed.get_rank()))
    # output = [None for _ in gather_objects]
    # dist.gather_object(
    #     gather_objects[dist.get_rank()],
    #     output if dist.get_rank() == 0 else None,
    #     dst=0
    # )
    # dist.barrier()
    if args.local_rank in [-1,0]:
        # Assumes world_size of 3.
        # y_true = []
        # y_pred = []
        # for data in output:
        #     y_pred.extend(data[0])
        #     print(len(data[0]))
        #     y_true.extend(data[1])
        acc = metrics.accuracy_score(y_true,y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        logger.info("acc = {:>5,}".format(np.array(avg_acc).mean(),'.4f'))
        logger.info("f1 = {}".format(f1))
        logger.info("kappa = {}".format(kappa))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(len(test_dataloader.dataset)))
    return f1

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file',help='config file path')
    config_file = 'configs/main_twosides.yaml'# parser.parse_args().config_file
    args = Args(config_file)
    ### set hyperparameters
    parser = argparse.ArgumentParser(description='TextDDI')
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--adversarial_task', default='S1', type=str, help='which set to use')
    parser.add_argument('--adversarial_weight', default=1, type=float, help='the weight of adversarial loss')
    parser.add_argument('--gamma_split', type=str, default="55", choices=["55","60","65","70"])
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--dataset_type', type=str, default='finger', help='type of dataset')
    parser.add_argument('--gpu', type=str, default="1", help='gpu id')
    op_args = parser.parse_args()
    adversarial = op_args.adversarial
    adversarial_task = op_args.adversarial_task
    adversarial_weight = op_args.adversarial_weight
    gamma_split = op_args.gamma_split
    if op_args.dataset_type == 'random':
        dtp = op_args.dataset_type
    else:
        dtp = op_args.dataset_type + '_' + str(gamma_split)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format("twosides_" + start_date + '_' + dtp,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    else: ### usually not eval
        log_path = './log/{}/{}.log'.format("twosides_" + start_date + '_' + dtp,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    # if args.eval:
    #     log_path = './log/{}/{}-eval.log'.format(start_date,args.annotation)
    # else:
    #     log_path = './log/{}/{}.log'.format(start_date,args.annotation)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache()
    seed_everything(op_args.seed)
    if args.multi_gpu and args.use_gpu:
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else :
        args.local_rank = -1
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cuda:"+ op_args.gpu if torch.cuda.is_available() else "cpu"
    if args.use_gpu == False:
        device = torch.device('cpu')
    args.device = device

    logger = None
    if args.local_rank in [-1,0]:
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=log_path,
            filemode=args.filemode)
        logger = logging.getLogger()
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank,device, bool(args.local_rank != -1)))
        logger.info("Training/evaluation parameters %s", args.to_str())
    
    if args.local_rank in [-1,0]:
        tensorboard_path = './tensorboard/{}/{}'.format(start_date,args.annotation)
        if not os.path.exists(os.path.dirname(tensorboard_path)):
            os.makedirs(os.path.dirname(tensorboard_path))
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None

    if args.eval: ### not eval
        model = MODEL_CLASSES[args.model_type](args)
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True)
        model.to(args.device)
        dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test') # SequentialSampler
        dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=0)
        roc_auc, prc_auc, ap, roc_auc_pos, prc_auc_pos, ap_pos = evaluate(dev_dataloader,model,args,logger)
        # writer.add_scalar('dev_acc', dev_acc, epoch)
        logger.info("roc_auc={:.4f},prc_auc={:.4f},ap={:.4f} roc_auc_pos={:.4f},prc_auc_pos={:.4f},ap_pos={:.4f}".format(roc_auc,prc_auc,ap,roc_auc_pos,prc_auc_pos,ap_pos))

        # logger.info("dev_acc{}".format(test_acc))
        return
    start_epoch = 0

    model = MODEL_CLASSES[args.model_type](args)
    if args.resume:
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True)
        model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
    ]

    if adversarial: # drugbank: 86
        random_layer = RandomLayer([768, 209], 500).to(args.device)
        random_layer.device(args.device)
        ad_net = AdversarialNetwork(500, 500).to(args.device)
        optimizer_ad = Adam(ad_net.parameters(), eps = args.epsilon,betas=(0.9,0.98), lr=args.lr)
    else:
        random_layer = None
        ad_net = None
        optimizer_ad = None

    start_epoch = 0
    optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    model.to(args.device)

    global_step = 0
    best_performance = 0
    best_checkpoint_path = None
    # train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train'),'rb'))
    patience = args.patience
    fail_time = 0

    logger.info("choose split as: {}".format(dtp))

    fail_time = {}
    fail_time['S0'], fail_time['S1'], fail_time['S2'] = 0, 0, 0
    best_performance = {}
    best_performance['S0'], best_performance['S1'], best_performance['S2'] = 0, 0, 0
    best_checkpoint_path = {}
    for epoch in range(int(args.num_train_epochs)):

        if fail_time['S0'] >= patience and fail_time['S1'] >= patience and fail_time['S2'] >= patience:
            break
        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = DATASET_MODEL_CLASS[args.dataset](args,'train',dtp)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=0)
        if adversarial:
            adv_content = len(train_dataset)
            train1_dataset_adv = DATASET_MODEL_CLASS[args.dataset](args,'valid_' + adversarial_task, dtp, adv_content)
            train2_dataset_adv = DATASET_MODEL_CLASS[args.dataset](args,'test_' + adversarial_task, dtp, adv_content)
            train_dataset_adv = ConcatDataset([train1_dataset_adv, train2_dataset_adv])
            train_sampler_adv = RandomSampler(train_dataset_adv) 
            train_dataloader_adv = DataLoader(train_dataset_adv, sampler=train_sampler_adv, batch_size=args.per_gpu_train_batch_size,num_workers=4)
        else:
            train_dataloader_adv = None
        torch.cuda.empty_cache()
        train_loss,global_step = train(args, train_dataloader,model,optimizer,writer,logger,global_step,adversarial,[random_layer,ad_net,optimizer_ad,train_dataloader_adv,args.device,adversarial_weight])
        if args.local_rank in [-1,0]:
            logger.info('epoch={},loss={}'.format(epoch, train_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if args.local_rank in [-1,0]:
            for setting in ['S0', 'S1', 'S2']:
                dev_data = DATASET_MODEL_CLASS[args.dataset](args,'valid_' + setting,dtp)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
                roc_auc, prc_auc, ap, roc_auc_pos, prc_auc_pos, ap_pos = evaluate(dev_dataloader,model,args,logger)
                # writer.add_scalar('dev_acc', dev_acc, epoch)
                logger.info("epoch={}, setting {},roc_auc={:.4f},prc_auc={:.4f},ap={:.4f} roc_auc_pos={:.4f},prc_auc_pos={:.4f},ap_pos={:.4f}".format(epoch,setting,roc_auc,prc_auc,ap,roc_auc_pos,prc_auc_pos,ap_pos))
                if roc_auc > best_performance[setting]: #epoch % 10 == 0: # :
                    checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
                    if not os.path.exists(checkpoints_dir):
                        os.makedirs(checkpoints_dir)
                    checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}_{}.pt'.format(epoch, setting))
                    best_checkpoint_path[setting] = checkpoint_path
                    best_performance[setting] = roc_auc
                    torch.save({'model':model.state_dict(),'epoch':epoch},checkpoint_path)
                    logger.info('Save best checkpoint to {}'.format(checkpoint_path))
                    fail_time[setting] = 0
                else:
                    fail_time[setting]+=1
    if args.test:
        if args.local_rank in [-1,0]:
            for setting in ['S0', 'S1', 'S2']:
                logger.info("test best_checkpoint_path={}".format(best_checkpoint_path[setting]))
                checkpoint = torch.load(best_checkpoint_path[setting])
                model.load_state_dict(checkpoint['model'])
                dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test_' + setting,dtp)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
                roc_auc, prc_auc, ap, roc_auc_pos, prc_auc_pos, ap_pos = evaluate(dev_dataloader,model,args,logger)
                # writer.add_scalar('dev_acc', dev_acc, epoch)
                logger.info("test best_checkpoint_path={}, setting {}, roc_auc={:.4f},prc_auc={:.4f},ap={:.4f} roc_auc_pos={:.4f},prc_auc_pos={:.4f},ap_pos={:.4f}".format(best_checkpoint_path[setting], setting,roc_auc,prc_auc,ap,roc_auc_pos,prc_auc_pos,ap_pos))
        
    #     if args.local_rank in [-1,0]:
    #         dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
    #         dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
    #         dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
    #         roc_auc, prc_auc, ap, roc_auc_pos, prc_auc_pos, ap_pos = evaluate(dev_dataloader,model,args,logger)
    #         # writer.add_scalar('dev_acc', dev_acc, epoch)
    #         logger.info("epoch={},roc_auc={:.4f},prc_auc={:.4f},ap={:.4f} roc_auc_pos={:.4f},prc_auc_pos={:.4f},ap_pos={:.4f}".format(epoch,roc_auc,prc_auc,ap,roc_auc_pos,prc_auc_pos,ap_pos))
    #         if roc_auc > best_performance: #epoch % 10 == 0: # :
    #             checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
    #             if not os.path.exists(checkpoints_dir):
    #                 os.makedirs(checkpoints_dir)
    #             checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}.pt'.format(epoch))
    #             best_checkpoint_path = checkpoint_path
    #             best_performance = roc_auc
    #             torch.save({'model':model.state_dict(),'epoch':epoch},checkpoint_path)
    #             logger.info('Save best checkpoint to {}'.format(checkpoint_path))
    #             fail_time = 0
    #         else:
    #             fail_time+=1
    # if args.test:
    #     if args.local_rank in [-1,0]:
    #         logger.info("test best_checkpoint_path={}".format(best_checkpoint_path))
    #         checkpoint = torch.load(best_checkpoint_path)
    #         model.load_state_dict(checkpoint['model'])
    #         dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
    #         dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
    #         dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
    #         roc_auc, prc_auc, ap, roc_auc_pos, prc_auc_pos, ap_pos = evaluate(dev_dataloader,model,args,logger)
    #         # writer.add_scalar('dev_acc', dev_acc, epoch)
    #         logger.info("best_checkpoint_path={},roc_auc={:.4f},prc_auc={:.4f},ap={:.4f} roc_auc_pos={:.4f},prc_auc_pos={:.4f},ap_pos={:.4f}".format(best_checkpoint_path,roc_auc,prc_auc,ap,roc_auc_pos,prc_auc_pos,ap_pos))

if __name__=='__main__':
    main()

