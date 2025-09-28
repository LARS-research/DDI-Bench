import gc
import sys
import os
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(current_path)
import pickle
import torch
import random
import time
import logging
import math
import json
import datetime
import numpy as np

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
# from apex import amp

from models.CDAN import *

MODEL_CLASSES = {
    'roberta_cls': roberta_cls,
}

DATASET_MODEL_CLASS = {
    'drugbank': drugbank_dataset_rl,
    'twosides': twosides_dataset_rl
}

def train(args,train_dataloader,model,optimizer,writer,logger=None,global_step=0,adversarial=0,adversarial_list=None):
    
    avg_loss, avg_acc = [],[]

    if adversarial:
        random_layer, ad_net, optimizer_ad,train_dataloader_adv, device, adversarial_weight  = adversarial_list

    model.zero_grad()
    t0 = time.time()

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

        output = model(*(tuple(batch)[0:3]))

        if adversarial:
            final_layer = activation['dense']
            softmax_output = nn.Softmax(dim=1)(output[1])
            # adv_data_batch = next(iter(train_dataloader_adv))
            # adv_batch = tuple(t.to(args.device) for t in adv_data_batch)
            output_adv = model(*(tuple(adv_batch)[0:3]))
            softmax_output_adv = nn.Softmax(dim=1)(output_adv[1])
            final_layer_adv = activation['dense']
            final_layer_comb = torch.cat((final_layer, final_layer_adv), 0)
            output_comb = torch.cat((softmax_output, softmax_output_adv), 0)
            ad_loss = CDAN([final_layer_comb, output_comb], ad_net, device, None, None, random_layer) * adversarial_weight
            loss = ad_loss + output[0]
        else:
            loss = output[0]

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            ad_loss = ad_loss / args.gradient_accumulation_steps

        loss.backward()
        loss = loss.item()
        avg_loss.append(loss)
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if adversarial:
                optimizer_ad.step()
                ad_net.zero_grad()
            # break

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
    class_correct = [0] * 86
    class_total = [0] * 86
    with torch.no_grad():
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            pos_input_ids,pos_attention_mask,labels = batch

            scores = model(pos_input_ids,pos_attention_mask,labels)[1]
            preds = torch.max(scores, dim=1)[1]
            for label, pred in zip(labels.flatten(), preds):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1
            
            correct = torch.eq(torch.max(scores, dim=1)[1], labels.flatten()).float()         
            acc = correct.sum().item() / len(correct)

            y_pred.extend(torch.max(scores, dim=1)[1].tolist())
            y_true.extend(labels.flatten().tolist())
            avg_acc.append(acc)
            # break

    class_acc = []
    for i in range(86):
        if class_total[i] > 0:
            class_acc.append(class_correct[i] / class_total[i])
        else:
            class_acc.append(0)

    if args.local_rank in [-1,0]:
        acc = metrics.accuracy_score(y_true,y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        logger.info("acc = {:>5,}".format(acc,'.4f'))
        logger.info("f1 = {}".format(f1))
        logger.info("kappa = {}".format(kappa))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(len(test_dataloader.dataset)))
        logger.info("Per-class accuracy = {}".format(class_acc))
    return acc, f1, kappa, class_acc

def main():
    config_file = 'configs/main_drugbank.yaml'# parser.parse_args().config_file
    args = Args(config_file)
    ### set hyperparameters
    parser = argparse.ArgumentParser(description='TextDDI')
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--adversarial_task', default='S1', type=str, help='which set to use')
    parser.add_argument('--adversarial_weight', default=1, type=float, help='the weight of adversarial loss')
    parser.add_argument('--gamma_split', type=str, default="55", choices=["55","60","65","70"])
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--dataset_type', type=str, default='finger', help='type of dataset')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    op_args = parser.parse_args()
    adversarial = op_args.adversarial
    adversarial_task = op_args.adversarial_task
    adversarial_weight = op_args.adversarial_weight
    gamma_split = op_args.gamma_split
    if op_args.dataset_type == 'random':
        dtp = op_args.dataset_type
    elif op_args.dataset_type == 'fin':
        dtp = op_args.dataset_type
    else:
        dtp = op_args.dataset_type + '_' + str(gamma_split)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format("drugbank_" + start_date +  '_' + dtp + ('_adv' if adversarial else ''),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    else: ### usually not eval
        log_path = './log/{}/{}.log'.format("drugbank_" + start_date + '_' + dtp + ('_adv' if adversarial else ''),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
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

    if args.eval:### pass
        model = MODEL_CLASSES[args.model_type](args)
        # checkpoint = torch.load(args.checkpoint,map_location='cpu')

        # state_dict = checkpoint['model']
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     if key.startswith('module.'):
        #         new_state_dict[key[7:]] = value
        #     else:
        #         new_state_dict[key] = value

        # model.load_state_dict(new_state_dict,strict=True)
        model.to(args.device)
        dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test')
        dev_sampler = RandomSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
        test_acc = evaluate(dev_dataloader,model,args,logger)
        # logger.info("dev_acc{}".format(test_acc))
        return
    start_epoch = 0

    model = MODEL_CLASSES[args.model_type](args)
    if args.resume: ### pass
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        state_dict = checkpoint['model']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict,strict=True)
        model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
    ]

    if adversarial: # drugbank: 86
        random_layer = RandomLayer([768, 86], 500).to(args.device)
        random_layer.device(args.device)
        ad_net = AdversarialNetwork(500, 500).to(args.device)
        optimizer_ad = Adam(ad_net.parameters(), eps = args.epsilon,betas=(0.9,0.98), lr=args.lr)
        # optimizer_ad = Adam(ad_net.parameters(), lr=0.0003)
    else:
        random_layer = None
        ad_net = None
        optimizer_ad = None

    start_epoch = 0
    optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    
    model.to(args.device)

    global_step = 0
    best_performance = 0
    best_checkpoint_path = {}
    # train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train'),'rb'))
    patience = args.patience
    # patience = 1
    fail_time = 0

    logger.info("choose split as: {}".format(dtp))

    fail_time = {}
    fail_time['S0'], fail_time['S1'], fail_time['S2'] = 0, 0, 0
    best_performance = {}
    best_performance['S0'], best_performance['S1'], best_performance['S2'] = 0, 0, 0
    if args.local_rank != -1: ### pass
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
                                                         #  find_unused_parameters=True)

    for epoch in range(int(args.num_train_epochs)):

        if fail_time['S0'] >= patience and fail_time['S1'] >= patience and fail_time['S2'] >= patience:
            break

        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]: ### local_rank = -1
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = DATASET_MODEL_CLASS[args.dataset](args,'train', dtp)
        train_sampler = RandomSampler(train_dataset) # if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=4)
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

        ### valid and test part for benchmark
        if args.local_rank in [-1,0]:
            for setting in ['S0', 'S1', 'S2']:
                dev_data = DATASET_MODEL_CLASS[args.dataset](args,'valid_' + setting, dtp)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
                dev_acc, dev_f1, dev_kappa, dev_class_acc = evaluate(dev_dataloader,model,args,logger)
                writer.add_scalar('dev_acc', dev_acc, epoch)
                logger.info("epoch={}, setting {}, dev_acc={}, dev_f1={}, dev_kappa={}".format(epoch, setting, dev_acc, dev_f1, dev_kappa))
                if dev_acc >= best_performance[setting]: #epoch % 10 == 0: # :
                    checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
                    if not os.path.exists(checkpoints_dir):
                        os.makedirs(checkpoints_dir)
                    checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}_{}.pt'.format(epoch, setting))
                    best_checkpoint_path[setting] = checkpoint_path
                    best_performance[setting] = dev_acc
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},checkpoint_path)
                    logger.info('Save best checkpoint to {}'.format(checkpoint_path))
                    fail_time[setting] = 0
                else:
                    fail_time[setting] += 1
    if args.test:
        if args.local_rank in [-1,0]:
            for setting in ['S0', 'S1', 'S2']:
                logger.info("test best_checkpoint_path={}".format(best_checkpoint_path[setting]))
                checkpoint = torch.load(best_checkpoint_path[setting])
                model.load_state_dict(checkpoint['model'])
                dev_data = DATASET_MODEL_CLASS[args.dataset](args,'test_' + setting, dtp)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
                test_acc, test_f1, test_kappa, test_class_acc = evaluate(dev_dataloader,model,args,logger)
                # logger.info("best epoch={},test_f1={}".format(checkpoint['epoch'], test_acc))
                logger.info("epoch={}, setting {}, test_acc={}, test_f1={}, test_kappa={}, test_class_acc={}".format(epoch, setting, test_acc, test_f1, test_kappa, test_class_acc))
        
    ### the former valid and test part
    #     if args.local_rank in [-1,0]:
    #         dev_data = DATASET_MODEL_CLASS[args.dataset](args,'dev')
    #         dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
    #         dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
    #         dev_acc = evaluate(dev_dataloader,model,args,logger)
    #         writer.add_scalar('dev_acc', dev_acc, epoch)
    #         logger.info("epoch={},dev_acc={}".format(epoch,dev_acc))
    #         if dev_acc > best_performance: #epoch % 10 == 0: # :
    #             checkpoints_dir = './checkpoints/{}/{}'.format(start_date,args.annotation)
    #             if not os.path.exists(checkpoints_dir):
    #                 os.makedirs(checkpoints_dir)
    #             checkpoint_path = os.path.join(checkpoints_dir,'checkpoint_epoch{}.pt'.format(epoch))
    #             best_checkpoint_path = checkpoint_path
    #             best_performance = dev_acc
    #             torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch},checkpoint_path)
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
    #         dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=4)
    #         test_acc = evaluate(dev_dataloader,model,args,logger)
    #         logger.info("best epoch={},test_f1={}".format(checkpoint['epoch'], test_acc))

if __name__=='__main__':
    main()

