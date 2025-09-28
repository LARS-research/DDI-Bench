import gc
import torch
from drugbank_dataset_rl import drugbank_dataset_rl
import yaml
from datetime import date
import datetime
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter  
from torch import nn
from transformers import BioGptModel, BioGptForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from sklearn import metrics
import setproctitle
import json
from torch.optim import Adam
import math

setproctitle.setproctitle("ddigpt_drugbank_shenzhenqian")

def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))   

class Args():
    def __init__(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.args_dict = config
            for k, v in config.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
    
    def to_str(self):
        mstr = '\n'
        for k,v in self.args_dict.items():
            mstr += k +': '+str(v)+'\n'
        return mstr

class biogpt_cls(nn.Module):
    def __init__(self, args):
        super(biogpt_cls, self).__init__()
        self.args = args

        self.llm = BioGptModel.from_pretrained(args.pretrained_model_path)

        self.config = self.llm.config
        self.activation = {}
        self.linear = nn.Linear(1024, 86)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,input_ids,attention_mask,labels
    ):
        batch_size = input_ids.size(0)
        outputs1 = self.llm(
            input_ids=input_ids[:, :64],
            attention_mask=attention_mask[:, :64]
        )[0]
        outputs2 = self.llm(
            input_ids=input_ids[:, 64:],
            attention_mask=attention_mask[:, 64:]
        )[0]
        outputs = self.dropout(outputs1 + outputs2)
        output_logits = self.linear(outputs)  # Combine the outputs from both
        final_outputs = output_logits[torch.arange(batch_size, device=output_logits.device), -1]
        loss = torch.nn.CrossEntropyLoss()(final_outputs, labels.view(-1))
        return loss, final_outputs

        # return outputs.loss, outputs.logits

class easy_bioclass(nn.Module):
    def __init__(self, args):
        super(easy_bioclass, self).__init__()
        self.args = args
        self.llm = BioGptForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=86)
    
    def forward(
        self,input_ids,attention_mask,labels
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits


def train(args,train_dataloader,model,optimizer,writer,logger=None,global_step=0):
    avg_loss, avg_acc = [],[]

    data_loader = train_dataloader

    model.zero_grad()
    t0 = time.time()

    for step, batch_all in enumerate(data_loader):
        model.train()
        batch = tuple(t.to(args.device) for t in batch_all)
        # output = model(*(tuple(batch)[0:3]))
        output = model(batch[0], batch[1], batch[2])  # input_ids, attention_mask, labels

        loss = output[0]

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        loss = loss.item()
        avg_loss.append(loss)
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
            logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Elapsed:{:}.'
            .format(step+1, len(train_dataloader),format(loss, '.4f'),elapsed))

    avg_loss = np.array(avg_loss).mean()
    return avg_loss,global_step

def evaluate(test_dataloader,model,args,logger):
    avg_acc = []
    model.eval()   

    y_pred = []
    y_true = []
    cbr_num = 0
    hit_num = 0
    with torch.no_grad():
        # for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            pos_input_ids,pos_attention_mask,labels = batch

            scores = model(pos_input_ids,pos_attention_mask,labels)[1]
            
            correct = torch.eq(torch.max(scores, dim=1)[1], labels.flatten()).float()         
            acc = correct.sum().item() / len(correct)

            y_pred.extend(torch.max(scores, dim=1)[1].tolist())
            y_true.extend(labels.flatten().tolist())
            avg_acc.append(acc)

    if args.local_rank in [-1,0]:
        acc = metrics.accuracy_score(y_true,y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average=None).mean() 
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        logger.info("acc = {:>5,}".format(acc,'.4f'))
        logger.info("f1 = {}".format(f1))
        logger.info("kappa = {}".format(kappa))
        logger.info("len = {}".format(len(y_true)))
        logger.info("len dataset= {}".format(len(test_dataloader.dataset)))
    
    return acc, f1, kappa

def main():
    config_file = 'configs/main_drugbank.yaml'# parser.parse_args().config_file
    args = Args(config_file)

    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format("drugbank_" + start_date,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    else: ### usually not eval
        log_path = './log/{}/{}.log'.format("drugbank_" + start_date,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache() 

    args.local_rank = -1
    device = "cuda:"+ args.gpuid if torch.cuda.is_available() else "cpu"

    args.device = device

    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=log_path,
        filemode=args.filemode)
    logger = logging.getLogger()
    logger.info("Process rank: {}, device: {}, distributed training: {}".format(args.local_rank,device, bool(args.local_rank != -1)))
    logger.info("Training/evaluation parameters %s", args.to_str())

    tensorboard_path = './tensorboard/{}/{}'.format(start_date,args.annotation)
    if not os.path.exists(os.path.dirname(tensorboard_path)):
        os.makedirs(os.path.dirname(tensorboard_path))
    writer = SummaryWriter(tensorboard_path)

    # model = biogpt_cls(args) 
    model = easy_bioclass(args) 

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)

    model.to(args.device)

    eval_set = ['S0', 'S1', 'S2']

    global_step = 0
    best_performance = {}
    best_checkpoint_path = {}
    patience = args.patience
    fail_time = {}
    for setting in eval_set:
        best_performance[setting] = 0
        fail_time[setting] = 0
    # fail_time['S1'], fail_time['S2'] = 0, 0
    # best_performance['S1'], best_performance['S2'] = 0, 0

    for epoch in range(int(args.num_train_epochs)):

        if False not in [fail_time[setting] >= patience for setting in eval_set]:
            break
        if args.local_rank in [-1,0]: ### local_rank = -1
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = drugbank_dataset_rl(args,'train')
        train_sampler = RandomSampler(train_dataset) # if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=4)

        train_dataloader_adv = None

        torch.cuda.empty_cache()
        train_loss,global_step = train(args, train_dataloader,model,optimizer,writer,logger,global_step)
        if args.local_rank in [-1,0]:
            logger.info('epoch={},loss={}'.format(epoch, train_loss))
        torch.cuda.empty_cache()
        gc.collect()

        ### valid and test part for benchmark
        if args.local_rank in [-1,0]:
            for setting in eval_set:
            # for setting in ['S2']:
                dev_data = drugbank_dataset_rl(args,'valid_' + setting)
                # dev_sampler = RandomSampler(dev_data) 
                dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.eval_batch_size,num_workers=4)
                dev_acc, dev_f1, dev_kappa = evaluate(dev_dataloader,model,args,logger)
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
            for setting in eval_set:
            # for setting in ['S2']:
                logger.info("test best_checkpoint_path={}".format(best_checkpoint_path[setting]))
                checkpoint = torch.load(best_checkpoint_path[setting])
                model.load_state_dict(checkpoint['model'])
                dev_data = drugbank_dataset_rl(args,'test_' + setting)
                # dev_sampler = RandomSampler(dev_data) 
                dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.eval_batch_size,num_workers=4)
                test_acc, test_f1, test_kappa = evaluate(dev_dataloader,model,args,logger)
                logger.info("setting {}, test_acc={}, test_f1={}, test_kappa={}. ".format(setting, test_acc, test_f1, test_kappa))
                # logger.info("best epoch={},test_f1={}".format(checkpoint['epoch'], test_acc))

if __name__=='__main__':
    main()

