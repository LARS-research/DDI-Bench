import gc
import torch
# from drugbank_dataset_rl import drugbank_dataset_rl
import yaml
from datetime import date
import datetime
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter  
from torch import nn
from transformers import BioGptModel
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from sklearn import metrics
import tqdm
import setproctitle
from twosides_dataset_rl import twosides_dataset_rl
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from torch.optim import Adam
import math

setproctitle.setproctitle("ddigpt_twosides_shenzhenqian")

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        self.linear = nn.Linear(1024, 209)
        self.dropout = nn.Dropout(0.1)
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,pos_input_ids,pos_attention_mask,labels,neg_input_ids,neg_attention_mask
    ):
        batch_size,max_length = pos_input_ids.size()

        ### version for easier way
        input_ids = torch.cat((pos_input_ids,neg_input_ids),dim=0)
        attention_mask = torch.cat((pos_attention_mask,neg_attention_mask),dim=0)
        
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        output_logits = self.linear(outputs)
        final_outputs = output_logits[torch.arange(batch_size*2, device=output_logits.device), -1]
        logits = torch.sigmoid(final_outputs)

        pos_logits,neg_logits = logits[0:batch_size],logits[batch_size:]

        losses = []
        for pos,neg,label in zip(pos_logits,neg_logits,labels):
            pos_scores = pos[label>0]
            neg_scores = neg[label>0]
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            label = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))], dim=0).to(pos_scores.device)
            loss = self.bce_loss(scores, label) 
            losses.append(loss)
        
        loss = torch.mean(torch.stack(losses,dim=0))

        return loss,pos_logits,neg_logits

def train(args,train_dataloader,model,optimizer,writer,logger=None,global_step=0):
    t0 = time.time()
    avg_loss, avg_acc = [],[]

    data_loader = train_dataloader

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    model.zero_grad()

    for step, batch_all in enumerate(data_loader):
        model.train()

        batch = tuple(t.to(args.device) for t in batch_all)

        output = model(*(tuple(batch)))

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
        # for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            # if len(batch)==5:
            pos_input_ids,pos_attention_mask,labels,neg_input_ids,neg_input_ids = batch

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
                'label': labels[:,r]}
    
    # if len(batch)==5:
    return np.mean(roc_auc), np.mean(prc_auc), np.mean(ap)
    # else:
    #     return np.mean(roc_auc_pos), np.mean(prc_auc_pos), np.mean(ap_pos),0,0,0

def main():
    config_file = 'configs/main_twosides.yaml'# parser.parse_args().config_file
    args = Args(config_file)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format("twosides_" + start_date,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    else: ### usually not eval
        log_path = './log/{}/{}.log'.format("twosides_" + start_date,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache()
    seed_everything(args.seed)

    args.local_rank = -1
    device = "cuda:"+ args.gpuid if torch.cuda.is_available() else "cpu"
    
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

    model = biogpt_cls(args)
    if args.resume:
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True)
        model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
    ]

    start_epoch = 0
    optimizer = torch.optim.Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    model.to(args.device)

    eval_set = ['S0', 'S1', 'S2']

    global_step = 0
    best_checkpoint_path = None

    patience = args.patience

    # logger.info("choose split as: {}".format(dtp))

    fail_time = {}
    best_performance = {}
    # fail_time['S1'], fail_time['S2'] = 0, 0
    # best_performance['S1'], best_performance['S2'] = 0, 0
    best_checkpoint_path = {}
    for setting in eval_set:
        best_performance[setting] = 0
        fail_time[setting] = 0

    for epoch in range(int(args.num_train_epochs)):
        # if fail_time['S1'] >= patience and fail_time['S2'] >= patience:
        #     break
        if False not in [fail_time[setting] >= patience for setting in eval_set]:
            break

        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = twosides_dataset_rl(args,'train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=0)


        torch.cuda.empty_cache()
        train_loss,global_step = train(args, train_dataloader,model,optimizer,writer,logger,global_step)
        if args.local_rank in [-1,0]:
            logger.info('epoch={},loss={}'.format(epoch, train_loss))
        torch.cuda.empty_cache()
        gc.collect()

        if args.local_rank in [-1,0]:
            # for setting in ['S1', 'S2']:
            for setting in eval_set:
                dev_data = twosides_dataset_rl(args,'valid_' + setting)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
                roc_auc, prc_auc, ap = evaluate(dev_dataloader,model,args,logger)
                # writer.add_scalar('dev_acc', dev_acc, epoch)
                logger.info("epoch={}, setting {},roc_auc={:.4f},prc_auc={:.4f},ap={:.4f}".format(epoch,setting,roc_auc,prc_auc,ap))
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
            # for setting in ['S1', 'S2']:
            for setting in eval_set:
                logger.info("test best_checkpoint_path={}".format(best_checkpoint_path[setting]))
                checkpoint = torch.load(best_checkpoint_path[setting])
                model.load_state_dict(checkpoint['model'])
                dev_data = twosides_dataset_rl(args,'test_' + setting)
                dev_sampler = RandomSampler(dev_data) # if args.local_rank == -1 else DistributedSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
                roc_auc, prc_auc, ap = evaluate(dev_dataloader,model,args,logger)

                logger.info("test best_checkpoint_path={}, setting {}, roc_auc={:.4f},prc_auc={:.4f},ap={:.4f}".format(best_checkpoint_path[setting], setting,roc_auc,prc_auc,ap))

if __name__=='__main__':
    main()