from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


class twosides_dataset_rl(Dataset):
    def __init__(self,args,state,adv=None):
        with open(args.info_dict_path, 'r') as file:
            self.info_dict = json.load(file)
        ### new part
        # self.data_path = dtp
        # used_set = [j for j in self.info_dict]

        self.data = [] ### here to see which is lacked [i for i in range(645) if i not in filter_set]
        with open('./data/twosides_{}/{}.txt'.format(args.split_strategy, state), 'r') as file:
            flag = 0
            temp = []
            for j in file:
                h,t,r,p = j[:-1].split(' ')
                r_0 = r.split(',')
                list_cun = [int(j) for j in r_0]
                temp.append([int(h), int(t), list_cun])
                if flag == 0:
                    flag = 1
                    continue
                if flag == 1:
                    # if int(h) in filter_set and int(t) in filter_set and temp[0][0] in filter_set and temp[0][1] in filter_set:
                    self.data.append([str(temp[0][0]), str(temp[0][1]), h, t, temp[0][2]])
                    temp = []
                    flag = 0
        
        if adv is not None:
            self.data = ((int(adv/len(self.data)) + 1)*self.data)[:adv]

        ### old part
        # with open('./data/twosides/id2drug.json', 'r') as file:
        #     id2drug = json.load(file)
        # self.cid2id = {}
        # for key,value in id2drug.items():
        #     cid = str(int(value['cid'][3:]))
        #     self.cid2id[cid] = key
        # self.id2cid = {}
        # for key,value in self.cid2id.items():
        #     self.id2cid[value] = key
        # if args.use_zero_shot:
        #     with open('./data/twosides/{}_twosides_zero_shot.pkl'.format(state), 'rb') as file:
        #         self.data = pickle.load(file)
        # self.data = self.data[:100]
        
        
            # with open('./data/twosides/{}_ddi.txt'.format(state), 'r') as file:
            #     self.data = []
            #     data = file.readlines()
            #     data = [data[i:i+2] for i in range(0,len(data),2)]
            #     for item in data:
            #         drug1_id,drug2_id,rel_ids,_ = item[0].strip().split('\t')
            #         drug3_id,drug4_id,_,_ = item[1].strip().split('\t')
            #         rel_ids = [int(i) for i in rel_ids.split(',')]
            #         self.data.append([drug1_id,drug2_id,drug3_id,drug4_id,rel_ids])
        # else:
        #     with open('./data/twosides/{}.txt'.format(state), 'r') as file:
        #         self.data = []
        #         data = file.readlines()
        #         data = [data[i:i+2] for i in range(0,len(data),2)]
        #         for item in data:
        #             drug1_id,drug2_id,rel_ids,_ = item[0].strip().split('\t')
        #             drug3_id,drug4_id,_,_ = item[1].strip().split('\t')
        #             rel_ids = [int(i) for i in rel_ids.split(',')]
        #             self.data.append([drug1_id,drug2_id,drug3_id,drug4_id,rel_ids])
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        self.state = state
        self.bg_max_length = int(200*(self.args.max_length/512))
        # gpu_num = self.args.gpu_num # torch.cuda.device_count()
        # pad_num = gpu_num - len(self.data)%gpu_num
        # if self.state != 'train' and  args.multi_gpu and pad_num!=0:
        #     for i in range(pad_num):
        #         self.data.append(None)
        # self.pad_num = pad_num
        # self.true_data_len = len(self.data)-self.pad_num
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        drug1_id,drug2_id,drug3_id,drug4_id,rel_ids = self.data[index]
        # drug1_cid,drug2_cid = self.id2cid[drug1_id],self.id2cid[drug2_id]
        # drug3_cid,drug4_cid = self.id2cid[drug3_id],self.id2cid[drug4_id]
        drug1_name,drug1_summary = self.info_dict[str(drug1_id)].values()
        drug2_name,drug2_summary = self.info_dict[str(drug2_id)].values()
        drug3_name,drug3_summary = self.info_dict[str(drug3_id)].values()
        drug4_name,drug4_summary = self.info_dict[str(drug4_id)].values()

        prompt_pos = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug1_name,drug2_name)
        prompt_neg = "In the above context, we can predict that the drug-drug interactions between {} and {} is that: ".format(drug3_name,drug4_name)
        prompt = prompt_pos
        # prompt_tokenized = self.tokenizer(prompt)
        # prompt_input_ids = prompt_tokenized['input_ids']
        # prompt_attention_mask = prompt_tokenized['attention_mask']
        # prompt_length = len(prompt_attention_mask)

        if self.args.drug_name_only:
            prompt_pos_tokenized = self.tokenizer(prompt_pos,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.drug_only_max_length)
            prompt_neg_tokenized = self.tokenizer(prompt_neg,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.drug_only_max_length)
            example = [prompt_pos_tokenized['input_ids'],prompt_pos_tokenized['attention_mask'],rel_ids,prompt_neg_tokenized['input_ids'],prompt_neg_tokenized['attention_mask']]
            example = [torch.tensor(t,dtype=torch.long) for t in example]
            return example
        
        # if self.args.not_random:
        drug1_summary_0 = self.tokenizer(drug1_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug1_summary_ = self.tokenizer.decode(drug1_summary_0)
        # if drug1_summary_[-1]!='.':
        #     drug1_summary_ = drug1_summary_[:-1] + '.'

        drug2_summary_0 = self.tokenizer(drug2_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug2_summary_ = self.tokenizer.decode(drug2_summary_0)
        # if drug2_summary_[-1]!='.':
        #     drug2_summary_ = drug2_summary_[:-1] + '.'

        prompt = drug1_summary_ + "</s>" + drug2_summary_ 

        prompt_tokenized1 = self.tokenizer(prompt,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length)

        input_ids = [2 for j in range(len(prompt_tokenized1['input_ids'])*2)]
        input_ids[1:len(drug1_summary_0)+1] = drug1_summary_0
        input_ids[len(prompt_tokenized1['input_ids']) + 2 + len(drug1_summary_0): len(prompt_tokenized1['input_ids']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = drug2_summary_0
        prompt_tokenized1['input_ids'] = input_ids

        mask_ids = [0 for j in range(len(prompt_tokenized1['attention_mask']) * 2)]
        mask_ids[1:len(drug1_summary_0)+1] = [1 for j in range(len(drug1_summary_0))]
        mask_ids[len(prompt_tokenized1['attention_mask']) + 2 + len(drug1_summary_0): len(prompt_tokenized1['attention_mask']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = [1 for j in range(len(drug2_summary_0))]
        prompt_tokenized1['attention_mask'] = mask_ids
        
        drug3_summary_0 = self.tokenizer(drug3_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug3_summary_ = self.tokenizer.decode(drug3_summary_0)

        drug4_summary_0 = self.tokenizer(drug4_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug4_summary_ = self.tokenizer.decode(drug4_summary_0)

        prompt = drug3_summary_ + "</s>" + drug4_summary_
        prompt_tokenized2 = self.tokenizer(prompt,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length)
        
        input_ids = [2 for j in range(len(prompt_tokenized2['input_ids'])*2)]
        input_ids[1:len(drug3_summary_0)+1] = drug3_summary_0
        input_ids[len(prompt_tokenized2['input_ids']) + 2 + len(drug3_summary_0): len(prompt_tokenized2['input_ids']) + 2 + len(drug3_summary_0) + len(drug4_summary_0)] = drug4_summary_0
        prompt_tokenized2['input_ids'] = input_ids

        mask_ids = [0 for j in range(len(prompt_tokenized2['attention_mask']) * 2)]
        mask_ids[1:len(drug3_summary_0)+1] = [1 for j in range(len(drug3_summary_0))]
        mask_ids[len(prompt_tokenized2['attention_mask']) + 2 + len(drug3_summary_0): len(prompt_tokenized2['attention_mask']) + 2 + len(drug3_summary_0) + len(drug4_summary_0)] = [1 for j in range(len(drug4_summary_0))]
        prompt_tokenized2['attention_mask'] = mask_ids

        example = [prompt_tokenized1['input_ids'],prompt_tokenized1['attention_mask'],rel_ids,prompt_tokenized2['input_ids'],prompt_tokenized2['attention_mask']]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example
        

        # drug1_sent_num = len(drug1_sent_list)
        # drug2_sent_num = len(drug2_sent_list)
        # sent_tokenized_list = drug1_sent_tokenized_list+drug2_sent_tokenized_list
        # sent_num = drug1_sent_num + drug2_sent_num
        # selected_idxs = []
        # latest_context_tokenized1 = self.tokenizer(prompt_pos,
        #                     add_special_tokens=True,
        #                     return_token_type_ids=False,
        #                     padding="max_length",
        #                     truncation=True,
        #                     max_length=self.args.max_length)
        # while True:
        #     if len(selected_idxs) == sent_num:
        #         break
        #     idx = random.randint(0,sent_num-1)
        #     if idx in selected_idxs:
        #         continue
        #     selected_idxs.append(idx)

        #     sorted(selected_idxs)
            
        #     drug1_sent_select_idx = []
        #     drug2_sent_select_idx = []
        #     for i in list(selected_idxs):
        #         if i<drug1_sent_num:
        #             drug1_sent_select_idx.append(i)
        #         elif i < sent_num:
        #             drug2_sent_select_idx.append(i)
        #         else:
        #             assert False

        #         drug1_sent_input_ids = []
        #         for item in drug1_sent_select_idx:
        #             drug1_sent_input_ids.extend(sent_tokenized_list[item])
        #         drug2_sent_input_ids = []
        #         for item in drug2_sent_select_idx:
        #             drug2_sent_input_ids.extend(sent_tokenized_list[item])

        #         prompt_input_ids = self.tokenizer(prompt_pos,add_special_tokens=False)['input_ids']

        #         if len(drug1_sent_select_idx)==0:
        #             current_drug1_name_postfix_input_ids = []
        #         else:
        #             current_drug1_name_postfix_input_ids = drug1_name_postfix_input_ids
        #         if len(drug2_sent_select_idx)==0:
        #             current_drug2_name_postfix_input_ids = []
        #         else:
        #             current_drug2_name_postfix_input_ids = drug2_name_postfix_input_ids
        #         # if len(drug1_sent_select_idx)==0:
        #         #     drug1_name_postfix_input_ids = []
        #         # if len(drug2_sent_select_idx)==0:
        #         #     drug2_name_postfix_input_ids = []
        #         input_ids = current_drug1_name_postfix_input_ids+drug1_sent_input_ids+current_drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids
        #         # input_ids = drug1_name_postfix_input_ids+drug1_sent_input_ids+drug2_name_postfix_input_ids+drug2_sent_input_ids+prompt_input_ids
        #         context = self.tokenizer.decode(input_ids)

        #         # true_context_length = len(self.tokenizer(context,add_special_tokens=True)['input_ids'])

        #         context_tokenized = self.tokenizer(context,
        #                     add_special_tokens=True,
        #                     return_token_type_ids=False,
        #                     padding="max_length",
        #                     truncation=True,
        #                     max_length=self.args.max_length)
            
        #         bad = False
        #         if sum(context_tokenized['attention_mask']) == self.args.max_length:
        #             bad = True
        #         if bad:
        #             break
        #         else:
        #             latest_context_tokenized1 = context_tokenized

        
        # drug3_sent_num = len(drug3_sent_list)
        # drug4_sent_num = len(drug4_sent_list)
        # sent_tokenized_list = drug3_sent_tokenized_list+drug4_sent_tokenized_list
        # sent_num = drug3_sent_num + drug4_sent_num
        # selected_idxs = []
        # latest_context_tokenized2 = self.tokenizer(prompt_neg,
        #                     add_special_tokens=True,
        #                     return_token_type_ids=False,
        #                     padding="max_length",
        #                     truncation=True,
        #                     max_length=self.args.max_length)
        # while True:
        #     if len(selected_idxs) == sent_num:
        #         break
        #     idx = random.randint(0,sent_num-1)
        #     if idx in selected_idxs:
        #         continue
        #     selected_idxs.append(idx)

        #     sorted(selected_idxs)
            
        #     drug3_sent_select_idx = []
        #     drug4_sent_select_idx = []
        #     for i in list(selected_idxs):
        #         if i<drug3_sent_num:
        #             drug3_sent_select_idx.append(i)
        #         elif i < sent_num:
        #             drug4_sent_select_idx.append(i)
        #         else:
        #             assert False

        #         drug3_sent_input_ids = []
        #         for item in drug3_sent_select_idx:
        #             drug3_sent_input_ids.extend(sent_tokenized_list[item])
        #         drug4_sent_input_ids = []
        #         for item in drug4_sent_select_idx:
        #             drug4_sent_input_ids.extend(sent_tokenized_list[item])

        #         prompt_input_ids = self.tokenizer(prompt_neg,add_special_tokens=False)['input_ids']

        #         if len(drug3_sent_select_idx)==0:
        #             current_drug3_name_postfix_input_ids = []
        #         else:
        #             current_drug3_name_postfix_input_ids = drug3_name_postfix_input_ids
        #         if len(drug4_sent_select_idx)==0:
        #             current_drug4_name_postfix_input_ids = []
        #         else:
        #             current_drug4_name_postfix_input_ids = drug4_name_postfix_input_ids

        #         # if len(drug3_sent_select_idx)==0:
        #         #     drug3_name_postfix_input_ids = []
        #         # if len(drug4_sent_select_idx)==0:
        #         #     drug4_name_postfix_input_ids = []
        #         # input_ids = drug3_name_postfix_input_ids+drug3_sent_input_ids+drug4_name_postfix_input_ids+drug4_sent_input_ids+prompt_input_ids
        #         input_ids = current_drug3_name_postfix_input_ids+drug3_sent_input_ids+current_drug4_name_postfix_input_ids+drug4_sent_input_ids+prompt_input_ids
        #         context = self.tokenizer.decode(input_ids)

        #         # true_context_length = len(self.tokenizer(context,add_special_tokens=True)['input_ids'])

        #         context_tokenized = self.tokenizer(context,
        #                     add_special_tokens=True,
        #                     return_token_type_ids=False,
        #                     padding="max_length",
        #                     truncation=True,
        #                     max_length=self.args.max_length)
            
        #         bad = False
        #         if sum(context_tokenized['attention_mask']) == self.args.max_length:
        #             bad = True
        #         if bad:
        #             break
        #         else:
        #             latest_context_tokenized2 = context_tokenized

        # example = [latest_context_tokenized1['input_ids'],latest_context_tokenized1['attention_mask'],rel_ids,latest_context_tokenized2['input_ids'],latest_context_tokenized2['attention_mask']]
        # example = [torch.tensor(t,dtype=torch.long) for t in example]
        # return example

