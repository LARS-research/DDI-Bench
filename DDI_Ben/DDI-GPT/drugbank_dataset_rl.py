from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
import pickle
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class drugbank_dataset_rl(Dataset):
    def __init__(self,args,state,adv=None):
        with open(args.ddi_dict_path, 'r') as file:
            self.ddi_dict = json.load(file)

        with open('./data/drugbank_{}/{}.txt'.format(args.split_strategy, state), 'r') as file:
            self.data = [[int(num) for num in item.strip().split()] for item in file.readlines()]

        if adv is not None:
            self.data = ((int(adv/len(self.data)) + 1)*self.data)[:adv]

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        if 'roberta' not in args.pretrained_model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.state = state
        self.bg_max_length = int(200*(self.args.max_length/512))
        # self.softmaxed_sim_martix = torch.load('./data/softmaxed_sim_martix.pt')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        drug1_id,drug2_id,rel_id = self.data[index]
        # drug1_accession,drug2_accession = self.id2accession[drug1_id],self.id2accession[drug2_id]
        drug1_name,drug1_summary = self.ddi_dict[str(drug1_id)].values()
        drug2_name,drug2_summary = self.ddi_dict[str(drug2_id)].values()
      
        prompt = "The drug-drug interactions between {} and {} is: ".format(drug1_name,drug2_name)

        if self.args.drug_name_only:
            prompt_tokenized = self.tokenizer(prompt,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.drug_only_max_length)
            example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
            example = [torch.tensor(t,dtype=torch.long) for t in example]
            return example
        
        # if self.args.not_random: ### pass
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
        # + " " + prompt 
        prompt_tokenized = self.tokenizer(prompt,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length)
        # prompt_tokenized['attention_mask'] = attention_mask_generation(len(drug1_summary_0), len(drug2_summary_0), len(prompt_tokenized['attention_mask']))

        input_ids = [2 for j in range(len(prompt_tokenized['input_ids'])*2)]
        input_ids[1:len(drug1_summary_0)+1] = drug1_summary_0
        input_ids[len(prompt_tokenized['input_ids']) + 2 + len(drug1_summary_0): len(prompt_tokenized['input_ids']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = drug2_summary_0
        prompt_tokenized['input_ids'] = input_ids

        mask_ids = [0 for j in range(len(prompt_tokenized['attention_mask']) * 2)]
        mask_ids[1:len(drug1_summary_0)+1] = [1 for j in range(len(drug1_summary_0))]
        mask_ids[len(prompt_tokenized['attention_mask']) + 2 + len(drug1_summary_0): len(prompt_tokenized['attention_mask']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = [1 for j in range(len(drug2_summary_0))]
        prompt_tokenized['attention_mask'] = mask_ids

        example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example
