import os
from torch.utils.data import Dataset, DataLoader
from utils import *
from collections import defaultdict as ddict
import torch
import json

from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA

### SSI-DDI ### 

import itertools
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np
import math

### SSI-DDI ### 


num_ent = {'drugbank': 1710, 'twosides': 645, 'HetioNet': 34124}
num_rel = {'drugbank': 86, 'twosides': 209} # 209, 309

class Data_record():
    def __init__(self, args):
        self.args = args

        folder_name = args.dataset + '_' + args.dataset_type

        self.link_aug_num = 0

        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        self.triplets = load_data(args)
        # self.triplets_all = self.triplets['train'] + self.triplets['valid'] + self.triplets['test']

        self.data = ddict(list)
        sr2o = ddict(set)

        self.link_aug_num = 0

        self.num_rel, self.args.num_rel = num_rel[args.dataset] + self.link_aug_num, num_rel[args.dataset] + self.link_aug_num
        
        self.num_ent, self.args.num_ent = num_ent[args.dataset], num_ent[args.dataset]

        self.include_splits = list(self.triplets.keys())
        self.split_not_train = [j for j in self.include_splits if j != 'train']
        
        for split in self.include_splits:
            if split == 'train' and self.args.model in ['MSTE'] and args.dataset == 'twosides': 
                for j in range(int(len(self.triplets[split])/2)):
                    sub, obj, rel, neg_add = self.triplets[split][j*2][0], self.triplets[split][j*2][1], np.where(np.array(self.triplets[split][j*2][2])[:-1]==1)[0], [self.triplets[split][j*2+1][0], self.triplets[split][j*2+1][1]]
                    for k in rel:
                        self.data[split].append((sub, obj, [neg_add[0], neg_add[1], k]))
                        sr2o[(sub, obj)].add((neg_add[0], neg_add[1], k))
            else:
                for j in self.triplets[split]:
                    # sub, obj, rel = self.ent2id[j[0]], self.ent2id[j[1]], self.rel2id[str(j[2])]
                    sub, obj, rel = j[0], j[1], j[2]
                    self.data[split].append((sub, obj, rel))

                    if split == 'train': 
                        if self.args.model in ['Decagon'] and args.dataset == 'twosides':
                            self.true_data = self.data[split]
                        sr2o[(sub, obj)].add(rel)
        
        if args.use_feat:
            self.feat = torch.FloatTensor(np.array(load_feature(args))).to(self.device)
            self.feat_dim = self.feat.shape[1]
        else:
            self.feat = 0
        
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.data = dict(self.data)

        for split in self.split_not_train:
            for sub, obj, rel in self.data[split]:
                sr2o[(sub, obj)].add(rel)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        ### train triples
        if self.args.dataset == 'twosides' and self.args.model in ['MSTE']:
            for sub, obj, rel in self.data['train']:
                self.triples['train'].append({'triple':(sub, obj, -1), 'label': rel, 'sub_samp': 1})
        else:
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        ### valid & test triplets
        for split in self.split_not_train:
            for sub, obj, rel  in self.data[split]:
                self.triples[split].append({'triple': (sub, obj, rel), 	   'label': self.sr2o_all[(sub, obj)]})

        self.triples = dict(self.triples)

        if args.model == 'Decagon':
            self.edge_index = []
            if args.dataset == 'twosides':
                for sub, obj, rel in self.data['train']:
                    if rel[-1] == 1:
                        self.edge_index.append([sub, obj])
            else:
                for sub, obj, rel in self.data['train']:
                    self.edge_index.append([sub, obj])
            trip_de = []
            if args.dataset == 'twosides':
                file = open('./data/initial/twosides/relations_2hop.txt')
            else:
                file = open('./data/initial/drugbank/relations_2hop.txt')
            for j in file:
                str_lin = j.strip().split(' ')
                trip = [int(j) for j in str_lin]
                # if trip[2] in [1,21]:
                if trip[2] in [1, 5, 6, 7, 10, 18]:
                    trip_de.append([trip[0], trip[1]])
            num_begin = num_ent[args.dataset]
            ent_in = np.unique(np.array(trip_de).flatten())
            ind_dict = {}
            for j in ent_in:
                if j >= num_ent[args.dataset]:
                    ind_dict[j] = num_begin
                    num_begin += 1
                else:
                    ind_dict[j] = j
            for j in trip_de:
                j = [ind_dict[j[0]], ind_dict[j[1]]]
                self.edge_index.append(j)

            self.edge_index	= torch.LongTensor(self.edge_index).to(self.device).t()

            if args.use_feat:
                feat = torch.zeros((num_begin, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat
        elif args.model == 'TIGER':
            if args.dataset == 'twosides':
                with open('./data/initial/twosides/cid2id.json', 'r') as file:
                    cid2id = json.load(file)
                with open('./data/initial/twosides/cid2smiles.json', 'r') as file:
                    cid2smiles = json.load(file)
                TG_id2smiles = {str(cid2id[j]):cid2smiles[j] for j in cid2smiles}
                print("load drug smiles graphs!!")
                TG_smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph('data/{}'.format(folder_name), TG_id2smiles)
                print("load networks !!")
                num_node, network_edge_index, network_rel_index, TG_num_rel = read_network('./data/initial/twosides/relations_2hop.txt')
                print("load DDI samples!!")
                TG_labels = 0
                TG_interactions = np.concatenate([np.array([j[:2] for j in self.triplets[k]]) for k in self.include_splits])
                all_contained_drugs = np.unique(TG_interactions)
                all_contained_drugs = set([str(j) for j in all_contained_drugs])
                print("generate subgraphs!!")
                TG_drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(folder_name, all_contained_drugs,network_edge_index, network_rel_index,TG_num_rel, args)

                TG_data_sta = {
                    'num_nodes': num_node + 1,
                    'num_rel_mol': num_rel_mol_update + 1,
                    'num_rel_graph': num_rel_update + 1,
                    'num_interactions': len(TG_interactions),
                    'num_drugs_DDI': len(all_contained_drugs),
                    'max_degree_graph': max_smiles_degree + 1,
                    'max_degree_node': int(max_subgraph_degree)+1
                }

                print(TG_data_sta)
                self.TG_interactions = TG_interactions
                self.TG_labels = TG_labels
                self.TG_smile_graph = TG_smile_graph
                self.TG_drug_subgraphs = TG_drug_subgraphs
                self.TG_data_sta = TG_data_sta

            elif args.dataset == 'drugbank':
                with open('data/initial/drugbank/DB_molecular_feats.pkl', 'rb') as f:
                    x = pkl.load(f, encoding='utf-8')
                TG_id2smiles = {str(j): x['SMILES'][j] for j in range(1710)}
                for j in [   6,  136,  889, 1171, 1239, 1254]:
                    TG_id2smiles[str(j)] = ''

                print("load drug smiles graphs!!")
                TG_smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph('data/{}'.format(folder_name), TG_id2smiles)
                print("load networks !!")
                num_node, network_edge_index, network_rel_index, TG_num_rel = read_network('./data/initial/drugbank/relations_2hop.txt')
                print("load DDI samples!!")
                ### this part remain (need to be simplified)
                TG_triplet_all = np.concatenate([np.array(self.triplets[j]) for j in self.include_splits])
                TG_interactions = TG_triplet_all[:, :2]
                TG_labels = TG_triplet_all[:, 2]
                all_contained_drugs = np.unique(TG_interactions)
                all_contained_drugs = set([str(j) for j in all_contained_drugs])
                print("generate subgraphs!!")
                TG_drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(folder_name, all_contained_drugs,network_edge_index, network_rel_index,TG_num_rel, args)

                TG_data_sta = {
                    'num_nodes': num_node + 1,
                    'num_rel_mol': num_rel_mol_update + 1,
                    'num_rel_graph': num_rel_update + 1,
                    'num_interactions': len(TG_interactions),
                    'num_drugs_DDI': len(all_contained_drugs),
                    'max_degree_graph': max_smiles_degree + 1,
                    'max_degree_node': int(max_subgraph_degree)+1
                }

                print(TG_data_sta)
                self.TG_interactions = TG_interactions
                self.TG_labels = TG_labels
                self.TG_smile_graph = TG_smile_graph
                self.TG_drug_subgraphs = TG_drug_subgraphs
                self.TG_data_sta = TG_data_sta
        elif args.model in ['SSI-DDI', 'SAGAN']:
            if args.dataset == 'drugbank':
                with open('data/initial/drugbank/id2smiles.json', 'r') as file:
                    id2smiles = json.load(file)
            elif args.dataset == 'twosides':
                with open('./data/initial/twosides/cid2id.json', 'r') as file:
                    cid2id = json.load(file)
                with open('./data/initial/twosides/cid2smiles.json', 'r') as file:
                    cid2smiles = json.load(file)
                id2smiles = {str(cid2id[j]):cid2smiles[j] for j in cid2smiles}

            drug_id_mol_graph_tup = [Chem.MolFromSmiles(id2smiles[j].strip()) for j in id2smiles] 
            self.ATOM_MAX_NUM = np.max([m.GetNumAtoms() for m in drug_id_mol_graph_tup])
            self.AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            self.AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            self.AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup))
            max_valence = max(max_valence, 9)
            self.AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

            self.MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)]))
            self.MAX_ATOM_FC = self.MAX_ATOM_FC if self.MAX_ATOM_FC else 0
            self.MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)]))
            self.MAX_RADICAL_ELC = self.MAX_RADICAL_ELC if self.MAX_RADICAL_ELC else 0

            self.MOL_EDGE_LIST_FEAT_MTX = [get_mol_edge_list_and_feat_mtx(mol) for mol in drug_id_mol_graph_tup]

            self.TOTAL_ATOM_FEATS = self.MOL_EDGE_LIST_FEAT_MTX[0][1].shape[-1]
        elif args.model == 'MRCGNN':
            ### feature: self.feat
            idd = np.arange(num_ent[args.dataset])
            iddd = np.random.permutation(idd)
            mrc_y_a = torch.cat((torch.ones(num_ent[args.dataset], 1), torch.zeros(num_ent[args.dataset], 1)), dim=1)
            if args.dataset == 'drugbank':
                edge_edge = np.array(self.triplets['train'])[:,:2].tolist()
                edge_type = np.array(self.triplets['train'])[:,2].tolist()
            elif args.dataset == 'twosides':
                edge_edge, edge_type = [], []
                for sub, obj, rel in self.data['train']:
                    if rel[-1] == 0:
                        continue
                    iddxx = np.where(np.array(rel)[:-1]==1)[0]
                    for ll in iddxx:
                        edge_edge.append([sub, obj])
                        edge_type.append(ll)

            edge_edge += [[b,a] for a,b in edge_edge]
            edge_edge = torch.tensor(edge_edge, dtype=torch.long) ### need add reverse edges in
            edge_type_reverse = edge_type + edge_type
            self.dt_o = DATA.Data(x=self.feat, edge_index=edge_edge.t().contiguous(), edge_type=edge_type_reverse)
            self.feat_a = self.feat[iddd]
            self.dt_s = DATA.Data(x=self.feat_a, edge_index=edge_edge.t().contiguous(), edge_type=edge_type_reverse)
            random.shuffle(edge_type)
            shuffle_edge_type_reverse = edge_type + edge_type
            self.dt_a = DATA.Data(x=self.feat, y = mrc_y_a, edge_type=shuffle_edge_type_reverse)

        ### the main part
        self.data_iter = {}
        if args.model == 'TIGER':
            if args.dataset == 'drugbank':
                self.data_iter['train'] = DataLoader(DTADataset(x=np.array(self.triplets['train'])[:,:2], y=np.array(self.triplets['train'])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
                for j in self.split_not_train:
                    self.data_iter[j] = DataLoader(DTADataset(x=np.array(self.triplets[j])[:,:2], y=np.array(self.triplets[j])[:,2], sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
            elif args.dataset == 'twosides':
                self.data_iter['train'] = DataLoader(DTADataset(x=np.array([k[:2] for k in self.triplets['train']]), y=np.array([list(k[2]) for k in self.triplets['train']]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
                for j in self.split_not_train:
                    self.data_iter[j] = DataLoader(DTADataset(x=np.array([k[:2] for k in self.triplets[j]]), y=np.array([list(k[2]) for k in self.triplets[j]]), sub_graph=TG_drug_subgraphs, smile_graph=TG_smile_graph, dt = args.dataset), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        elif args.model in ['SSI-DDI', 'SAGAN']:
            if args.dataset == 'drugbank':
                train_dataset = SSIDataset(self.data['train'], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                self.data_iter['train'] = SSILoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                if args.adversarial:
                    copy_triplets = ((int(len(self.data['train'])/len(self.data['valid_S1'])) + 1) * self.data['valid_S1'])[:int(len(self.data['train']))]
                    train_dataset_adv = SSIDataset(copy_triplets, self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter['train_adv'] = SSILoader(train_dataset_adv, batch_size=args.batch_size, shuffle=True)
                for j in self.split_not_train:
                    dts = SSIDataset(self.data[j], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter[j] = SSILoader(dts, batch_size=args.batch_size, shuffle=False)
            elif args.dataset == 'twosides':
                train_dataset = SSIDataset(self.data['train'], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                self.data_iter['train'] = SSILoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                if args.adversarial:
                    copy_triplets = ((int(len(self.data['train'])/len(self.data['valid_S1'])) + 1) * self.data['valid_S1'])[:int(len(self.data['train']))]
                    train_dataset_adv = SSIDataset(copy_triplets, self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter['train_adv'] = SSILoader(train_dataset_adv, batch_size=args.batch_size, shuffle=True)
                for j in self.split_not_train:
                    dts = SSIDataset(self.data[j], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter[j] = SSILoader(dts, batch_size=args.batch_size, shuffle=False)
        else:
            self.data_iter['train'] = self.get_data_loader(TrainDataset, 'train', args.batch_size)
            for j in self.split_not_train:
                self.data_iter[j] = self.get_data_loader(TestDataset, j, args.batch_size, shuffle = False)

    def get_atom_features(self, atom, mode='one_hot'): ### data process for SSI-DDI

        if mode == 'one_hot':
            atom_feature = torch.cat([
                one_of_k_encoding_unk(atom.GetSymbol(), self.AVAILABLE_ATOM_SYMBOLS),
                one_of_k_encoding_unk(atom.GetDegree(), self.AVAILABLE_ATOM_DEGREES),
                one_of_k_encoding_unk(atom.GetTotalNumHs(), self.AVAILABLE_ATOM_TOTAL_HS),
                one_of_k_encoding_unk(atom.GetImplicitValence(), self.AVAILABLE_ATOM_VALENCE),
                torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
            ])
        else:
            atom_feature = torch.cat([
                one_of_k_encoding_unk(atom.GetSymbol(), self.AVAILABLE_ATOM_SYMBOLS),
                torch.tensor([atom.GetDegree()]).float(),
                torch.tensor([atom.GetTotalNumHs()]).float(),
                torch.tensor([atom.GetImplicitValence()]).float(),
                torch.tensor([atom.GetIsAromatic()]).float()
            ])

        return atom_feature


    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        return  DataLoader(
            dataset_class(self.triples[split], self.args),
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = 10, ### set the default numworkers to 10
            collate_fn      = dataset_class.collate_fn,
            drop_last=True
        )

class TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		# self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele	= self.triples[idx]
		if 'sub_samp' in ele:
			triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		else:
			triple, label = torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label']).astype(int) 
		if self.p.dataset == 'drugbank': trp_label = self.get_label_ddi(label) 
		elif self.p.dataset == 'twosides': 
			label = label[0]
			trp_label = torch.FloatTensor(label)
        
		if self.p.model in ['MSTE']:
			if self.p.dataset == 'drugbank':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])
			elif self.p.dataset == 'twosides':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][2], ele['label'][0] , ele['label'][1]])
				trp_label = torch.LongTensor([ele['label'][2]])

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		if self.p.dataset == 'drugbank': triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		elif self.p.dataset == 'twosides': triple, label	= torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label'])[0]
		if self.p.dataset == 'drugbank': label		= self.get_label_ddi(label)
		elif self.p.dataset == 'twosides': label = torch.FloatTensor(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

### dataset for tiger

class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, dt = None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.dt = dt

    def read_drug_info(self, drug_id):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        if edge_index == 0:
            c_size = 1
            features = [[0 for j in range(67)]]
            edge_index = [[0, 0]]
            rel_index = [0]
            sp_edge_index = [[0, 0]]
            sp_value = [1]
            sp_rel = [1]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            #   y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                # y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        # labels = int(self.labels[idx])
        if self.dt == 'drugbank':
            labels = torch.LongTensor([self.labels[idx]])
        else:
            labels = torch.FloatTensor(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id)
        drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subrgraph, labels


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = torch.stack([data[4] for data in data_list]).squeeze(1)

    return batchA, batchB, batchC, batchD, batchE

### Dataset for SSI-DDI

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    
    return undirected_edge_list.T, features


class SSIDataset(Dataset):
    def __init__(self, tri_list, MOL_EDGE_LIST_FEAT_MTX, args, ratio=1.0,  neg_ent=1, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        ''' 
        # self.neg_ent = neg_ent
        self.tri_list = []
        # self.ratio = ratio
        self.MOL_EDGE_LIST_FEAT_MTX = MOL_EDGE_LIST_FEAT_MTX

        for h, t, r, *_ in tri_list:
            self.tri_list.append((h, t, r))

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)
    
    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []

        for h, t, r in batch:
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels)
        pos_tri = (pos_h_samples, pos_t_samples, pos_rels)

        return pos_tri

    def __create_graph_data(self, id):
        edge_index = self.MOL_EDGE_LIST_FEAT_MTX[id][0]
        features = self.MOL_EDGE_LIST_FEAT_MTX[id][1]

        return Data(x=features, edge_index=edge_index)

class SSILoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
