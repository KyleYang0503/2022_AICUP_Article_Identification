import torch
from torch.utils.data import Dataset
import random
import numpy as np
import ast


# for testing
class TestDataset(Dataset):
    def __init__(self, test_did, ref_dids, data, tokenizer):
        self.tok = tokenizer
        
        self.test_did = test_did
        self.did2text  = {}
        self.did2title = {}
        self.build_did2data(data)
        self.test_marker_token_id = self.tok.convert_tokens_to_ids('[unused0]')
        print(f'test_marker_token_id : {self.test_marker_token_id}')
        self.ref_marker_token_id = self.tok.convert_tokens_to_ids('[unused1]')
        print(f'ref_marker_token_id : {self.ref_marker_token_id}')
        # list of id
        self.ref_dids = ref_dids

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title

    def tensorsize(self, text):
            input_dict = self.tok(
                text,
                add_special_tokens=True,
                max_length=512,
                return_tensors='pt',
                pad_to_max_length=True,
                truncation='longest_first',
            )

            input_ids = input_dict['input_ids'][0]
            token_type_ids = input_dict['token_type_ids'][0]
            attention_mask = input_dict['attention_mask'][0]

            return (input_ids, attention_mask, token_type_ids)

    def tensorsize_colbert(self, text, marker):
        # assert marker in ['test', ref]
        text = '. ' + text

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        if marker == 'test':
            input_ids[1] = self.test_marker_token_id
        elif marker == 'ref':
            input_ids[1] = self.ref_marker_token_id

        return (input_ids, attention_mask, token_type_ids)

    def __getitem__(self, idx):
        ref_did = self.ref_dids[idx]

        test_text = self.did2title[self.test_did] + self.did2text[self.test_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]

        test_input_ids , test_attention_mask, test_token_type_ids = self.tensorsize_colbert(test_text, 'test')
        ref_input_ids , ref_attention_mask, ref_token_type_ids = self.tensorsize_colbert(ref_text, 'ref')

        return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask

            
    def __len__(self):
        return len(self.ref_dids)


        







# for train score model (ColBERT)
class MyDataset_triples(Dataset):
    def __init__(self, mode, data, tokenizer, hard_negative_nums=2, rand_negative_nums=10):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.hard_negative_nums = hard_negative_nums
        self.rand_negative_nums = rand_negative_nums



        self.tok = tokenizer
        self.unknown_token_id = self.tok.unk_token_id
        print(self.unknown_token_id)
        self.test_marker_token_id = self.tok.convert_tokens_to_ids('[unused1]')
        print(f'test_marker_token_id : {self.test_marker_token_id}')
        self.ref_marker_token_id = self.tok.convert_tokens_to_ids('[unused2]')
        print(f'ref_marker_token_id : {self.ref_marker_token_id}')

        # (test_did, pos_ref_did, neg_ref_did)
        self.train_triples = []
        
        self.did2text  = {}
        self.did2title = {}
        self.did2keywords = {}
        self.all_dids = set()
        self.build_did2data(data)

        if mode == 'train':
            self.build_train_triples(data)
        if mode == 'test':
            self.build_test_pairs(data)

    def build_did2data(self, data):

        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            keywords = ','.join(d['keyword_set'])
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.did2keywords[did] = keywords

            self.all_dids.add(did)



    def build_train_triples(self,data):
        import random
        for d in data:
            did = d['did']
            pos_dids = d['pos_dids']
            neg_dids = d['hard_neg_dids']
            if len(pos_dids) == 0:
                continue

            hard_negative_nums = len(pos_dids) * 2
            rand_negative_nums = len(pos_dids) * 3
            

            for pos_did in pos_dids:
                for neg_did in neg_dids[:hard_negative_nums]:
                    self.train_triples.append((did, pos_did, neg_did))

                random_negative_dids = random.sample(self.all_dids, 2 * rand_negative_nums)
                c = 0
                for neg_did in random_negative_dids:
                    if neg_did == did or neg_did in pos_dids:
                        continue
                    self.train_triples.append((did, pos_did, neg_did))
                    c += 1
                    if c >= rand_negative_nums:
                        break

            
            # pos_dids_set = set(pos_dids)
            # for pos_did in pos_dids:
            #     if pos_did == '887':
            #         continue
            #     for neg_did in self.all_dids:
            #         if neg_did in pos_dids_set or neg_did == did:
            #             continue
            #         self.train_triples.append((did, pos_did, neg_did))

    def tensorsize(self, text, keywords, marker):
        assert marker in ['test', 'ref']
        text = '. ' + text

        input_dict = self.tok(
            keywords + text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        if marker == 'test':
            input_ids[1] = self.test_marker_token_id
        elif marker == 'ref':
            input_ids[1] = self.ref_marker_token_id
        return (input_ids, attention_mask, token_type_ids)


    def tensorsize_nomarker(self, text, keywords):
        # assert marker in ['test', ref]
        # text = keywords + ',' + text
        # print(text)
        # input()

        input_dict = self.tok(
            text, keywords,
            add_special_tokens=True,
            max_length=256,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)

    def combine_ref(self, pos_ref, neg_ref):
        pos_ref_input_ids, pos_ref_attention_mask, pos_ref_token_type_ids = pos_ref
        neg_ref_input_ids, neg_ref_attention_mask, neg_ref_token_type_ids = neg_ref


        input_ids = torch.cat((pos_ref_input_ids.unsqueeze(0), neg_ref_input_ids.unsqueeze(0)))
        attention_mask = torch.cat((pos_ref_attention_mask.unsqueeze(0), neg_ref_attention_mask.unsqueeze(0)))
        token_type_ids = torch.cat((pos_ref_token_type_ids.unsqueeze(0), neg_ref_token_type_ids.unsqueeze(0)))
 
        return (input_ids, attention_mask, token_type_ids)

    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , pos_ref_did, neg_ref_did = self.train_triples[idx]


        # test_text = self.did2title[test_did] + self.did2text[test_did]
        # pos_ref_text = self.did2title[pos_ref_did] + self.did2text[pos_ref_did]
        # neg_ref_text = self.did2title[neg_ref_did] + self.did2text[neg_ref_did]

        # test_text = self.did2keywords[test_did]
        # pos_ref_text = self.did2keywords[pos_ref_did]
        # neg_ref_text = self.did2keywords[neg_ref_did]

        # test_text = self.did2title[test_did] + self.did2keywords[test_did]

        test_text = self.did2title[test_did] + self.did2text[test_did]
        test_keywords = self.did2keywords[test_did]

        pos_ref_text = self.did2title[pos_ref_did] + self.did2text[pos_ref_did]
        pos_ref_keywords = self.did2keywords[pos_ref_did]

        neg_ref_text = self.did2title[neg_ref_did] + self.did2text[neg_ref_did]
        neg_ref_keywords = self.did2keywords[neg_ref_did]

        test = self.tensorsize(test_text, test_keywords, marker='test')
        pos_ref = self.tensorsize(pos_ref_text, pos_ref_keywords, marker='ref')
        neg_ref = self.tensorsize(neg_ref_text, neg_ref_keywords, marker='ref')

        ref = self.combine_ref(pos_ref, neg_ref)

        test_input_ids , test_attention_mask, test_token_type_ids = test
        ref_input_ids , ref_attention_mask, ref_token_type_ids = ref
        if self.mode == 'train':
            # return test_input_ids , test_attention_mask, ref_input_ids , ref_attention_mask
            return test, ref

            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_triples)
        # if self.mode == 'test':
        #     return len(self.test_pairs)

class BinaryDataset(Dataset):
    def __init__(self, mode, data, tokenizer,  train_pairs_path):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.did2text  = {}
        self.did2title = {}
        self.all_dids = set()
        self.tok = tokenizer

        self.build_did2data(data)
        import pickle
        with open(train_pairs_path, 'rb') as handle:
            self.train_pairs =  pickle.load(handle)


    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.all_dids.add(did)

    def tensorsize(self, text):

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)
         
    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , ref_did, label = self.train_pairs[idx]
            label = torch.tensor(label)

        if self.mode == 'test':
            test_did , ref_did = self.test_pairs[idx]


        test_text = self.did2title[test_did] + self.did2text[test_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]

        test_input_ids, test_attention_mask, test_token_type_ids = self.tensorsize(test_text)

        ref_input_ids, ref_attention_mask, ref_token_type_ids = self.tensorsize(ref_text)
        
        if self.mode == 'train':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, label
        if self.mode == 'test':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, test_did, ref_did 




    def __len__(self):
        if self.mode == 'train':
            return len(self.train_pairs)
        






# for train binary
class MyDataset(Dataset):
    def __init__(self, mode, data, tokenizer, negative_nums=2, add_marker=False):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.negative_nums = negative_nums
        self.tok = tokenizer
        # list of list , elemets be like (test_did, reference_did, label)
        self.train_pairs = []
        self.add_marker = add_marker
        self.did2keywords = {}
        self.test_pairs = []
        self.did2text  = {}
        self.did2title = {}
        self.all_dids = set()
        self.build_did2data(data)

        self.test_marker_token_id = self.tok.convert_tokens_to_ids('[unused1]')
        print(f'test_marker_token_id : {self.test_marker_token_id}')
        self.ref_marker_token_id = self.tok.convert_tokens_to_ids('[unused2]')
        print(f'ref_marker_token_id : {self.ref_marker_token_id}')

        if mode == 'train':
            self.build_train_pairs(data)
        if mode == 'test':
            self.build_test_pairs(data)

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            keywords = ','.join(d['keyword_set'])
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.did2keywords[did] = keywords
            self.all_dids.add(did)


    def build_train_pairs(self, data):
        pos_count = 0
        neg_count = 0
        for d in data:
            did = d['did']
            pos_dids = d['pos_dids']
            neg_dids = d['hard_neg_dids']

            pos_dids = set(pos_dids)

            if len(pos_dids) == 0:
                continue
            for pos_did in pos_dids:
                self.train_pairs.append((did, pos_did, 1))
                pos_count += 1
            
            for neg_did in neg_dids[:self.negative_nums]:
                self.train_pairs.append((did, neg_did, 0))
                neg_count += 1
            # for neg_did in self.all_dids:
            #     if neg_did != did and neg_did not in pos_dids:
            #         self.train_pairs.append((did, neg_did, 0))
            #         neg_count += 1
        print(f'training set postive rate == {pos_count/(pos_count + neg_count) :.3f}, pos={pos_count} neg={neg_count}')

    def build_test_pairs(self, data):
        all_dids = sorted([int(k) for k in self.did2text.keys()])
        for test_did in all_dids:
            for ref_did in all_dids:
                if ref_did != test_did:
                    self.test_pairs.append((str(test_did), str(ref_did)))

    def tensorsize_cross_encoder(self, test_text, ref_text):

        test_text = test_text[:250]
        ref_text = ref_text[:250]

        input_dict = self.tok(
            test_text, ref_text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)

    def tensorsize_colbert(self, text, keywords, marker):
        assert marker in ['test', 'ref']
        text = '. ' + text

        input_dict = self.tok(
            keywords, text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        if marker == 'test':
            input_ids[1] = self.test_marker_token_id
        elif marker == 'ref':
            input_ids[1] = self.ref_marker_token_id
        return (input_ids, attention_mask, token_type_ids)

    def tensorsize(self, text):

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=200,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)
         
    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , ref_did, label = self.train_pairs[idx]
            label = torch.tensor(label)

        if self.mode == 'test':
            test_did , ref_did = self.test_pairs[idx]


        # test_text = self.did2title[test_did] + self.did2keywords[test_did]
        # ref_text = self.did2title[ref_did] + self.did2keywords[ref_did]
        test_text = self.did2title[test_did] + self.did2text[test_did]
        # test_text = self.did2title[test_did] 
        test_keywords = self.did2keywords[test_did]

        # ref_text = self.did2title[ref_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]
        ref_keywords = self.did2keywords[ref_did]

        test = self.tensorsize_colbert(test_text, test_keywords, marker='test')
        ref = self.tensorsize_colbert(ref_text, ref_keywords, marker='ref')


        # if self.add_marker:
        #     test_input_ids, test_attention_mask, test_token_type_ids = self.tensorsize_colbert(test_text, 'test')
        #     ref_input_ids, ref_attention_mask, ref_token_type_ids = self.tensorsize_colbert(ref_text, 'ref')
        # else:
        #     test_input_ids, test_attention_mask, test_token_type_ids = self.tensorsize(test_text)
        #     ref_input_ids, ref_attention_mask, ref_token_type_ids = self.tensorsize(ref_text)

            
        
        if self.mode == 'train':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, label
        if self.mode == 'test':
            return test, ref, (test_did, ref_did)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_pairs)
        if self.mode == 'test':
            return len(self.test_pairs)



# for cross encoder 
class CrossEncoderDataset(Dataset):
    def __init__(self, mode, data, tokenizer, hard_negative_nums=2, rand_negative_nums=2, train_pairs_path=None, test_pairs_path=None):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.hard_negative_nums = hard_negative_nums
        self.rand_negative_nums = rand_negative_nums
        self.tok = tokenizer
        # list of list , elemets be like (test_did, reference_did, label)
        self.train_pairs = []
        
        self.test_pairs = []
        self.did2text  = {}
        self.did2title = {}
        self.did2keywords = {}
        self.all_dids = set()
        self.build_did2data(data)

        if train_pairs_path:
            import pickle
            with open(train_pairs_path, 'rb') as handle:
                self.train_pairs =  pickle.load(handle)
                print('using train pairs from pickle file!!')
        else:
            if mode == 'train':
                self.build_train_pairs(data)

        if mode == 'test':
            self.build_test_pairs(data)

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            keywords = ','.join(d['keyword_set'])
            did = d['did']
            self.all_dids.add(did)
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.did2keywords[did] = keywords


    def build_train_pairs(self, data):
        pos_count = 0
        neg_count = 0
        for d in data:
            did = d['did']
            pos_dids = d['pos_dids']
            neg_dids = d['hard_neg_dids']

            pos_dids = set(pos_dids)

            len_of_pos_dids = len(pos_dids)
            if len(pos_dids) == 0:
                len_of_pos_dids = 1

                random_negative_dids = random.sample(self.all_dids, 5)
                for neg_did in random_negative_dids:
                    if neg_did in self.all_dids:
                        if neg_did == did:
                            continue
                        self.train_pairs.append((did, neg_did, 0))
                        neg_count += 1
                # hard_negative_nums = 5
                # for neg_did in neg_dids[:hard_negative_nums]:
                #     if neg_did in self.all_dids:
                #         self.train_pairs.append((did, neg_did, 0))
                #         neg_count += 1
                continue
            
            for i in range(1):
                for pos_did in pos_dids:
                    if pos_did in self.all_dids:
                        self.train_pairs.append((did, pos_did, 1))
                        pos_count += 1

            hard_negative_nums = len_of_pos_dids * 10
            for neg_did in neg_dids[:hard_negative_nums]:
                if neg_did in self.all_dids:
                    self.train_pairs.append((did, neg_did, 0))
                    neg_count += 1

            rand_negative_nums = len_of_pos_dids * 20
            if rand_negative_nums == 0:
                continue
            try:
                random_negative_dids = random.sample(self.all_dids, max(len(self.all_dids), 2 * rand_negative_nums))
            except:
                random_negative_dids = self.all_dids
            c = 0
            for neg_did in random_negative_dids:
                if neg_did in self.all_dids:
                    if neg_did == did or neg_did in pos_dids:
                        continue
                    self.train_pairs.append((did, neg_did, 0))
                    neg_count += 1
                    c += 1
                    if c >= self.rand_negative_nums:
                        break
            # for neg_did in self.all_dids:
            #     if neg_did != did and neg_did not in pos_dids:
            #         self.train_pairs.append((did, neg_did, 0))
            #         neg_count += 1

        print(f'training set postive rate == {pos_count/(pos_count + neg_count) :.3f}, pos={pos_count} neg={neg_count}')

    def build_test_pairs(self, data):
        all_dids = sorted([int(k) for k in self.did2text.keys()])
        for test_did in all_dids:
            for ref_did in all_dids:
                if ref_did != test_did:
                    self.test_pairs.append((str(test_did), str(ref_did)))

    def tokenize(self, keywords, title, text):
        keywords_tokens = self.tok.tokenize(keywords)
        title_tokens = self.tok.tokenize(title)
        text_tokens = self.tok.tokenize(text)
        
        # tokens = ['[unused1]'] + keywords_tokens + ['[unused1]'] + title_tokens + text_tokens
        # tokens = keywords_tokens + title_tokens + text_tokens
        tokens = keywords_tokens + title_tokens + text_tokens 

        # print(tokens)
        return tokens
        # ids = self.tok.convert_tokens_to_ids(tokens)
        # print(ids)
        # input()


    def tensorsize(self, test, ref):
        test_keywords, test_title, test_text = test
        test_tokens = self.tokenize(test_keywords, test_title, test_text)

        ref_keywords, ref_title, ref_text = ref
        ref_tokens = self.tokenize(ref_keywords, ref_title, ref_text)

        
        tokens = ['[CLS]'] + test_tokens[:255] + ['[SEP]'] + ref_tokens[:254] + ['[SEP]']

        token_type_ids = [0] * (1 + len(test_tokens[:255]) + 1) + [1] * (len(ref_tokens[:254]) + 1)
        input_ids = self.tok.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        assert len(token_type_ids) == len(input_ids) == len(attention_mask) 
        assert len(token_type_ids) <= 512

        if len(token_type_ids) < 512:
            token_type_ids = token_type_ids  + [0] * (512 - len(token_type_ids))
            input_ids = input_ids + [0] * (512 - len(input_ids))
            attention_mask = attention_mask + [0] * (512 - len(attention_mask))

        assert len(token_type_ids) == 512

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        return (input_ids, attention_mask, token_type_ids)
        


    # def tensorsize(self, test_text, ref_text):

    #     test_text = test_text[:250]
    #     ref_text = ref_text[:250]

    #     input_dict = self.tok(
    #         test_text, ref_text,
    #         add_special_tokens=True,
    #         max_length=512,
    #         return_tensors='pt',
    #         pad_to_max_length=True,
    #         padding='max_length',
    #         truncation='longest_first',
    #     )

    #     input_ids = input_dict['input_ids'][0]
    #     token_type_ids = input_dict['token_type_ids'][0]
    #     attention_mask = input_dict['attention_mask'][0]

    #     return (input_ids, attention_mask, token_type_ids)
         
    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , ref_did, label = self.train_pairs[idx]
            label = torch.tensor(label)

        if self.mode == 'test':
            test_did , ref_did = self.test_pairs[idx]


        test_keywords = self.did2keywords[test_did] 
        test_title = self.did2title[test_did]
        test_text = self.did2text[test_did]

        ref_keywords = self.did2keywords[ref_did]
        ref_title = self.did2title[ref_did]
        ref_text = self.did2text[ref_did]

        test = (test_keywords, test_title, test_text)
        ref = (ref_keywords, ref_title, ref_text)

        
        # ref_text =  '[unused1] ' + self.did2keywords[ref_did] + ' [unused1] ' + self.did2title[ref_did] + self.did2text[ref_did]
        # test_text = '[unused1] ' + self.did2keywords[test_did] + ' [unused1] ' + self.did2title[test_did] + self.did2text[test_did]

        # ref_text =  self.did2title[ref_did] + self.did2text[ref_did]
        # test_text = self.did2title[test_did] + self.did2text[test_did]
        # test_keywords = self.did2keywords[test_did]

        # pos_ref_text = self.did2title[pos_ref_did] + self.did2text[pos_ref_did]
        # pos_ref_keywords = self.did2keywords[pos_ref_did]

        # test_text = self.did2title[test_did] + self.did2keywords[test_did]
        # ref_text = self.did2title[ref_did] + self.did2keywords[ref_did]


        # input_ids, attention_mask, token_type_ids = self.tensorsize(test_text, ref_text)
        input_ids, attention_mask, token_type_ids = self.tensorsize(test, ref)


        
        if self.mode == 'train':
            return input_ids, attention_mask, token_type_ids, label, (test_did, ref_did)
        if self.mode == 'test':
            return (input_ids, attention_mask, token_type_ids), (test_did, ref_did)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_pairs)
        if self.mode == 'test':
            return len(self.test_pairs)