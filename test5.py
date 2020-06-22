import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import models
import dataloader
import utils 
import dict as dic

import os, sys
import argparse
import time
import math
import json
import collections

from collections import defaultdict
#config
parser = argparse.ArgumentParser(description='predict.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='data/log/norml_mwNestedNOND128NOsaA1FNN/checkpoint.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='predict', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-memory', action='store_true',
                    help="memory efficiency")
parser.add_argument('-beam_size', type=int, default=1,
                    help="beam search size")

opt = parser.parse_args([])
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

# pap ---------
from pprint import pprint
#--------

#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['test']  # other possible values 'train' and 'valid'
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['src']
testloader = dataloader.get_loader(testset, batch_size=1, shuffle=False, num_workers=2)

# pap -----------
# here for both (which it is normal because of the lines above) I get the
# same output as for reading the dictionary with test_read_comernet_dics_save_data_tgt.py
#pprint( src_vocab )
#pprint( tgt_vocab )
inverse_src_vocab = {}
for k,v in src_vocab.items():
       inverse_src_vocab[ v ] = k
#----------     

print("\n\n*********** Ajouts de Léon *********\n\n")  
# ------------------- ajouts Léon 
from pytorch_pretrained_bert import BertModel
from convert_mw import bert,tokenizer,bert_type

# fonction qu'on utilise dès qu'on veut convertir les tenseurs en langage naturel
def generate_tensors(turns, max_tok, inverse_src_vocab, value, msg, usr_turn= "", label = False):
    for turn_idx in range( 0, turns ):
        if label : 
            elem = value
        else : 
            elem =  value[ turn_idx ]
        for tkj in range( 0, max_tok ):

            a_user_tok = inverse_src_vocab[ elem[ tkj ].item() ]
            msg += '{0} '.format( a_user_tok )
            usr_turn += '{0} '.format( a_user_tok )
    return msg

if opt.restore:
    print('loading checkpoint...\n')    
    checkpoints = torch.load(opt.restore,map_location=torch.device('cpu'))
pretrain_embed={}
pretrain_embed['slot'] = torch.load('emb_tgt_mw.pt')
print('building model...\n')
bmodel = BertModel.from_pretrained(bert_type)
bmodel.eval()
use_cuda = False
if use_cuda:
    bmodel.to('cuda')
model = getattr(models, opt.model)(config, src_vocab, tgt_vocab, use_cuda,bmodel,
                       pretrain=pretrain_embed, score_fn=opt.score) 
 

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]


model.eval()
# print(model)
# print(type(model))
# sys.exit()
preds = []
labels = []
joint_preds=[]
joint_labels=[]
joint_allps=[]
joint_alls=[]
    
reference, candidate, source, alignments = [], [], [], []
for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in testloader:
    samples,ssamples,vsamples,_ = model.sample(src1, src1_len, src2,src2_len, src3, src3_len,tgtpv, tgtpv_len)
    user_turn = ''
    msg = ""
    domain_dic = {}
    for i, s in enumerate( samples ):
        turn_nbr, max_token = s.size()
#def generate_tensors(turns, max_tok, inverse_src_vocab, s, msg ):

        # print(generate_tensors(turn_nbr, max_token, inverse_src_vocab, s, "\t----- domain:", user_turn))
            # print( '\t{0}'.format( msg ))
        user_turns = ""
        for j, ss in enumerate( ssamples[i] ):
            turn_nbrs, max_tokens = ss.size()
            # print(generate_tensors(turn_nbrs, max_tokens, inverse_src_vocab, ss, "\t\t----- slots:", user_turns))
            user_turnv = ''
            for k, sv in enumerate(vsamples[i][j]) : 
                turn_nbrv, max_tokenv = sv.size()
                # print(generate_tensors(turn_nbrv, max_tokenv, inverse_src_vocab, sv, "\t\t\t----- values:", user_turnv))
    for x,xv,xvv,y,yv,yvv in zip(samples,ssamples,vsamples, tgt[0],tgtv[0],tgtpv[0]):
        x=x.data.cpu()
        y=y.data.cpu()
        xt=x[0][:-1].tolist()

        # print(f"xt = {xt}")
        # print(f"len xt = {len(xt)}")
        # print(f"type xt = {type(xt)}")
        print(f"xv = {xv}")
        print(f"len xv = {len(xv)}")
        print(f"type xv = {type(xv)}")
        print(f"xvv = {xvv}")
        print(f"len xvv = {len(xvv)}")
        print(f"type xvv = {type(xvv)}")
        # print(f"y = {y}")
        # print(f"len y = {len(y)}")
        # print(f"type y = {type(y)}")
        print(f"yv = {yv}")
        print(f"len yv = {len(yv)}")
        print(f"type yv = {type(yv)}")
        print(f"yvv = {yvv}")
        print(f"len yvv = {len(yvv)}")
        print(f"type yvv = {type(yvv)}")
        turn, mx = x.size()
        print(generate_tensors(turn, mx, inverse_src_vocab, x, '--- X domain = '))
        print(generate_tensors(1, len(y), inverse_src_vocab, y, '--- Y domain = ', label = True))
        # msg = '--- Y domain = '
        # for tkj in range( 0, len(y) ):
        #     a_user_tok = inverse_src_vocab[ y[ tkj ].item() ]
        #     msg += '{0} '.format( a_user_tok )
        # print(msg)

        # print(generate_tensors(1, len(y), inverse_src_vocab, y,))
        # print(generate_tensors(turn, mx, inverse_src_vocab, y, '--- Y domain = '))
        for i, s in enumerate(xv) : 
            turn_xv, mxv = s.size()
            print(generate_tensors(turn_xv, mxv, inverse_src_vocab, s, '\t--- XV slots = '))
        for i, s in enumerate(yv) :
            print(generate_tensors(1, len(s), inverse_src_vocab, s, '\t--- YV slots = ', label = True))
        for i, s in enumerate(xvv):

            for j, sprime in enumerate(s) : 
                turn_xvv, mxvv = sprime.size()
                print(generate_tensors(turn_xvv, mxvv, inverse_src_vocab, sprime, '\t\t--- XVV values = '))
        for i, s in enumerate(yvv):
            print(s)
            print(s.size())
            turn_yvv, my = s.size()
            print(generate_tensors(turn_yvv, my, inverse_src_vocab, s, '\t\t--- YVV values = '))

        vvt=defaultdict(dict)
        svt={}
        for i,k in enumerate(xvv[:-1]):
            slots=xv[:-1][i][0][:-1].tolist()
            if len(slots)!=0:
                for ji,j in enumerate(k[:-1]):
                    jt=j[0][:-1].tolist()
                    svt[xt[i]]=set(slots)
#                       print("jt:",jt)
                    if len(jt)!=0:
                        vvt[xt[i]][slots[ji]]=jt
            preds.append(set(xt))  
            joint_preds.append(svt)
            joint_allps.append(vvt)
            #print(joint_preds)
            label = []
            for l in y[1:].tolist():
                if l == dic.EOS:
                    break
                label.append(l)

            labels.append(set(label)) 
            print(preds)
            print(labels)

            joint_label = {}
            joint_all=defaultdict(dict)
            for i,j in enumerate(yvv[1:].tolist()):
                slots=yv[1:].tolist()[i]
                if sum(slots[1:])==0:
                    break
                else:
                    s=[]
                    for l in slots[1:]:
                        if l == dic.EOS:
                            break
                        s.append(l)
                    if len(s)!=0:
                        joint_label[label[i]]=set(s)   
                        for ki,k in enumerate(j[1:]):
                            if sum(k[1:])==0:
                                break
                            else:
                                v=[]
                                for l in k[1:]:
                                    if l == dic.EOS:
                                        break
                                    v.append(l)
                                if len(v)!=0:
                                    joint_all[label[i]][s[ki]]=v
            pprint(joint_label)
            pprint(joint_preds)
            sys.exit()
            joint_labels.append(joint_label)
            joint_alls.append(joint_all) 

    sys.exit()

    print(f"predict dict = \n{preds}")
    print(f"joint predict dict = \n{joint_preds}")
    print(f"joint all predict dict = \n{joint_allps}")
    print(f" label = \n{label}")
    print(f" joint label = \n{joint_labels}")
    print(f" joint all label = \n{joint_alls}")

    
print("\n\n*********** Ajouts de Léon fin *********\n\n")
# ------------------- ajouts Léon 
zzz = 0
for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in testloader:
    # pap ----------
    print( 'zzz={0}==================================='.format( zzz ))
    pprint( src1_len )

    system_turn = ''
    for i, s in enumerate( src2 ):
       # print( '\t\ttype( s )= {0}'.format( type( s ))) 
       # print( '\t\tdimension = {0}'.format( s.dim() ))
       # print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_token = s.size()
       for turn_idx in range( 0, turn_nbr ):
              msg = '----- system:'
              for tkj in range( 0, max_token ):
                     # print( s[ turn_idx ][ tkj ] )
                     # print( s[ turn_idx ][ tkj ].item() )
                     a_system_tok = inverse_src_vocab[ s[ turn_idx ][ tkj ].item() ]
                     msg += '{0} '.format(  a_system_tok )
                     system_turn += '{0} '.format(  a_system_tok )
                     
    print( '\t{0}'.format( msg ))

    user_turn = ''
    for i, s in enumerate( src1 ):
       # print( '\t\ttype( s )= {0}'.format( type( s ))) 
       # print( '\t\tdimension = {0}'.format( s.dim() ))
       # print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_token = s.size()
       for turn_idx in range( 0, turn_nbr ):
              msg = '----- user:' 
              for tkj in range( 0, max_token ):
                     a_user_tok = inverse_src_vocab[ s[ turn_idx ][ tkj ].item() ]
                     msg += '{0} '.format( a_user_tok )
                     user_turn += '{0} '.format( a_user_tok )
    print( '\t{0}'.format( msg ))

    previous_state = ''
    for i, s in enumerate( src3 ):
       # print( '\t\ttype( s )= {0}'.format( type( s ))) 
       # print( '\t\tdimension = {0}'.format( s.dim() ))
       # print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_token = s.size()
       for turn_idx in range( 0, turn_nbr ):
              msg = 'PREVIOUS_STATE:'    
              for tkj in range( 0, max_token ):
                      # print( s[ turn_idx ][ tkj ] )
                      # print( s[ turn_idx ][ tkj ].item() )
                      a_previous_state_tok = inverse_src_vocab[ s[ turn_idx ][ tkj ].item() ]
                      msg += '{0} '.format( a_previous_state_tok )
                      previous_state += '{0} '.format( a_previous_state_tok )
       print( '\t{0}'.format( msg ))

    domain_dic = {}
    for i, s in enumerate( tgt ):
       print( '\t\ttype( s )= {0}'.format( type( s ))) 
       print( '\t\tdimension = {0}'.format( s.dim() ))
       print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_domain = s.size()
       for turn_idx in range( 0, turn_nbr ):
               msg = 'DOMAINS:\n'
               for domj in range( 0, max_domain ):
                      # print( s[ turn_idx ][ domj ] )
                      # print( s[ turn_idx ][ domj ].item() )
                      a_dom = inverse_src_vocab[ s[ turn_idx ][ domj ].item() ]
                      msg += "\t\t'{0}'\n".format( a_dom )
                      domain_dic[ domj ] = { 'name' : a_dom, 'slots' : {} }
       print( '\t{0}'.format( msg ))

    for i, s in enumerate( tgtv ):
       print( '\t\ttype( s )= {0}'.format( type( s ))) 
       print( '\t\tdimension = {0}'.format( s.dim()))
       print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_domain, max_slot = s.size()
       for turn_idx in range( 0, turn_nbr ):
              msg = 'SLOTS:\n\t\t'
              for domj in range( 0, max_domain ):
                slot_nbr = 0     
                for sli in range( 0, max_slot ):
                     a_slot = inverse_src_vocab[ s[ turn_idx ][ domj ][ sli ].item() ]
                     msg += " {0} ".format( a_slot )
                     domain_dic[ domj ][ 'slots' ][ sli ] = { 'name' : a_slot, 'value' : None }
                msg += '\n\t\t'
       print( '\t{0}'.format( msg ))
       
      
    for i, s in enumerate( tgtpv ):
       print( '\t\ttype( s )= {0}'.format( type( s ))) 
       print( '\t\tdimension = {0}'.format( s.dim() ))
       print( '\t\ts size  is {0}'.format( s.size()))
       turn_nbr, max_domain, max_slot, max_value = s.size()
       for turn_idx in range( 0, turn_nbr ):
              msg = 'VALUES:\n\t\t\t'
              for domj in range( 0, max_domain ):
                for sli in range( 0, max_slot ):
                     a_value = ''
                     for vj in range( 0, max_value ):
                            a_value_tok = inverse_src_vocab[ s[ turn_idx ][ domj ][ sli ][ vj ].item() ]
                            msg += " {0} ".format( a_value_tok )
                            a_value += a_value_tok
                     msg += '\n\t\t\t'
                     domain_dic[ domj ][ 'slots' ][ sli ][ 'value' ] = a_value
       print( '\t{0}'.format( msg ))

    print( '===== ANNOTAION DOMAIN dictionary ======' )
    #pprint( domain_dic )
    print( 'DEBUG' )
    pprint( domain_dic )
    zzz += 1
    if( zzz > 100 ):
           print( 'stopping before regular end of the loop' )
           break
    #-----------    
