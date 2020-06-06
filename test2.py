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
print(type(testloader))
# for elem in testloader :
# 	print(elem)
# 	print(type(elem))
# 	print(len(elem))
# 	print(elem[1])
# 	sys.exit()
# pap -----------
# here for both (which it is normal because of the lines above) I get the
# same output as for reading the dictionary with test_read_comernet_dics_save_data_tgt.py
#pprint( src_vocab )
# pprint( tgt_vocab )
inverse_src_vocab = {}
for k,v in src_vocab.items():
       inverse_src_vocab[ v ] = k
#----------       
inverse_tgt_vocab = {}
for k,v in tgt_vocab.items():
       inverse_tgt_vocab[ v ] = k
#----------       

d = 0
for src1, src1_len, src2,src2_len, src3, src3_len, tgt, tgt_len,tgtv, tgtv_len,tgtpv, tgtpv_len in testloader:
    # pap ----------
    print( '______________\n\n' )
    print( 'dialog d={0}=================='.format( d ))
    pprint( src1_len )
    print( '______________\nUSER\n' )
    for i, s in enumerate( src1 ):
        # print(  s.dim() )
        # print( 's size  is {0}'.format( s.size()))
        turn_nbr, max_token = s.size()
        for turn_idx in range( 0, turn_nbr ):
            msg = ''
            for tok_idx in range( 0, max_token ):
                # print( s[ turn_idx ][ tok_idx ] )
                # print( s[ turn_idx ][ tok_idx ].item() )
                msg += '{0} '.format( inverse_src_vocab[ s[ turn_idx ][ tok_idx ].item() ] )
            # print( msg )
    d += 1
    print( '______________\n system\n' )
    for i, s in enumerate( src2 ):
        # print(  s.dim() )
        # print( 's size  is {0}'.format( s.size()))
        turn_nbr, max_token = s.size()
        for turn_idx in range( 0, turn_nbr ):
            msg = ''
            for tok_idx in range( 0, max_token ):
                print( s[ turn_idx ][ tok_idx ] )
                # print( s[ turn_idx ][ tok_idx ].item() )
                msg += '{0} '.format( inverse_src_vocab[ s[ turn_idx ][ tok_idx ].item() ] )
            # print( msg )
        d += 1
    print( '______________\ndst\n' )
    for i, s in enumerate( src3 ):
        # print(  s.dim() )
        print( 's size  is {0}'.format( s.size()))
        turn_nbr, max_token = s.size()
        for turn_idx in range( 0, turn_nbr ):
            msg = ''
            for tok_idx in range( 0, max_token ):
                # print( s[ turn_idx ][ tok_idx ] )
                # print( s[ turn_idx ][ tok_idx ].item() )
                msg += '{0} '.format( inverse_src_vocab[ s[ turn_idx ][ tok_idx ].item() ] )
            print( msg )
        d += 1   
    #-----------    
    print(f'{tgt}, len = {len(tgt)}, type = {type(tgt)}')
    for i, s in enumerate(tgt):
        # print(  s.dim() )
        print( 's size  is {0}'.format( s.size()))
        turn_nbr, max_token = s.size()
        for turn_idx in range( 0, turn_nbr ):
            msg = ''
            for tok_idx in range( 0, max_token ):
                print( s[ turn_idx ][ tok_idx ] )
                # print( s[ turn_idx ][ tok_idx ].item() )
                msg += '{0} '.format( inverse_src_vocab[ s[ turn_idx ][ tok_idx ].item() ] )
            print( msg )
        d += 1

    print( '______________\n system\n' )
    # print(tgtv)

    for i, s in enumerate(tgtv):
        for j,v in enumerate(s):
            print( 'v size  is {0}'.format( v.size()))
            turn_nbr, max_token = v.size()
            for turn_idx in range( 0, turn_nbr ):
                msg = ''
                for tok_idx in range( 0, max_token ):
                    # print( s[ turn_idx ][ tok_idx ] )
                    # print( s[ turn_idx ][ tok_idx ].item() )
                    msg += '{0} '.format( inverse_src_vocab[ v[ turn_idx ][ tok_idx ].item() ] )
                print( msg )
        d += 1              
        
    for i, s in enumerate(tgtpv):
        for j,v in enumerate(s):
            for k, vv in enumerate(v):
                print( 'vv size  is {0}'.format( vv.size()))
                turn_nbr, max_token = vv.size()
                for turn_idx in range( 0, turn_nbr ):
                    msg = ''
                    for tok_idx in range( 0, max_token ):
                        # print( s[ turn_idx ][ tok_idx ] )
                        # print( s[ turn_idx ][ tok_idx ].item() )
                        msg += '{0} '.format( inverse_src_vocab[ vv[ turn_idx ][ tok_idx ].item() ] )
        print( msg )
        d += 1
    sys.exit()              
    #-----------    
