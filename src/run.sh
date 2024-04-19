#!/bin/sh -x

# base model MF
python main.py --model_name MF --emb_size 32 --lr 5e-4 --l2 1e-7 --dataset "MMTD"
python main.py --model_name MF_SLRC --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset "MMTD"
python main.py --model_name MF_ReCODE --emb_size 32 --hidden_size 64 --steps 77 --method 'euler' \
    --lr 5e-4 --l2 1e-6 --dataset "MMTD"

# base model NCF
python main.py --model_name NCF --emb_size 32 --layers '[64]' --dropout 0.2 --lr 1e-4 --l2 1e-7 --dataset "MMTD"
python main.py --model_name NCF_SLRC --emb_size 32 --layers '[64]' --dropout 0.2 --lr 5e-4 --l2 1e-7 --dataset "MMTD"
python main.py --model_name NCF_ReCODE --emb_size 32 --hidden_size 64 --steps 77 --method 'euler' \
    --layers '[64]' --dropout 0.2 --lr 1e-4 --l2 1e-6 --dataset "MMTD"

# base model GRU4Rec
python main.py --model_name GRU4Rec --emb_size 32 --hidden_size 64 --history_max 20 --lr 1e-3 --l2 1e-5 --dataset "MMTD"
python main.py --model_name GRU4Rec_SLRC --emb_size 32 --hidden_size 64 --history_max 20 --lr 1e-3 --l2 1e-6 --dataset "MMTD"
python main.py --model_name GRU4Rec_ReCODE --emb_size 32 --hidden_size 64 --steps 77 --method 'euler' \
    --history_max 20 --lr 1e-4 --l2 1e-6 --dataset "MMTD"

# base model SASRec
python main.py --model_name SASRec --emb_size 32 --num_layers 1 --num_heads 1 --history_max 20 --lr 1e-4 --l2 1e-5 --dataset "MMTD"
python main.py --model_name SASRec_SLRC --emb_size 32 --num_layers 1 --num_heads 1 --history_max 20 --lr 1e-4 --l2 1e-5 --dataset "MMTD"
python main.py --model_name SASRec_ReCODE --emb_size 32 --num_layers 1 --num_heads 1 --hidden_size 64 --steps 77 --method 'euler' \
    --history_max 20 --lr 1e-4 --l2 1e-5 --dataset "MMTD"