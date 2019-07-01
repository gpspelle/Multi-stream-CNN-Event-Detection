python3 train.py -actions cross-train -streams temporal spatial -class Falls NotFalls -w0 1 -ep 2 -batch_norm True -fold_norm 2 -kfold video -mini_batch 1024 -id URFD -nsplits 5 -lr 0.0001
