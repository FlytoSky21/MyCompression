## train
CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d /home/adminroot/taofei/dataset/vimeo_septuplet 
--cuda  --epochs 100   --save_path /home/adminroot/taofei/DCC2023fuxian/bn16


## test
python eval.py --checkpoint [path of the pretrained checkpoint] 
--data [path of testing dataset] --cuda

python eval.py --checkpoint /home/adminroot/taofei/DCC2023fuxian/result_fk/0.0067checkpoint_best.pth.tar
--data /home/adminroot/taofei/dataset/Kodak24 --cuda


cd /taofei_9F/imgCompression/MyCompression && /root/anaconda3/envs/tic/bin/python -u train_edge_grad_rrdb.py -d  /taofei_9F/dataset/flicker  --save_path  /taofei_9F/imgCompression/MyCompression/result/grad_joint_training/flicker/concate_rrdb  --lambda 0.0483


cd /taofei/imageCompression/MyCompression && /root/anaconda3/envs/tdvc111/bin/python  -u ./train_edge_grad_rrdb.py -d /taofei/dataset/flicker --cuda  --epochs 1000   --save_path /taofei/imageCompression/MyCompression/result/grad_joint_training/flicker/concate_rrdb --lambda 0.0067