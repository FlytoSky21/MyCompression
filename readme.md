## train
CUDA_VISIBLE_DEVICES='0' python -u ./train.py -d /home/adminroot/taofei/dataset/vimeo_septuplet 
--cuda  --epochs 100   --save_path /home/adminroot/taofei/DCC2023fuxian/bn16


## test
python eval.py --checkpoint [path of the pretrained checkpoint] 
--data [path of testing dataset] --cuda

python eval.py --checkpoint /home/adminroot/taofei/DCC2023fuxian/result_fk/0.0067checkpoint_best.pth.tar
--data /home/adminroot/taofei/dataset/Kodak24 --cuda

