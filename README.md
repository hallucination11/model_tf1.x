# MODEL

conda activate tr1.15

note:
dataset is criteo dadaset. all dense features are transformed to float type (missing data are completed by their 0.0).
all sparse features are transformed to int using label encoder

# DCN

python main.py --optimizer PAO --model DCN --lr 0.001 --epoch 1 --task_type binary --cross_num 4
--cross_parameterization matrix

