#python run.py \
#    --savedir "./results/run1" \
#    --gpu_num 5 \
#    --zdim 20 \
#    --mode train
#
#python run.py \
#    --savedir "./results/run1" \
#    --gpu_num 5 \
#    --zdim 20 \
#    --mode test

python run.py \
    --savedir "./results/run1" \
    --gpu_num 5 \
    --zdim 20 \
    --mode visualize \
    --scatter \
    --reconstruct
