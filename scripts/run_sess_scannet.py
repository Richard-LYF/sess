import os

def pretrain(labeled_ratio=0.1):

    command = 'CUDA_VISIBLE_DEVICES=%d python pretrain_votenet.py' %GPU_ID \
        + ' --dataset=scannet' \
        + ' --labeled_ratio=' + str(labeled_ratio) \
        + ' --log_dir=./log_scannet/votenet_%.1f' %labeled_ratio \
        + ' --print_interval=10'  \
        + ' --eval_interval=9' \
        + ' --labeled_sample_list=scannetv2_train_%.1f.txt' %labeled_ratio

    os.system(command)


def train(labeled_ratio=0.1):

    command = 'CUDA_VISIBLE_DEVICES=%d python train_sess.py' %GPU_ID \
        + ' --dataset=scannet' \
        + ' --labeled_sample_list=scannetv2_train_%.1f.txt'  %labeled_ratio \
        + ' --detector_checkpoint=' + '/home/yifan/Code/sess-master/votenet_%.1f/checkpoint.tar' %labeled_ratio \
        + ' --log_dir=./3log_scannet/sess_%.1f' %labeled_ratio \
        + ' --print_interval=20'  \
        + ' --eval_interval=5' \

    os.system(command)


def eval_inductive(labeled_ratio=0.1):
    # Evaluate on inductive semi-supervised 3D object detection
    command = 'CUDA_VISIBLE_DEVICES=%d python eval.py' %GPU_ID \
        + ' --dataset=scannet' \
        + ' --labeled_sample_list=scannetv2_train_%.1f.txt'  %labeled_ratio \
        + ' --checkpoint_path=' + './3log_scannet/sess_%.1f/checkpoint.tar' %labeled_ratio \
        + ' --dump_dir=./3dump_scannet/sess_%.1f' %labeled_ratio \
        + ' --use_3d_nms'  \
        + ' --use_cls_nms' \
        + ' --per_class_proposal' \
        + ' --dump_files' \

    os.system(command)


def eval_transductive(labeled_ratio=0.1):
    # Evaluate on transductive semi-supervised 3D object detection
    command = 'CUDA_VISIBLE_DEVICES=%d python eval.py' %GPU_ID \
        + ' --dataset=scannet' \
        + ' --labeled_sample_list=scannetv2_train_%.1f.txt'  %labeled_ratio \
        + ' --checkpoint_path=' + './3log_scannet/sess_%.1f/checkpoint.tar' %labeled_ratio \
        + ' --dump_dir=./3dump_scannet/sess_%.1f' %labeled_ratio \
        + ' --use_3d_nms'  \
        + ' --use_cls_nms' \
        + ' --per_class_proposal' \
        + ' --transductive' \

    os.system(command)


if __name__ == '__main__':
    GPU_ID = 3
    #labeled_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7] 2021/02/18
    labeled_ratio_list = [0.1]
    import sys
    sys.path.append('/home/yifan/Code/sess-master')
    for labeled_ratio in labeled_ratio_list:
        print('++++++++ RUN labeled_ratio=%.1f on ScanNet ++++++++' %labeled_ratio)
        # Please uncomment any line below to skip the execution of the corresponding phase.
        #pretrain(labeled_ratio=labeled_ratio) #2021/02/18
        #print('no pretrain, only train')
        train(labeled_ratio=labeled_ratio)
        eval_inductive(labeled_ratio=labeled_ratio)
        eval_transductive(labeled_ratio=labeled_ratio)
