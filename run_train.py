# coding=utf-8

# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets'

# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
img_train = 'data55789/img_train.zip'
lab_train = 'data55789/lab_train.zip'
img_testA = 'data55789/img_testA.zip'
paddleseg = 'data56153/PaddleSeg-release-v0.6.0.zip'

# 文件夹所在的目录
main_dir = '/root/paddlejob/workspace/code'

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = '/root/paddlejob/workspace/output'

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息.

import os

if __name__ == "__main__":
    os.system("mv my_pspnet101.txt my_pspnet101.yaml")  # 重命名

    print("解压PaddleSeg源码...")
    os.system(f"unzip {datasets_prefix}/{paddleseg}")
    os.system("mv PaddleSeg-release-v0.6.0 PaddleSeg")  # 重命名

    # print("替换修改过的训练和预测文件")
    # os.system("cp train.py PaddleSeg/pdseg/train.py")
    # os.system("cp vis.py PaddleSeg/pdseg/vis.py")

    print("解压数据集...")
    os.system("mkdir PaddleSeg/dataset/rs_data")
    os.system(f"cd PaddleSeg && unzip {datasets_prefix}/{img_train} -d dataset/rs_data/")
    os.system(f"cd PaddleSeg && unzip {datasets_prefix}/{lab_train} -d dataset/rs_data/")
    os.system(f"cd PaddleSeg && unzip {datasets_prefix}/{img_testA} -d dataset/rs_data/")

    print("Copy train_valid_test_list.txt...")
    os.system("cp *.txt PaddleSeg/dataset/rs_data/")
    os.system(f"ls PaddleSeg/dataset/rs_data/")

    # os.system("cd PaddleSeg && pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple")

    print('下载预训练模型...')
    os.system(f"cd PaddleSeg && python pretrained_model/download_model.py pspnet101_bn_coco")

    ####################注意修改卡的数量and相应的batchsize、lr、log_step##########################
    print('Training ...')
    # os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3")
    os.system("export CUDA_VISIBLE_DEVICES=0")
    os.system(
        f"cd PaddleSeg && python -m paddle.distributed.launch pdseg/train.py --cfg ../my_pspnet101.yaml --use_gpu --use_vdl --vdl_log_dir ./visual_logdir --do_eval --log_steps 5")

    print("Moving visual_logdir to output_dir...")
    os.system(f'cd PaddleSeg && mv visual_logdir {output_dir}')

    print("Moving saved_model_best_ckpt to output_dir...")
    os.system(f'cd PaddleSeg && mv saved_model/pspnet101_coco/best_model {output_dir}')

    # vis.py用notebook单卡跑, 脚本难以实现因为还要导入saved_model
