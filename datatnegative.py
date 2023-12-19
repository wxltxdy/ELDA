"""用于对数据进行负采样的函数"""
from joblib import Parallel, delayed
import glob
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from scipy import sparse
import numpy as np
import global_constants as gc
import time
import text_utils
import pandas as pd
import argparse

parser = argparse.ArgumentParser("描述:对数据开始进行——负采样")
parser.add_argument('--data_path', default='data', type=str, help='path to the data')
parser.add_argument('--dataset', default="ml10m", type=str, help='dataset')
parser.add_argument('--n_factors', default=40, type=int, help='number of hidden factors for user/item representation')
parser.add_argument('--neg_item_inference', default=1, type=int, help='if there is no available disliked items, set this to 1 to infer '
                                                                      'negative items for users using our user-oriented EM like algorithm')
parser.add_argument('--neg_sample_ratio', default=0.6, type=float, help='negative sample ratio per user. If a user consumed 10 items, and this'
                                                                        'neg_sample_ratio = 0.2 --> randomly sample 2 negative items for the user')
args = parser.parse_args()
DATA_DIR = os.path.join(args.data_path, args.dataset)
gc.DATA_DIR = DATA_DIR
n_components = args.n_factors#潜在变量的维度
NEGATIVE_SAMPLE_RATIO = args.neg_sample_ratio#负采样比例

# 计算出来一共有多少用户和项目数量
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_movieId = list()
with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_movieId.append(line.strip())
n_items = 744
n_users = 744
print (n_users, n_items)

def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['userId']), np.array(tp['movieId'])
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int')), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    rows, cols, vals = np.array(tp['movieId1']), np.array(tp['movieId2']), np.array(tp['count'])
    seq = np.concatenate((rows[:, None], cols[:, None], vals[:, None]), axis=1)
    data = sparse.csr_matrix((vals, (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

if args.neg_item_inference:
    #initialize with WMF:
    print("开始运行自己定义的负采样模型算法!!!")
    import wmf
    import rec_eval
    from scipy import sparse
    import produce_negative_embedding as pne
    import glob
    import os
    # 输入的是一个数组X
    def softmax(x):
        """Compute softmax values for each ranked list."""
        #我们希望排名得分较高、可能性较低的项目作为负面实例
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    # 负面实例的概率值
    def compute_neg_prob(ranks):
        return softmax(np.negative(ranks))
    #这是一个用于进行评分预测的函数定义。它接受训练数据 train_data，用户和物品的 latent factors Et 和 Eb，
    # 一段用户范围 user_idx 和 batch_users（表示待处理的用户数量）、可选参数 mu 和 vad_data（表示验证集）
    def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None,vad_data=None):
        n_songs = train_data.shape[1]# train_data 的列数（即物品数量）赋值给 n_songs
        # batch_users 行、n_songs 列的布尔类型数组 item_idx，并将其初始化为 False。
        item_idx = np.zeros((batch_users, n_songs), dtype=bool)
        item_idx[train_data[user_idx].nonzero()] = True
        if vad_data is not None:
            item_idx[vad_data[user_idx].nonzero()] = True
        # 从 Et（用户隐变量）中选取 user_idx 所对应的那些用户 latent factor，然后对它们和所有物品 latent factor Eb 进行内积运算，得到一个 batch_users * n_songs 的矩阵 X_pred，
        X_pred = Et[user_idx].dot(Eb)
        # 这段代码的作用是对评分矩阵 X_pred 进行修正。通过乘以一个修正因子来调节评分，在某种程度上可以提高模型的性能
        if mu is not None:
            if isinstance(mu, np.ndarray):
                assert mu.size == n_songs  # mu_i
                X_pred *= mu
            elif isinstance(mu, dict):  # func(mu_ui)
                params, func = mu['params'], mu['func']
                args = [params[0][user_idx], params[1]]
                if len(params) > 2:  # for bias term in document or length-scale
                    args += [params[2][user_idx]]
                if not callable(func):
                    raise TypeError("expecting a callable function")
                X_pred *= func(*args)
            else:
                raise ValueError("unsupported mu type")
        # 因为softmax函数在得分为无穷大时会趋近于0
        X_pred[item_idx] = np.inf#item_idx 将那些已知的正样本对应位置的评分修改为 np.inf，从而保证这些已知的正样本不会被误判成负样本
        return X_pred

    # 这是一个生成负面样本的函数，用于训练协同过滤模型。其中，train_data是原始的用户与物品交互矩阵，
    # U和VT是分解得到的低维矩阵，user_idx是当前批次要处理的用户索引范围，neg_ratio参数表示生成负样本的比率。
    def gen_neg_instances(train_data, U, VT, user_idx, neg_ratio = 1.0):
        print ('Job start... %d to %d'%(user_idx.start, user_idx.stop))
        batch_users = user_idx.stop - user_idx.start
        # 函数首先调用 _make_prediction 函数来为每个用户预测评分，最后的那个为验证集
        X_pred = _make_prediction(train_data, U, VT, user_idx, batch_users, vad_data=vad_data)
        rows = []
        cols = []
        total_lost = 0
        # 这段代码是对隐反馈数据进行负面采样的过程。
        for idx, uid in enumerate(range(user_idx.start, user_idx.stop)):
            num_pos = train_data[uid].count_nonzero()# 该用户的正样本的数量
            num_neg = int(num_pos * neg_ratio)#统计负面采样个数，每一个用户的数据都不同
            if num_neg <= 0: continue
            ranks = X_pred[idx]# X_pred[idx] 中查找其对（所有）物品的预测得分 ranks
            # 使用 compute_neg_prob 函数计算出每个物品被作为负样本的概率,
            # 因为之前已经把交互的项目设置为inf了，所以导致选择为未交互的项目
            neg_withdrawn_prob = compute_neg_prob(ranks)
            #从 n_items 个物品中随机选择 num_neg 个物品作为负样本，根据 neg_withdrawn_prob 定义的概率进行采样。
            # 所以这里选中的物品都是未交互过的，set 函数将结果转换为一个集合
            neg_instances = list(set(np.random.choice(range(n_items), num_neg, p = neg_withdrawn_prob)))
            if uid < 0: print ('error with %d to %d'%(user_idx.start, user_idx.stop))
            # 将用户 uid 和对应的负样本编号组成的二元组加入到稀疏矩阵中。
            rows = np.append(rows, np.full( len(neg_instances), uid ))
            cols = np.append(cols, neg_instances)
        if len(rows) > 0:
            path = os.path.join(DATA_DIR, 'sub_dataframe_idxstart_%d.csv' % (user_idx.start))
            assert len(rows) == len(cols)
            # 这段代码的作用是将构建出来的数据点写入到文件中
            with open(path, 'w') as writer:
                for i in range(len(rows)): writer.write(str(rows[i]) + "," + str(cols[i]) + '\n')
                writer.flush()
            #df = pd.DataFrame({'uid':rows, 'sid':cols}, columns=["uid", "sid"], dtype=np.int16)
            #df.to_csv(path, sep=",",header=False, index = False)
        # return df


    U, V = None, None
    vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'Api_Api_valids.csv'))
    train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'Api_Api_trains.csv'))

    # 调用传统的非负矩阵分解的模型 !!!!!!!!!!!!!!!!!!!!!!!!!!!
    U, V = wmf.decompose(train_data, vad_data, num_factors= n_components)
    VT = V.T

    ################（期望）E step: ######################
    # 通常用于最大期望（Expectation-Maximization，EM）算法中。
    # E步的主要作用是根据当前参数（或初始化参数）计算隐变量的后验概率分布，即计算每个样本属于每个类别的概率值，以便后续更新模型参数。
    user_slices = rec_eval.user_idx_generator(n_users, batch_users=50)
    print ('正在生成每一批次的负面实例采样...')
    t1 = time.time()
    # 每个用户索引对应的数据子集上生成负嵌入样本
    # 针对每个用户，它会生成一些随机的负面样本。这些负面样本可以被用来更新模型参数，以帮助模型更好地拟合训练数据
    df = Parallel(n_jobs=16)(delayed(gen_neg_instances)(train_data, U, VT, user_idx, neg_ratio = NEGATIVE_SAMPLE_RATIO)
                                  for user_idx in user_slices)
    t2 = time.time()
    print ('Time : %d seconds' % (t2 - t1))

    print ('合并到一个文件写到本地(写入方式是采用追加模式)...')
    t1 = time.time()
    # 代码使用 os.system 命令将所有生成的负面实例文件合并成一个文件，并保存在本地。
    # 这些负面实例将在接下来的最大化步骤中被用来更新模型，一般会保存多个train_neg_iter_%d.csv文件，
    # 每一个文件都包含所有的sub_dataframe_iter*，也就是完整的负面数据
    neg_file_out = os.path.join(DATA_DIR, 'train_neg.csv')
    with open(neg_file_out, 'w', encoding='utf-8') as writer:
        writer.write('APIId1,APIId2\n')
        for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_*')):
            with open(f, 'r', encoding='utf-8') as reader:
                for line in reader:
                    writer.write(line.strip() + '\n')

    #clean

    for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_*')):
        os.remove(f)

    t2 = time.time()
    print ('Time : %d seconds' % (t2 - t1))

    best_train_neg_file = os.path.join(DATA_DIR, 'train_neg.csv')
    '@到此生成最终的负采样文件！！！'
    best_train_neg_file_newname = os.path.join(DATA_DIR, 'train_neg_result.csv')
    if os.path.exists(best_train_neg_file_newname):
        os.remove(best_train_neg_file_newname)
    print ('renaming from %s to %s'%(best_train_neg_file, best_train_neg_file_newname))
    os.rename(best_train_neg_file, best_train_neg_file_newname)

