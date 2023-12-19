# -*- coding: utf-8 -*-
"""
[统计]估计量
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Optional
import neptune.new as neptune
import numpy as np
import pandas as pd
import pyro
import weakref
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyro.infer import SVI, JitTraceEnum_ELBO, Predictive, TraceEnum_ELBO
from pyro.optim import ClippedAdam

from . import lda
from .datasets import Interactions
from .losses import adaptive_hinge_loss, bpr_loss, hinge_loss, logistic_loss
from .nets import MFNet, NMFNet, SNMFNet
from .utils import cpu, gpu, minibatch, process_ids, sample_items, set_seed, shuffle,log_summary
from . import evaluations as lda_eval
import dill
_logger = logging.getLogger(__name__)

# 最原始的基类   它包含用于拟合、预测和计算推荐分数的公共方法
class EstimatorMixin(metaclass=ABCMeta):

    # 这是一个Python类中的特殊方法——repr，用于返回一个“可打印”的字符串，用于表示该类的实例对象。
    def __repr__(self):
        return "<{}: {}>".format(
            self.__class__.__name__,
            repr(self._model),
        )

    @abstractmethod
    def fit(self, interactions):
        pass

    # 它实际上实现了预测过程,一个用户的ID,一个项目的ID,进行的一对一预测
    def _predict(self, user_ids, item_ids):
        """如果您需要做更多的事情而不是仅仅应用模型，则(重写函数)此内容"""
        self._model.train(False)
        return self._model(user_ids, item_ids)

    # 返回的结果就是在指定用户的item_ids中所有项目的预测分数。
    def predict(self, user_ids, item_ids=None, cartesian=False):
        """
        预测:给定用户id,计算建议项目分数
        使用这个作为Mixin来避免双重实现
        Args:
            user_ids (int或array):如果是int，将预测推荐分数 对于这个用户，item_ids中的所有项。如果是数组，将会预测user_ids和item_ids定义的所有(user, item)对的得分。
            item_ids (array，可选):包含项id的数组 预测分数是需要的。如果不提供，所有的预测项目将被计算。
            cartesian (bool，可选):计算每个项目的预测次数每个用户。
        Returns:
            np.array: item_ids中所有项目的预测分数。.
        """
        # 若为标量 则说明只有一个用户需要计算推荐分数。否则，将同时计算多个用户的推荐分数。
        if np.isscalar(user_ids):
            n_users = 1
        else:
            n_users = len(user_ids)
        try:
            user_ids, item_ids = process_ids(
                user_ids, item_ids, self._n_items, self._use_cuda, cartesian
            )
        except RuntimeError as e:
            raise RuntimeError("Maybe you want to set `cartesian=True`?") from e
        # print(f"需要预测的项目的个数{self._n_items}")
        # 注意本文中是预测所有的项目中应该得到的分数
        out = self._predict(user_ids, item_ids)
        out = cpu(out).detach().numpy()
        if cartesian:
            return out.reshape(n_users, -1)
        else:
            return out.flatten()


class PopEst(EstimatorMixin):
    """估计器使用的流行度"""

    def __init__(self, *, rng=None):
        self.pops = None
        self._n_users = None
        self._n_items = None
        self._use_cuda = False
        # Only for consistence to other estimators
        self._rng = np.random.default_rng(rng)

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    """这段代码是一个 fit 方法的实现。根据代码的逻辑，它的作用是将输入的交互数据进行处理，并计算出每个物品的流行度分数。
        具体来说，该方法包含以下步骤：
        获取交互数据中的用户数量和物品数量，并将其存储在对象的 _n_users 和 _n_items 属性中。
        将交互数据转换为 Pandas 数据框，并保存在变量 df 中。
        使用 Pandas 的 groupby 方法，按照 item_id 进行分组统计，并计算每个物品的流行度分数（即每个物品被交互的次数）。
        将流行度分数归一化到 [0, 1] 的范围内，以便后续使用。首先，通过将流行度分数除以最大分数来进行归一化，得到归一化后的流行度分数。
        将归一化后的流行度分数转换为 Torch 张量，并保存在对象的 pops 属性中。
        总结起来，这段代码的目的是计算每个物品的流行度分数，并将其归一化后保存在模型对象的属性中。这些流行度分数可以在后续的模型训练或推荐过程中使用。"""
    def fit(self, interactions):
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items

        df = interactions.to_pandas()
        # 数据框按照 item_id 进行分组统计，得到不同物品的流行度分数并将其归一化
        # 流行度分数:反映了每个物品在所有交互记录中被交互的次数，因此较高的流行度分数意味着该物品的流行度更高
        pops = (
            df.groupby("item_id", as_index=False)["rating"]
            .agg("count")
            .rename(columns={"rating": "score"})
        )
        all_items = pd.DataFrame({"item_id": np.arange(interactions.n_items)})
        pops = pd.merge(pops, all_items, how="right", on="item_id").fillna(0.0)["score"]
        pops = pops / pops.max()
        self.pops = torch.from_numpy(pops.to_numpy().squeeze())

    def _predict(self, user_ids, item_ids):
        """Overwrite this if you need to do more then just applying the model"""
        assert self.pops is not None
        return self.pops[item_ids]

    def save(self, filename):
        torch.save(self.pops, filename)
        return self

    def load(self, filename, interactions):
        """Load model, interactions are only used to infer metadata"""
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items
        self.pops = torch.load(filename)
        return self


class LDA4RecEst(EstimatorMixin):
    """LDA估计器"""
    def __init__(
        self,
        *,
        embedding_dim: int,#表示用户、物品的嵌入维度；
        n_iter: int,#表示训练迭代次数；
        alpha: Optional[float] = None,#表示LDA分布的Dirichlet先验分布参数
        batch_size: Optional[int] = 32,
        w: float = 0.01, #表示负模型的比例
        learning_rate: float = 0.01,#表示学习率；
        use_jit: bool = True,#表示是否使用JIT加速训练；
        use_cuda: bool = False,#表示是否使用CUDA加速训练；
        rng=None,#表示随机数生成器，默认使用numpy.random.default_rng()；
        predict_posterior=False,#是否需要计算后验分布，如果为True，则会大大减慢模型的训练速度；
        clear_param_store: bool = True,#是否在每次迭代之前清空参数存储库（pyro.param())中的参数；
        log_steps=200,#表示训练中打印日志的频率，即每隔多少次迭代输出一次训练日志；
        model=None,
        pred_model=None,
        guide=None,
        pred_guide=None,
        n_samples=1000,
    ):
        self._embedding_dim = (
            embedding_dim  # 便于与其他估算器进行比较
        )
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._w = w
        self._use_jit = use_jit
        self._alpha = alpha
        self._use_cuda = use_cuda
        self._clear_param_store = clear_param_store
        self._log_steps = log_steps
        self._rng = np.random.default_rng(rng)
        self._model = lda.model if model is None else model
        self._pred_model = lda.pred_model if pred_model is None else pred_model
        self._guide = lda.guide if guide is None else guide
        self._pred_guide = lda.pred_guide if pred_guide is None else pred_guide

        self._n_samples = n_samples
        set_seed(self._rng.integers(0, 2 ** 32 - 1), cuda=self._use_cuda)

        # Initialized after fit
        self.pops = None
        self.user_topics = None
        self.topic_items = None
        self.params = None  # 所有推断的params包括上面
        self._n_users = None
        self._n_items = None
        self._model_params = None  # 安装时使用的params
        self.predict_posterior = predict_posterior  # 如果为真，则极慢

    def _initialize(self, model_params):
        self._model_params = model_params
        self.params = pyro.get_param_store()
        'item 的流行度'
        self.pops = self.params[lda.Param.item_pops_loc].detach()
        '每个 topic 对应的 item 向量'
        self.topic_items = self.params[lda.Param.topic_items_loc].detach()
        '每个用户对应的 topic 分布'
        self.user_topics = F.softmax(
            self.params[lda.Param.user_topics_logits], dim=-1
        ).detach()
        # 每个用户的偏置向量
        self.user_pop_devs = self.params[lda.Param.user_pop_devs_loc].detach()

    # ** 用于对 LDA 推荐模型进行训练。方法的输入参数为一个 Interactions 类型的对象 ，表示包含着用户和物品之间的交互信息，
    # 并且还可以接受一个布尔值变量 clear_params 作为可选参数，用于指示是否清空 Pyro 参数存储器中的参数。
    def fit(self, interactions,test,est, clear_params=None):
        print("我要开始训练模型了！！")
        run = neptune.get_last_run()
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items
        train = interactions
        #interactions 对象中保存的交互数据转换为 PyTorch 张量格式
        # to_ratings_per_user()把原始的交互矩阵转换成一个列表，其中每个元素表示一个用户与其评分过的所有物品之间的交互关系
        # print(f"训练之前的数据格式是{interactions.ratings}")
        interactions = torch.tensor(interactions.to_ratings_per_user(), dtype=torch.int)
        # print(interactions[:5])
        # print(interactions[:,:5])
        # print(interactions.shape)
        clear_params = self._clear_param_store if clear_params is None else clear_params
        if clear_params:
            pyro.clear_param_store()
        pyro.enable_validation(__debug__)
        # 代码将模型训练所需的参数和超参数保存在一个字典 model_params 中
        model_params = dict(
            interactions=interactions,
            n_topics=self._embedding_dim,
            n_users=self._n_users,
            n_items=self._n_items,
            alpha=self._alpha,
            batch_size=self._batch_size,
        )
        # 指定适合当前模型的 ELBO 估计器
        Elbo = JitTraceEnum_ELBO if self._use_jit else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=2)


        # 代码创建了一个基于 ClippedAdam 优化器的对象 optim,是一种基于动量的随机梯度下降 (SGD) 优化器
        # SVI 算法是一种基于随机优化的近似推断方法
        optim = ClippedAdam({"lr": self._learning_rate})
        svi = SVI(self._model, self._guide, optim, elbo)
        print("一共要经过 %d 次训练" % self._n_iter)
        epoch_loss = None
        for epoch_num in range(self._n_iter):
            # 利用 SVI 算法对模型进行训练，并计算当前迭代的损失值 epoch_loss
            epoch_loss = svi.step(**model_params)
            run["train/loss"].log(epoch_loss)
            if epoch_num % self._log_steps == 0:
                _logger.info("Epoch {: >5d}: loss {}".format(epoch_num, epoch_loss))
            if epoch_num % 200 == 0 and epoch_num != 0:
                # 1. 正模型的存储位置
                model_path = f"//home/wxl/code/LDA4rec/wxl_model/528/fu/0.8_{epoch_num}model"
                # model_path = f"/home/wxl/code/LDA4rec/models_huawei/1.0_{epoch_num}"
                self._initialize(model_params)
                "我希望进行一次中断操作，调到另一个名称为cli.py文件中的一行代码语句"
                df = lda_eval.summary_positive(est,train=train, test=test, eval_train=False)
                log_summary(df.reset_index())
                _logger.info(f"Result:\n{df.reset_index()}")
                # if epoch_num == 1400:
                _logger.info(f"存储正模型 {model_path}...")
                # 保存正模型
                with open(model_path, 'wb') as f:
                    dill.dump(est, f)
        # 代码计算最终迭代的损失值，并调用 self._initialize() 方法进行模型参数的初始化
        # Pyro 框架内部提供的 elbo.loss() 方法，利用最新参数计算模型的 ELBO 损失，并把该损失值用于模型参数的初始化
        if self._n_iter != 0:
            epoch_loss = elbo.loss(self._model, self._guide, **model_params)
        # self._initialize(model_params)
        return epoch_loss

    def _predict_posterior(self, user_ids, item_ids):
        """计算贝叶斯后验，极慢!"""
        assert len(torch.unique(user_ids)) == 1, "invalid usage"

        user_id = user_ids[0]
        params = self._model_params.copy()
        params["batch_size"] = None
        del params["interactions"]
        params["user_id"] = user_id

        predictive = Predictive(
            self._pred_model,
            guide=self._pred_guide,
            num_samples=self._n_samples,
            return_sites=[lda.Site.interactions],
            parallel=False,
        )
        items = predictive(**params)["interactions"].squeeze(1)
        counts = torch.bincount(items, minlength=self._n_items)
        # 通过从[0,1]中随机添加值来打破关系
        counts = counts + torch.rand(counts.shape)

        return counts[item_ids]

    def _predict_point(self, user_ids, item_ids):
        """计算点估计的频率方法，快"""
        assert len(torch.unique(user_ids)) == 1, "invalid usage"

        user_topics = self.user_topics[user_ids]
        topic_items = self.topic_items[:, item_ids].T
        item_pops = self.pops[item_ids].unsqueeze(1)
        user_pop_devs = self.user_pop_devs[user_ids].unsqueeze(1)
        topic_prefs = topic_items + torch.exp(user_pop_devs) * item_pops

        dot = user_topics * topic_prefs
        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        return dot

    def _predict(self, user_ids, item_ids):
        if self.predict_posterior:
            return self._predict_posterior(user_ids, item_ids)
        else:
            return self._predict_point(user_ids, item_ids)

    def save(self, filename):
        pyro.get_param_store().save(filename)
        return self

    def load(self, filename, interactions):
        """加载模型，交互仅用于推断元数据"""
        self._n_iter = 0
        pyro.get_param_store().load(filename)
        self.fit(interactions, clear_params=False)
        return self




# MFEst的第2父基类
class BaseEstimator(EstimatorMixin, metaclass=ABCMeta):
    """基估计器处理隐式反馈训练和预测"""

    def __init__(
        self,
        *,
        model_class,
        loss,
        embedding_dim,
        n_iter=10,
        batch_size=128,
        l2=0.0,
        learning_rate=1e-2,
        optimizer=None,
        use_cuda=False,
        rng=None,
        sparse=False,
        deque_max_len=10,
        conv_slope_max=-1e-4,
    ):
        self._model_class = model_class
        self._embedding_dim = embedding_dim
        self._use_cuda = use_cuda
        self._rng = np.random.default_rng(rng)
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._optimizer = optimizer
        self._sparse = sparse
        self._conv_deque = deque(maxlen=deque_max_len)
        self._conv_slope_max = conv_slope_max
        set_seed(self._rng.integers(0, 2 ** 32 - 1), cuda=self._use_cuda)

        self._loss = {
            "bpr": bpr_loss,
            "logistic": logistic_loss,
            "hinge": hinge_loss,
            "adpative-hinge": adaptive_hinge_loss,
        }[loss]

    def _initialize(self, interactions):
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items

        model = self._model_class(
            self._n_users,
            self._n_items,
            embedding_dim=self._embedding_dim,
            sparse=self._sparse,
        )
        self._model = gpu(model, self._use_cuda)

        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self._model.parameters(), weight_decay=self._l2, lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer(self._model.parameters())

    def converged(self, loss):
        self._conv_deque.append(loss)
        if not len(self._conv_deque) == self._conv_deque.maxlen:
            return False

        diffs = np.diff(self._conv_deque)
        return np.mean(diffs) >= self._conv_slope_max

    # 这段代码实现了主题模型的训练过程，包括对交互数据进行随机洗牌（shuffle）和分批次训练。具体来说：
    def fit(self, interactions: Interactions):
        """Fit the model   符合模型  """
        run = neptune.get_last_run()
        self._initialize(interactions)
        self._model.train(True)
        self._conv_deque.clear()

        epoch_loss = None

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        #这段代码实现了主题模型的训练过程，包括对交互数据进行随机洗牌（shuffle）和分批次训练。具体来说：
        for epoch_num in range(self._n_iter):
            # 将用户ID和物品ID随机洗牌，以减少模型训练过程中的偏差。
            users, items = shuffle(user_ids, item_ids, rng=self._rng)
            # gpu()函数将PyTorch张量放置在CUDA设备上，以利用GPU的并行计算能力加速模型训练过程
            user_ids_tensor = gpu(torch.from_numpy(users), self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items), self._use_cuda)

            epoch_loss = 0.0
            minibatch_num = -1
            # minibatch()函数是一个生成器，用于将数据按照batch_size分批次处理
            # 返回的batches是一个可迭代的对象，它包含了所有mini-batch的信息
            batches = minibatch(
                user_ids_tensor, item_ids_tensor, batch_size=self._batch_size
            )
            # 这段代码主要实现了对每个mini-batch的训练过程，包括正样本和负样本的预测值计算、损失函数的计算、梯度的反向传播和参数的更新。
            for minibatch_num, (batch_user, batch_item) in enumerate(batches):
                # 计算当前mini-batch中每个正样本(user_id, item_id)的预测值
                positive_prediction = self._model(batch_user, batch_item)
                # 计算当前mini-batch中每个正样本对应的负样本的预测值，该操作是为了增强模型的泛化能力
                negative_prediction = self._get_negative_prediction(batch_user)
                # 清空优化器中之前积累的梯度信息
                self._optimizer.zero_grad()

                loss = self._loss(positive_prediction, negative_prediction)
                epoch_loss += loss.item()
                # 执行梯度的反向传播，计算每个模型参数对应的梯度信息
                loss.backward()
                # 利用优化器进行梯度下降，更新模型参数
                self._optimizer.step()

            if minibatch_num == -1:
                raise RuntimeError("There is not even a single mini-batch to train on!")

            epoch_loss /= minibatch_num + 1

            run["train/loss"].log(epoch_loss)
            _logger.info("Epoch {: >5d}: loss {}".format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError("Degenerate epoch loss: {}".format(epoch_loss))

            if self.converged(epoch_loss):
                _logger.info("Converged after {} epochs.".format(epoch_num))
                break

        run["train/n_epochs"].log(epoch_num)
        return epoch_loss

    # ToDo: 检查我们是否不能在PyTorch中完成所有工作
    # 为了训练模型而生成的负样本，是通过从训练集中随机选择未评分、或者低评分的物品而生成的
    def _get_negative_prediction(self, user_ids):
        """从整个项目集中统一抽取负项目"""
        negative_items = sample_items(self._model.n_items, len(user_ids), rng=self._rng)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._model(user_ids, negative_var)

        return negative_prediction

    def save(self, filename):
        torch.save(self._model.state_dict(), filename)
        return self

    # 基本的类模型
    def load(self, filename, interactions):
        """加载模型时，交互仅用于推断元数据"""
        self._initialize(interactions)
        self._model.load_state_dict(torch.load(filename))
        self._model.eval()
        return self

# MFEst的第1父基类    本论文的基本的方法-数学公式 **** 它是一个Mixin类，用于将MF转换为伴随的LDA公式
class LDATrafoMixin(metaclass=ABCMeta):
    """混合(Mixin)类 --> 将MF转化为伴随的LDA配方"""

    def __init__(self, *args, **kwargs):
        # 将此实例变量  切换为使用LDA公式进行预测
        self.lda_trafo = False
        super().__init__(*args, **kwargs)

    '''该代码块是一个函数，用于从模型中获取单个用户或所有用户的NMF表示。
        1.首先从模型中获取用户、物品和偏置向量的嵌入表示，然后将用户的嵌入表示根据正向部分和负向部分进行划分，
        2.之后将正向部分与负向部分合并成一整个矩阵，同时也将对应的物品嵌入表示进行同样的操作，保证其非负性。
        3.最后，通过加上一个offset值来保证其非负性，并返回经过处理后的 w, h, b 三个矩阵。'''
    def get_nmf_params(self, user_id=None):
        """获取单个user_id或所有user_id的NMF表示(如果None)
            根据论文的引理
        """
        if user_id is None:
            user_ids = torch.arange(self._n_users, dtype=torch.int64)
        else:
            user_ids = user_id.expand(1)#如果不为空就获取指定用户的 NMF 表示

        item_ids = torch.arange(self._n_items, dtype=torch.int64)

        w = self._model.user_embeddings(user_ids).detach()
        b = self._model.item_biases(item_ids).squeeze().detach()
        h = self._model.item_embeddings(item_ids).detach()

        # 代码中对w h b 进行了非负处理
        w_pos, w_neg = torch.zeros_like(w), torch.zeros_like(w)
        pos_mask = w >= 0
        neg_mask = ~pos_mask
        w_pos[pos_mask], w_neg[neg_mask] = w[pos_mask], -w[neg_mask]
        w = torch.cat([w_pos, w_neg], dim=1)
        h = torch.cat([h, -h], dim=1)

        # 选择最小偏移以保证非负性，而不仅仅是abs的最大值。
        h += torch.minimum(torch.min(h, dim=0).values, torch.zeros(h.shape[1])).abs()
        b += torch.minimum(torch.min(b), torch.zeros(1)).abs()

        assert torch.all(h >= 0.0)
        assert torch.all(w >= 0.0)
        assert torch.all(b >= 0.0)

        return w, h, b

    def get_lda_params(self, user_id=None, eps=1e-6):
        """
            获取单个user_id或所有user_id的伴随LDA公式(如果None)从论文的定理
        """
        # 调用get_nmf_params()方法获取NMF模型中W、H、B矩阵的值。然后使用这些值计算LDA的伴随公式。
        w, h, b = self.get_nmf_params(user_id)
        n_users, n_topics = w.shape # w的行数表示用户数量，列数表示潜在因素（主题）数量

        # 断言队列是分类分布
        m = h.sum(dim=0)
        h = h / m.unsqueeze(0)
        w = w * m

        # 右边的2个
        b_sum = b.sum()
        b = b / b_sum
        t = w.sum(dim=1, keepdim=True) / b_sum

        n = h.sum(dim=0, keepdim=True).expand(n_users, -1) + 1.0 / t
        v = w * n
        v = v / v.sum(dim=1, keepdim=True)

        assert torch.all(b >= 0.0)
        assert (b.sum() - 1.0).abs() <= eps
        assert torch.all(v >= 0.0)
        assert torch.all((v.sum(dim=1) - 1.0).abs() <= eps)
        assert torch.all(h >= 0.0)
        assert torch.all((h.sum(dim=0) - 1.0).abs()) <= eps

        return v, t, h, b

    #该代码块是一个函数，用于获取特定用户(user_id)对所有物品的推荐概率
    def get_item_probs(self, user_id, eps=1e-6) -> np.array:
        v, t, h, b = self.get_lda_params(user_id, eps=eps)
        # 从定理的证明中
        g = h + b.unsqueeze(-1) / t
        n = g.sum(dim=0)
        g = g / n

        assert torch.all(g >= 0.0)
        topic_sums = (g.sum(dim=0) - np.ones(g.shape[1])).abs()
        assert torch.all(topic_sums <= eps * topic_sums.shape[0])

        probs = torch.matmul(v, g.T).squeeze()

        assert torch.all(probs >= 0.0)
        assert (probs.sum() - 1.0).abs() <= eps

        return probs

    def _predict(self, user_ids, item_ids):
        if self.lda_trafo:
            assert len(torch.unique(user_ids)) == 1, "invalid usage"
            return self.get_item_probs(user_ids[0], eps=1e-4)
        else:
            # dispatch to BaseEstimator, if the gods of MRO have mercy on me
            return super()._predict(user_ids, item_ids)

# 定义了一个名为MFEst的模型类，该类继承了LDATrafoMixin和BaseEstimator两个父类，是一个具有一定通用性的基础模型类
class MFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=MFNet, loss=loss, **kwargs)


class SNMFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=SNMFNet, loss=loss, **kwargs)


class NMFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=NMFNet, loss=loss, **kwargs)

class HierLDA4RecEst(LDA4RecEst):
    # 在 __init__ 方法中，首先调用了父类 LDA4RecEst 的构造方法，将参数 model 和 guide 分别传递给父类的构造方法。
    # 这里的 model 和 guide 参数是指定该推荐系统使用的主题模型和变分推断算法
    def __init__(self, **kwargs):
        super().__init__(model=lda.lda_plus_model, guide=lda.hier_guide, **kwargs)

class HierVarLDA4RecEst(LDA4RecEst):
    def __init__(self, **kwargs):
        super().__init__(model=lda.lda_plus_model, guide=lda.hier_var_guide, **kwargs)

class LDA(LDA4RecEst):
    def __init__(self, **kwargs):
        super().__init__(model=lda.lda_model, guide=lda.hier_guide, **kwargs)

