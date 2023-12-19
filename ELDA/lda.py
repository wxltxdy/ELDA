""""
协同过滤的潜在Dirichlet分配模型
形状符号:(batch_shape | event_shape)，例如(10,2 | 3)
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import pyro
import pyro.distributions as dist
import pyro.optim
import torch
import torch.nn.functional as F
from pyro import poutine

from .utils import reparam_beta

#  int值丢失
NA = -999

# 类型声明
Model = Callable[..., torch.Tensor]
Guide = Callable[..., None]


class Plate:
    topics = "plate_topics"
    users = "plate_users"
    interactions = "plate_interactions"


class Site:
    user_topics_weights = "user_topics_weights" #用户主题权重
    interactions = "interactions"#在综合了所有的先验分布 + 后验分布 后生成的最终的的 交互数据
    topic_items = "topic_items"#主题 -单词矩阵，或者就是单词生成多项式
    item_pops = "item_pops"#！物品最终的流行度
    user_topics = "user_topics"#（就是当前用户的 主题分布多项式）
    user_inv = "user_inv"#用户流行度偏差
    item_topics = "item_topics"# 当前用户下的交互过的项目主题分布
    user_pop_devs = "user_pop_devs"# ！代表最终的用户流行度偏差，衡量用户对物品的整体喜好程度；
    topic_prior = "topic_prior"# 代表主题分布的先验概率(生成主题分布多项式的先验分布)，即在没有数据的情况下，每个主题被选择的概率；
    user_pop_devs_prior_mu = "user_pop_devs_prior_mu"#代表用户流行度偏差的均值先验，；
    user_pop_devs_prior_sigma = "user_pop_devs_prior_sigma"#表示随机生成的用户流行度偏差服从一个均值为 mu，标准差为 sigma 的正态分布

    @classmethod
    def all(cls) -> List[str]:
        return [
            a for a in dir(cls) if not (a.startswith("_") or callable(getattr(cls, a)))
        ]


class Param:
    topic_items_loc = "topic_items_loc"
    topic_items_scale = "topic_items_scale"
    user_topics_logits = "user_topics_logits"
    user_inv_logits = "user_inv_logits"
    item_pops_loc = "item_pops_loc"
    item_pops_scale = "item_pops_scale"
    user_pop_devs_loc = "user_pop_devs_loc"
    user_pop_devs_scale = "user_pop_devs_scale"
    topic_prior_logits = "topic_prior_logits"
    topic_prior_alpha = "topic_prior_alpha"
    topic_prior_beta = "topic_prior_beta"
    topic_prior_p = "topic_prior_p"
    topic_prior_q = "topic_prior_q"
    user_pop_devs_prior_mu_loc = "user_pop_devs_prior_mu_loc"
    user_pop_devs_prior_mu_scale = "user_pop_devs_prior_mu_scale"
    user_pop_devs_prior_sigma_loc = "user_pop_devs_prior_sigma_loc"
    user_pop_devs_prior_sigma_scale = "user_pop_devs_prior_sigma_scale"


@dataclass()
class ModelData:
    interactions: torch.Tensor #用户对物品的评分值
    item_pops: torch.Tensor  #物品流行度分布
    user_topics: torch.Tensor  #用户主题分布。


def model(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> ModelData:
    """贝叶斯型模型

    Args:
        交互:形状的二维数组(n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha
    n_interactions = interactions.shape[0]

    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )

    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None:
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            interactions = interactions[:, ind]

        user_topics = pyro.sample(  # (n_users | n_topics)
            Site.user_topics,
            dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
        )

        user_pop_devs = pyro.sample(  # (n_users | )
            Site.user_pop_devs,
            dist.LogNormal(-0.5 * torch.ones(1), 0.5),
        ).unsqueeze(1)

        with pyro.plate(Plate.interactions, n_interactions):
            item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                # 最终偏好取决于主题分布，
                # 产品的受欢迎程度和用户关心的程度
                # 商品受欢迎程度
                prefs = topic_items[item_topics] + user_pop_devs * item_pops
                interactions = pyro.sample(  # (n_interactions, n_users)
                    Site.interactions,
                    dist.Categorical(logits=prefs),
                    obs=interactions,
                )

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
    )


# 最原始的LDA主题模型
def lda_model(
    interactions: torch.Tensor,
    *,
    n_topics: int, #主题数量
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> ModelData:
    alpha = 1.0 / n_topics if alpha is None else alpha
    # 交互数据矩阵 interactions 的第一维长度，即用户和物品之间的交互次数。
    # 后续可以使用这个变量来确定模型中需要重复多少次的计算过程
    n_interactions = interactions.shape[0]
    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)
    #主题分布的先验概率——>提供一个合理的先验分布，辅助模型更好地拟合数据和泛化新数据。
    beta_alpha, beta_beta = reparam_beta(alpha, 0.9)#reparam_beta() 函数对 beta 分布的参数进行重新参数化, beta 分布在采样时更加稳定。
    topic_prior = pyro.sample(  # ( | n_topics)——> 维度大小正好和主题K的大小一致
        Site.topic_prior,
        dist.Beta(beta_alpha * torch.ones(n_topics), beta_beta).to_event(1),
    ).unsqueeze(0)

    # 每个主题（topic）对应的物品流行度分布。n_topics 则表示循环的上限次数，即主题的数量。
    '得到的就是主题-单词矩阵，或者就是单词生成多项式'
    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )
    # 模型的核心！！！
    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None: #输入数据中包含用户与物品的交互记录
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            #将其按列索引提取与当前循环的用户相对应的交互行，并保存到变量 interactions中
            interactions = interactions[:, ind]
        # 采样得到一个符合 Dirichlet 分布的随机变量，表示{当前这个用户}对每个主题的偏好程度
        '这里代表的就是 ——> 用户对应的主题分布多项式（向量）'
        user_topics = pyro.sample(  # (n_users | n_topics)
            Site.user_topics,
            dist.Dirichlet(topic_prior),#先验知识分布
        )

        '到此就已经知道 这里显示的是当前用户下与之交互的项目的主题分布'
        with pyro.plate(Plate.interactions, n_interactions):
            # {当前}用户对所有交互过的物品的主题偏好程度。
            item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                # 最终的偏好取决于用户下的主题分布，
        # 在给定用户与物品之间的交互情况和用户对商品的偏好得分的条件下，通过 Categorical分布（即多项式分布）采样生成用户与物品之间的交互情况
                prefs = topic_items[item_topics]
                interactions = pyro.sample(  # (n_interactions, n_users)
                    Site.interactions,
                    dist.Categorical(logits=prefs),
                    obs=interactions,
                )
    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
    )



# 该代码块实现了一个基于层次贝叶斯模型的推荐系统，
# 具体来说，它首先对一些先验分布进行了采样，如物品的流行度分布、主题的先验分布、用户对物品流行度关注程度的分布等
# 然后根据用户和物品的交互数据进行推断，得到用户对物品的评分值，进而不断更新模型中的参数。
def lda_plus_model(
    interactions: torch.Tensor,
    *,
    n_topics: int, #主题数量
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> ModelData:
    """层次贝叶斯类型模型
    Plate 类表示了各个维度上的循环结构，在这里分别包括主题（n_topics）、用户（n_users）和交互行为（n_interactions）
    Site 类则表示了各个变量所在的分布类型，包括物品流行度分布、主题分布、用户主题偏差分布、用户对物品流行度关注程度分布以及评分值分布
    Args:
        交互:形状的二维数组(n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha
    # 交互数据矩阵 interactions 的第一维长度，即用户和物品之间的交互次数。
    # 后续可以使用这个变量来确定模型中需要重复多少次的计算过程
    n_interactions = interactions.shape[0]

    # 物品最终的项目流行度——>转化成一行矩阵的形式。样本的维度 (1, n_items)
    '物品流行度，我们使用了正态分布——>定义了一个名称为item_pops的随机变量，并且为这个随机变量指定了先验分布'
    '正态分布的先验知识，均值设为0，方差设为2.0'
    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    #主题分布的先验概率——>提供一个合理的先验分布，辅助模型更好地拟合数据和泛化新数据。
    beta_alpha, beta_beta = reparam_beta(alpha, 0.9)#reparam_beta() 函数对 beta 分布的参数进行重新参数化, beta 分布在采样时更加稳定。
    topic_prior = pyro.sample(  # ( | n_topics)——> 维度大小正好和主题K的大小一致
        Site.topic_prior,
        dist.Beta(beta_alpha * torch.ones(n_topics), beta_beta).to_event(1),
    ).unsqueeze(0)

    #  用于描述 {用户流行度一致性的均值} 先验分布的随机变量
    user_pop_devs_prior_mu = pyro.sample(  # ( | 1)
        Site.user_pop_devs_prior_mu, dist.Normal(-0.5 * torch.ones(1), 1.0)
    )
    # 生成了一个半正态分布的样本,用于描述{用户流行度一致性的标准差}
    user_pop_devs_prior_sigma = pyro.sample(  # ( | 1)
        Site.user_pop_devs_prior_sigma, dist.HalfNormal(scale=4.0 * torch.ones(1))
    )
    # 每个主题（topic）对应的物品流行度分布。n_topics 则表示循环的上限次数，即主题的数量。
    '得到的就是主题-单词矩阵，或者就是单词生成多项式'
    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )
    # 模型的核心！！！
    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None: #输入数据中包含用户与物品的交互记录
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            #将其按列索引提取与当前循环的用户相对应的交互行，并保存到变量 interactions中
            interactions = interactions[:, ind]
        # 采样得到一个符合 Dirichlet 分布的随机变量，表示{当前这个用户}对每个主题的偏好程度
        '这里代表的就是 ——> 用户对应的主题分布多项式（向量）'
        user_topics = pyro.sample(  # (n_users | n_topics)
            Site.user_topics,
            dist.Dirichlet(topic_prior),#先验知识分布
        )
        # 用户流行度一致性——>描述每个用户的流行度的重视程度。
        user_pop_devs = pyro.sample(  # (n_users | )
            Site.user_pop_devs,
            dist.LogNormal(user_pop_devs_prior_mu, user_pop_devs_prior_sigma),
        ).unsqueeze(1)
        '到此就已经知道 这里显示的是当前用户下与之交互的项目的主题分布'
        with pyro.plate(Plate.interactions, n_interactions):
            # {当前}用户对所有交互过的物品的主题偏好程度。
            item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                # 最终的偏好取决于用户下的主题分布，
                # 物品的受欢迎程度
                # 用户关心（一致性）的程度
        # 在给定用户与物品之间的交互情况和用户对商品的偏好得分的条件下，通过 Categorical分布（即多项式分布）采样生成用户与物品之间的交互情况
                prefs = topic_items[item_topics] + user_pop_devs * item_pops
                interactions = pyro.sample(  # (n_interactions, n_users)
                    Site.interactions,
                    dist.Categorical(logits=prefs),
                    obs=interactions,
                )

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
    )

# 该函数没有进行用户流行度偏差的采样，而是直接使用指数函数计算其值，并通过 Delta 分布直接设置为其对数的指数。
def guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )

    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )

    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )

        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )

# 这段代码是一个名为 hier_guide 的函数的定义，该函数用于构建 LDA（Latent Dirichlet Allocation）模型的指导器。。
def hier_guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    alpha = 1.0 / n_topics if alpha is None else alpha

    # 商品流行度的均值  通过使用 Pyro 的 param 函数并采样自高斯分布来进行初始化
    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    # 商品流行度标准差
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    # 物品最终的流行度
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))



    init_alpha, init_beta = reparam_beta(alpha, 0.1)
    # 分别表示主题分布的参数 alpha
    topic_prior_alpha = pyro.param(
        Param.topic_prior_alpha,
        lambda: init_alpha * torch.ones(n_topics),
        constraint=dist.constraints.interval(0.05, 1.0),
    )
    # 和参数 beta
    topic_prior_beta = pyro.param(
        Param.topic_prior_beta,
        lambda: init_beta * torch.ones(n_topics),
        constraint=dist.constraints.interval(0.05, 1.0),
    )
    pyro.sample(
        Site.topic_prior, dist.Beta(topic_prior_alpha, topic_prior_beta).to_event(1)
    )

    # LDA模型指导器中的一部分   定义并采样用户流行度偏差的均值和标准差。均值为 -0.5、标准差为 0.5 的正态分布中进行采样来初始化该参数
    user_pop_devs_prior_mu_loc = pyro.param(
        Param.user_pop_devs_prior_mu_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_mu_scale = pyro.param(
        Param.user_pop_devs_prior_mu_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    # 从正态分布中采样用户流行度偏差的均值，并将结果存储在 Pyro 的 Site.user_pop_devs_prior_mu 中。
    pyro.sample(
        Site.user_pop_devs_prior_mu,
        dist.Normal(user_pop_devs_prior_mu_loc, user_pop_devs_prior_mu_scale),
    )
    # LDA模型指导器中的一部分，用于定义并采样用户流行度偏差的标准差。
    user_pop_devs_prior_sigma_loc = pyro.param(
        Param.user_pop_devs_prior_sigma_loc,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_sigma_scale = pyro.param(
        Param.user_pop_devs_prior_sigma_scale,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    # 变换的分布中采样用户流行度偏差的标准差，并将结果存储在 Pyro 的 Site.user_pop_devs_prior_sigma 中。
    pyro.sample(
        Site.user_pop_devs_prior_sigma,
        dist.TransformedDistribution(
            dist.Normal(
                loc=user_pop_devs_prior_sigma_loc, scale=user_pop_devs_prior_sigma_scale
            ),
            transforms=dist.transforms.ExpTransform(),
        ),
    )



    # topic_items_loc 和 topic_items_scale：分别表示主题对应的商品特征的均值和标准差
    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )


    # 表示用户各个主题的概率分布
    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )
    # 表示用户的流行度偏差
    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )
        # 使用Delta区域进行MAP，避免狄利克雷后验的高方差
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )

def hier_var_guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    alpha = 1.0 / n_topics if alpha is None else alpha

    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_prior_p = pyro.param(
        Param.topic_prior_p,
        lambda: torch.ones(1),
        constraint=dist.constraints.interval(0.1, 1.0),
    )
    topic_prior_q = pyro.param(
        Param.topic_prior_q,
        lambda: 0.9 * torch.ones(1),
        constraint=dist.constraints.interval(0.2, 1.0),
    )

    pyro.sample(
        Site.topic_prior,
        dist.Delta(
            (topic_prior_p.log() + topic_prior_q.log() * torch.arange(n_topics)).exp()
        ).to_event(1),
    )

    user_pop_devs_prior_mu_loc = pyro.param(
        Param.user_pop_devs_prior_mu_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_mu_scale = pyro.param(
        Param.user_pop_devs_prior_mu_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_mu,
        dist.Normal(user_pop_devs_prior_mu_loc, user_pop_devs_prior_mu_scale),
    )

    user_pop_devs_prior_sigma_loc = pyro.param(
        Param.user_pop_devs_prior_sigma_loc,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_sigma_scale = pyro.param(
        Param.user_pop_devs_prior_sigma_scale,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_sigma,
        dist.TransformedDistribution(
            dist.Normal(
                loc=user_pop_devs_prior_sigma_loc, scale=user_pop_devs_prior_sigma_scale
            ),
            transforms=dist.transforms.ExpTransform(),
        ),
    )

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )
    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )

        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )


def pred_model(
    user_id: int,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    """像模型一样，只是为了预测

    Args:
        交互:形状的二维数组(n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha

    # omega
    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )

    user_topics = pyro.sample(  # (n_users | n_topics)
        Site.user_topics,
        dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
    )

    user_pop_devs = pyro.sample(  # (n_users | )
        Site.user_pop_devs,
        dist.LogNormal(-0.5 * torch.ones(1), 0.5),
    ).unsqueeze(1)

    item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
        Site.item_topics,
        dist.Categorical(user_topics),
        infer={"enumerate": "parallel"},
    )

    # 最终偏好取决于主题分布，
    # 产品的受欢迎程度和用户关心的程度
    # 商品受欢迎程度
    prefs = topic_items[item_topics] + user_pop_devs * item_pops
    interactions = pyro.sample(  # (n_interactions, n_users)
        Site.interactions,
        dist.Categorical(logits=prefs),
    )

    return interactions


def pred_guide(
    user_id: int,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.zeros(n_users, n_topics),
    )
    user_pop_devs_loc = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    user_pop_devs_scale = pyro.param(
        Param.user_pop_devs_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(n_users), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )

    pyro.sample(
        Site.user_pop_devs,
        dist.LogNormal(
            loc=user_pop_devs_loc[user_id], scale=user_pop_devs_scale[user_id]
        ),
    )

    # use Delta dist for MAP avoiding high variances with Dirichlet posterior
    pyro.sample(
        Site.user_topics,
        dist.Delta(F.softmax(user_topics_logits[user_id], dim=-1), event_dim=1),
    )
