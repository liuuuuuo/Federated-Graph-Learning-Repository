# GraphFL：用于图上半监督节点分类的联邦学习框架

王炳辉，李昂，李海，陈怡然

杜克大学电子与计算机工程系

{binghui.wang, ang.li630, hai.li, yiran.chen}@duke.edu

# 摘要

基于图的半监督节点分类（GraphSSC）在网络、安全、数据挖掘和机器学习等领域有着广泛的应用。然而，现有的集中式GraphSSC方法在解决许多实际图问题时并不现实，因为收集整个图和标注足够数量的标签既耗时又昂贵，同时还可能侵犯数据隐私。联邦学习（FL）是一种新兴的学习范式，能够实现多个客户端之间的协作学习，既能缓解标签稀缺问题，也能保护数据隐私。因此，在FL环境下进行GraphSSC是解决实际图问题的有前景的方案。然而，现有的FL方法存在以下问题：1）当客户端间数据为非IID时表现较差；2）无法处理具有新标签域的数据；3）无法利用未标注数据，而这些问题在实际图问题中普遍存在。

为了解决上述问题，我们提出了第一个用于图上半监督节点分类的FL框架——GraphFL。我们的框架受到元学习方法的启发。具体来说，我们提出了两种GraphFL方法，分别针对图数据的非IID问题和新标签域任务。此外，我们设计了一种自训练方法以利用未标注的图数据。我们采用了代表性的图神经网络作为GraphSSC方法，并在多个图数据集上评估了GraphFL。实验结果表明，GraphFL显著优于对比的FL基线，并且结合自训练的GraphFL能获得更好的性能。

# 1 引言

给定一个图和少量带标签的训练节点，GraphSSC（GraphSSC）旨在预测图中测试节点的标签。GraphSSC有多种应用，如基于图的欺诈检测[40, 39, 3, 10, 32, 14, 29, 33, 1, 18, 24, 34]、属性推断[15, 12]、文档分类[16, 25]等。然而，现有的GraphSSC方法采用集中式学习方式，这使得它们在解决许多实际图问题时不切实际。例如，考虑在社交网络（如Facebook）中检测虚假用户的问题。由单一方收集整个社交图并获得足够数量的带标签节点以执行集中式GraphSSC既耗时又昂贵。当只有部分图或有限标签可用时，现有GraphSSC方法的性能远不能令人满意（见图1和图2）。此外，集中式学习方法需要访问原始数据，可能会侵犯数据隐私。

联邦学习（FL）[22]是一项最近提出的技术，能够实现多个参与方/客户端之间的协作学习。它旨在缓解数据/标签稀缺问题，同时保护数据隐私。在FL环境下，存在多个客户端和一个服务器，每个客户端假设只有有限的带标签数据，并用这些数据训练本地模型；服务器通过以隐私保护的方式聚合本地客户端模型来学习全局模型用于预测。因此，将FL引入GraphSSC是解决实际大规模图问题的有前景的方案。在这种情况下，每个客户端可能拥有原始大图的一个子图，并且只有极少的带标签节点。例如，在Facebook的虚假用户检测中，每个客户端可以是一个移动用户，他只能收集一个自我网络，带标签用户是他自我网络中的已验证Facebook账号。然而，我们注意到，直接应用现有FL方法解决GraphSSC问题面临以下挑战：1）现有FL方法在客户端间数据非独立同分布（non-IID）时表现不佳[22, 41, 19]。而图数据本质上在客户端间高度非IID。例如，从大图中采样和标注代表性节点具有挑战性[17]，不同客户端中有限的带标签节点很难具有相同分布。2）现有FL方法主要关注训练数据和测试数据标签域相同的问题。然而，实际图是动态变化的，新的测试节点类型随时可能出现。例如，社交网络中随时可能出现新类型账号。3）现有FL方法主要用于监督学习，无法利用未标注数据。然而，实际GraphSSC问题通常带标签节点有限，但未标注节点数量巨大。

我们的工作：我们设计了一个新颖的FL框架GraphFL，用于执行基于图的半监督节点分类并解决上述挑战。据我们所知，这是第一个在FL环境下针对GraphSSC问题的工作。我们的框架受到一种最新元学习方法——模型无关元学习（MAML）[8]的启发，该方法在新任务上展现出快速适应能力。给定一组从底层分布采样的任务，MAML学习一个任务无关的初始化，使其在所有任务上经过少量梯度更新后表现良好。

我们观察到MAML非常适合FL环境。具体来说，我们可以将每个任务视为一个客户端，将任务无关初始化视为服务器上学习的全局模型。受此启发，我们提出了两种GraphFL方法，分别针对挑战1和挑战2。为解决挑战1，我们注意到MAML中的任务间数据无需IID，因此提出将MAML引入FL。具体来说，我们的方法分为两个阶段。首先，服务器按照MAML的训练方案学习全局模型，从而缓解非IID图数据带来的问题。然后，我们利用现有FL方法进一步更新全局模型，使其在测试节点上具有良好泛化能力。为解决挑战2，我们提出在FL框架下重构MAML，定义了一个与现有FL方法不同的新目标函数。这样，我们可以在服务器上学习一个共享的全局模型，使其能快速适应标签域不同于训练节点的测试节点。我们进一步提出了一种自训练方法以解决挑战3。具体来说，我们首先在每个客户端用GraphSSC方法和本地标签训练本地模型。然后，用训练好的本地模型预测未标注节点，并选择预测最有信心的未标注节点。将这些节点及其预测标签作为额外带标签节点用于训练GraphSSC方法。

我们采用了两种代表性的图神经网络，即图卷积网络（GCN）[16]和简单图卷积（SGC）[35]，作为GraphSSC方法，并将GraphFL集成到GCN和SGC中以实现联邦半监督节点分类。我们在多个基准图数据集上评估了GraphFL。结果表明，GraphFL在客户端间带标签节点高度非IID时显著优于对比FL基线；GraphFL在处理新标签域测试节点方面优于FL；结合自训练的GraphFL能进一步获得更好性能。我们的贡献总结如下：

- 我们提出了GraphFL，第一个用于图上联邦半监督节点分类的方法。  
- GraphFL解决了图数据的非IID问题；能处理新标签域测试节点；并通过自训练利用未标注节点。  
- 我们在多个图数据集上评估了GraphFL在联邦GraphSSC中的表现，并展示了其优于对比FL基线的优势。

# 2 相关工作

图上的半监督节点分类：基于图的半监督节点分类已被广泛研究，并从网络[40, 3, 32, 31]、安全[39, 10, 11, 14, 30]、数据挖掘[37, 1, 18, 24, 23, 15, 29, 34]、机器学习[43, 38, 16, 28, 35]等多个领域提出了多种方法。传统方法包括标签传播[43, 42]、迭代分类[21]、流形正则化[2]、置信传播[9]等。图神经网络（GNNs）[16, 28, 13, 36, 35]是近年来用于图上半监督节点分类的方法。例如，GCN[16]和SGC是两种代表性GNN。GCN堆叠了可学习的一阶谱滤波器层，受谱图卷积[6]启发，并跟随非线性激活函数。SGC是GCN的变体，去除了GCN层之间的非线性激活函数。SGC计算效率更高，性能与GCN相当。GNN已被证明优于传统方法。本文为简明起见，主要采用GNN作为半监督节点分类方法。

联邦学习（FL）：FL[22, 27, 41, 19, 20]是一种新兴的分布式学习范式，能够协作训练多个客户端模型，并在服务器上维护共享全局模型。FL既能缓解数据/标签稀缺问题，也能保护客户端数据隐私。具体来说，每个客户端假设只有极少带标签数据，并用其训练本地模型。服务器通过以隐私保护的方式聚合本地客户端模型来学习全局模型。例如，最广泛使用的FL方法FedAvg[22]采用平均聚合本地模型。FL的目标是学习一个在所有客户端上表现良好的全局模型。然而，如[41, 19, 20]所示，当客户端间数据为非IID时，现有FL方法难以学习具有良好泛化能力的全局模型。除了非IID问题，现有FL方法无法处理标签域不同的新数据，也无法利用未标注数据。然而，这些问题在实际图上半监督节点分类中普遍存在。

# 3 问题定义与背景

# 3.1 问题定义

假设我们有 $I$ 个客户端 $\mathbb{C} = \{C^{(1)}, C^{(2)}, \dots, C^{(I)}\}$，每个客户端 $C^{(i)}$ 拥有一个图 $^1 G^{(i)} = (\mathbb{V}^{(i)}, \mathbb{E}^{(i)})$，其中 $\mathbb{V}^{(i)}$ 为节点集，$\mathbb{E}^{(i)}$ 为边集。每个节点 $v^{(i)} \in \mathbb{V}^{(i)}$ 关联有特征向量 $\mathbf{x}_{v^{(i)}}$ 和标签 $y_{v^{(i)}}$，标签集合为 $\mathbb{K} = \{1, 2, \dots, K\}$。此外，每个客户端图 $G^{(i)}$ 有一组少量带标签节点 $\mathbb{L}^{(i)} \subset \mathbb{V}^{(i)}$。我们假设每个客户端 $C^{(i)}$ 可以基于其带标签集 $\mathbb{L}^{(i)}$ 和图 $G^{(i)}$ 学习本地半监督节点分类模型 $f_{\theta^{(i)}}$，参数为 $\theta^{(i)}$。我们还考虑一个服务器 $\mathbb{S}$，通过聚合客户端本地模型参数 $\{\theta^{(i)}\}_{i=1}^I$ 学习全局模型参数 $\theta$，但不访问客户端图。现在，假设我们有一组测试节点 $\mathbb{T}$，其标签域可能与训练节点相同或不同。那么，我们的联邦图上半监督节点分类问题定义如下：

定义1：给定 $I$ 个客户端图 $\{G^{(i)}\}_{i = 1}^{I}$ 及其带标签节点 $\{\mathbb{L}^{(i)}\}_{i = 1}^{I}$，一个服务器 $\mathbb{S}$，和一组测试节点 $\mathbb{T}$，我们的目标是基于服务器上学习的全局模型参数 $\theta$（由客户端本地模型参数 $\{\theta_i\}_{i = 1}^I$ 聚合而成）预测测试节点的标签。

设计目标：我们旨在设计一种联邦GraphSSC方法，实现以下三点：1）解决图数据的非IID问题；2）泛化到新标签域的测试节点；3）利用客户端中的未标注节点。接下来，我们介绍启发我们方法设计的模型无关元学习（MAML）[8]。

# 3.2 模型无关元学习（MAML）

给定一组从底层任务分布 $\mathcal{T}$ 中抽取的训练任务 $\{T_i\}$，MAML[8]不是学习一个在所有任务上表现良好的模型，而是学习一个任务无关的初始化 $\theta$，使其在所有任务上经过少量梯度更新后表现良好。具体来说，每个任务 $T_i$ 将其带标签训练集 $\mathbb{L}^{(i)}$ 分为支持集 $\mathbb{L}_S^{(i)}$ 和不相交的查询集 $\mathbb{L}_Q^{(i)}$。然后，MAML进行两级优化：内优化和元优化。在内优化中，对于每个任务 $T_i$，MAML在支持集 $\mathbb{L}_S^{(i)}$ 上用初始化 $\theta$ 训练模型 $f_{\theta}$，输出任务特定的模型参数 $\theta^{(i)}$。然后，MAML将每个 $\theta^{(i)}$ 作为初始化，在相应的查询集 $\mathbb{L}_Q^{(i)}$ 上评估模型 $f_{\theta^{(i)}}$，计算任务特定的损失。在元优化中，MAML同时最小化所有任务查询集上的总损失，以学习任务无关的初始化。形式化地，MAML的目标函数如下：

$$
\min _ {\theta} \mathcal {L} (\theta) = \sum_ {T _ {i} \sim \mathcal {T}} \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} (\theta^ {(i)}) = \sum_ {T _ {i} \sim \mathcal {T}} \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} (\theta - \alpha \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta)), \tag {1}
$$

假设内优化中使用一步梯度下降，$\alpha$ 为学习率，支持集和查询集上的任务特定损失分别定义为：

$$
\mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta) = \frac {1}{| \mathbb {L} _ {S} ^ {(i)} |} \sum_ {(x, y) \in \mathbb {L} _ {S} ^ {(i)}} \ell \left(f _ {\theta} (x), y\right), \tag {2}
$$

$$
\mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} \left(\theta^ {(i)}\right) = \frac {1}{\left| \mathbb {L} _ {Q} ^ {(i)} \right|} \sum_ {(x, y) \in \mathbb {L} _ {Q} ^ {(i)}} \ell \left(f _ {\theta^ {(i)}} (x), y\right), \tag {3}
$$

其中 $\ell (\cdot)$ 为特定任务定义的损失函数。

MAML的目标函数（公式1）通过元学习率 $\beta$ 的梯度下降求解，即：

$$
\theta \leftarrow \theta - \beta \cdot \nabla \mathcal {L} (\theta). \tag {4}
$$

MAML以迭代方式进行，每轮从任务分布 $\mathcal{T}$ 中抽取一批任务进行训练。当新任务到来时，MAML用学习到的任务无关初始化作为初始模型，通过少量梯度下降更新模型。然后，用更新后的模型进行预测。

与FL的联系：我们观察到MAML非常适合FL环境。具体来说，如果将每个任务视为一个客户端，将任务无关初始化视为服务器上学习的全局模型，那么MAML自然适用于FL。受此启发，我们将MAML引入FL，提出GraphFL框架研究基于图的半监督节点分类问题。我们注意到[4, 7]也有类似观察。

# 4 提出的GraphFL框架

我们提出了一种新颖的FL框架用于图上的半监督节点分类，旨在实现上述目标。我们的框架主要将MAML引入FL，我们将其命名为GraphFL。首先，我们开发了两种GraphFL方法，分别针对图数据的非IID问题和新标签域测试节点。然后，我们设计了一种自训练方法以利用客户端图中的未标注节点。

# 4.1 针对非IID图数据的联邦GraphSSC的GraphFL

解决联邦GraphSSC的一个关键挑战是，图数据在客户端间本质上高度非IID，而现有FL方法在非IID数据下表现不佳[22, 41, 19]。这是因为现有FL的目标是协作学习所有客户端的全局模型，但当数据非IID时，学习到的全局模型难以在客户端数据上泛化。我们设计了一种新颖的FL方法GraphFL，用于联邦GraphSSC，能够处理客户端间的非IID图数据。我们的GraphFL方法受到MAML的启发，因为它适用于FL框架，同时训练一组任务且不要求任务间数据IID。具体来说，我们的GraphFL方法将MAML引入FL，并在服务器上学习一个能在测试节点上泛化的全局模型。GraphFL包括两个阶段：阶段I在服务器上按照MAML的训练方案学习全局模型，从而缓解非IID图数据带来的问题。

---
**算法1：GraphFL：非IID图数据**
---

**输入：** 客户端图 {G^(i)}_i，支持节点 {L_s^(i)}_i 和查询节点 {L_q^(i)}_i，服务器上的初始全局模型 θ_0，学习率 α，元学习率 β，#轮数 T，#本地梯度下降步数 T_e，参与客户端比例 ρ。测试节点 T。

---

// 训练  
for episode $t = 0, 1, ..., T-1$ do  
&emsp;Server randomly samples a set $C_t$ of $\rho I$ clients  
&emsp;Server sends the initialized global model $\theta_t$ to clients $C_t$  

&emsp;// 阶段I：MAML  
&emsp;// 客户端更新  
&emsp;for each client $C^{(i)} \in C_t$ do  
&emsp;&emsp;$\hat{\theta}_t^{(i)} \leftarrow \theta_t$  
&emsp;&emsp;for epoch $t = 1, 2, ..., T_e$ do  
&emsp;&emsp;&emsp;$\hat{\theta}_t^{(i)} \leftarrow \hat{\theta}_t^{(i)} - \alpha \cdot \nabla L_s^{(i)}(\hat{\theta}_t^{(i)})$  
&emsp;&emsp;$g_i \leftarrow \nabla_{\theta} L_q^{(i)}(\hat{\theta}_t^{(i)})$  

&emsp;// 服务器更新  
&emsp;$\theta_t \leftarrow \theta_t - \beta \cdot \sum_{i \in C_t} g_i$  

&emsp;// 阶段II：FL  
&emsp;// 客户端微调  
&emsp;for each client $C^{(i)} \in C_t$ do  
&emsp;&emsp;$\hat{\theta}_t^{(i)} \leftarrow \theta_t$  
&emsp;&emsp;for each epoch $t = 1, 2, ..., T_e$ do  
&emsp;&emsp;&emsp;$\hat{\theta}_t^{(i)} \leftarrow \hat{\theta}_t^{(i)} - \alpha \cdot \nabla L_s^{(i)}(\hat{\theta}_t^{(i)})$  

&emsp;// 服务器聚合 (FedAvg)  
&emsp;$\theta_{t+1} \leftarrow \frac{1}{|C_t|} \sum_{C^{(i)} \in C_t} \hat{\theta}_t^{(i)}$

---

// 测试

使用全局模型 θ_T 预测测试节点 T 的标签

---

阶段II利用现有FL方法进一步更新全局模型，使其在所有客户端上具有良好泛化能力。

我们首先定义一些符号。在每个客户端 $C^{(i)}$ 中，我们将训练集 $\mathbb{L}^{(i)}$ 分为支持节点 $\mathbb{L}_S^{(i)}$ 和查询节点 $\mathbb{L}_Q^{(i)}$。假设在第 $t$ 轮，服务器 $\mathbb{S}$ 具有全局模型参数 $\theta_t$，每个客户端 $C^{(i)}$ 具有本地模型参数 $\theta_t^{(i)}$。我们定义客户端 $C^{(i)}$ 中支持节点 $\mathbb{L}_S^{(i)}$ 上的损失为 $\mathcal{L}_{\mathbb{L}_S^{(i)}}(\theta_t) = \frac{1}{|\mathbb{L}_S^{(i)}|}\sum_{v^{(i)}\in \mathbb{L}_S^{(i)}}\ell (f_{\theta_t}(\mathbf{x}_v^{(i)},G^{(i)}),y_v^{(i)})$，查询节点 $\mathbb{L}_Q^{(i)}$ 上的损失为 $\mathcal{L}_{\mathbb{L}_Q^{(i)}}(\theta_t^{(i)}) = \frac{1}{|\mathbb{L}_Q^{(i)}|}\sum_{v^{(i)}\in \mathbb{L}_Q^{(i)}}\ell (f_{\theta_t^{(i)}}(\mathbf{x}_v^{(i)},G^{(i)}),y_v^{(i)})$，其中 $f_{\theta_t}$ 和 $f_{\theta_t^{(i)}}$ 分别为在客户端图 $G^{(i)}$ 上支持节点和查询节点上学习的GraphSSC模型。此外，假设在第 $t$ 轮，服务器已学习到全局模型参数 $\theta_t$。然后，我们的GraphFL方法包括以下步骤。

步骤I：服务器随机将 $\theta_t$ 发送给总客户端数的 $\rho$，记为 $\mathbb{C}_t$。

步骤II：每个参与客户端 $C^{(i)}$ 在 $\mathbb{C}_t$ 中首先通过最小化支持节点 $\mathbb{L}_S^{(i)}$ 上的损失学习本地模型参数 $\theta_t^{(i)}$。

---
**算法2：GraphFL：新标签域**
---

**输入：** 客户端图 {G^(i)}_i，支持节点 {L_s^(i)}_i 和查询节点 {L_q^(i)}_i，服务器上的初始全局模型 θ_0，学习率 α，元学习率 β，#轮数 T，#本地梯度下降步数 T_e，参与客户端比例 ρ。测试节点 T。

---

// 训练  
for episode $t = 0, 1, ..., T-1$ do  
&nbsp;&nbsp;&nbsp;&nbsp;服务器随机抽取一组 $\rho I$ 客户端 $C_t$  
&nbsp;&nbsp;&nbsp;&nbsp;服务器将全局模型 $\theta_t$ 发送给客户端 $C_t$  

&nbsp;&nbsp;&nbsp;&nbsp;// 客户端更新  
&nbsp;&nbsp;&nbsp;&nbsp;for each client $C^{(i)} \in C_t$ do  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{\theta}_t^{(i)} \leftarrow \theta_t$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for epoch $t = 1, 2, ..., T_e$ do  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{\theta}_t^{(i)} \leftarrow \hat{\theta}_t^{(i)} - \alpha \cdot \nabla L_s^{(i)}(\hat{\theta}_t^{(i)})$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\hat{\theta}_t^{(i)} \leftarrow \hat{\theta}_t^{(i)} - \beta(1 - \alpha \nabla^2 L_s^{(i)}(\hat{\theta}_t^{(i)})) \nabla L_q^{(i)}(\hat{\theta}_t^{(i)})$  

&nbsp;&nbsp;&nbsp;&nbsp;// 服务器聚合（FedAvg）  
&nbsp;&nbsp;&nbsp;&nbsp;$\theta_{t+1} \leftarrow \frac{1}{|C_t|} \sum_{C^{(i)} \in C_t} \hat{\theta}_t^{(i)}$  

---

// 测试

将全局模型 θ_T 作为初始模型，用少量新标签域的带标签节点更新模型  
使用更新后的模型预测测试节点 T 的标签

---

通过最小化支持节点 $\mathbb{L}_S^{(i)}$ 上的损失进行梯度下降。假设一步梯度下降，我们有：

$$
\theta_ {t} ^ {(i)} \leftarrow \theta_ {t} - \alpha \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta_ {t}), \tag {5}
$$

接下来，每个客户端 $C^{(i)}$ 在查询节点 $\mathbb{L}_Q^{(i)}$ 上评估本地模型参数 $\theta_t^{(i)}$，获得损失梯度 $\nabla_{\theta}\mathcal{L}_{\mathbb{L}_Q^{(i)}}(\theta_t^{(i)})$，并将梯度发送给服务器。

步骤III：服务器利用所有参与客户端 $\mathbb{C}_t$ 的梯度，通过梯度下降更新全局模型 $\theta_t$ 为 $\hat{\theta}_t$。假设一步梯度下降，我们有：

$$
\hat {\theta} _ {t} \leftarrow \theta_ {t} - \beta \nabla_ {\theta} \sum_ {C ^ {(i)} \in \mathbb {C} _ {t}} \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} \left(\theta_ {t} ^ {(i)}\right). \tag {6}
$$

通过遵循MAML的训练方案，服务器学习到一个能缓解图数据非IID问题的全局模型。接下来，我们采取以下两步进一步更新全局模型，使其在所有客户端上具有良好泛化能力。

步骤IV：每个参与客户端 $C^{(i)}$ 在 $\mathbb{C}_t$ 中下载全局模型参数 $\hat{\theta}_t$，并在支持节点上通过梯度下降微调本地模型。假设一步梯度下降，我们有：

$$
\hat {\theta} _ {t} ^ {(i)} \leftarrow \hat {\theta} _ {t} - \alpha \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\hat {\theta} _ {t}). \tag {7}
$$

步骤V：服务器采用现有FL方法，如FedAvg[22]，更新全局模型。

$$
\theta_ {t + 1} \leftarrow \frac {1}{| \mathbb {C} _ {t} |} \sum_ {C ^ {(i)} \in \mathbb {C} _ {t}} \hat {\theta} _ {t} ^ {(i)}. \tag {8}
$$

经过若干轮后，最终全局模型用于预测测试节点 $\mathbb{T}$。算法1详细描述了我们针对非IID图数据的联邦GraphSSC的GraphFL方法。

# 4.2 针对新标签域的联邦GraphSSC的GraphFL

现有FL主要针对所有客户端数据标签域相同的问题。然而，实际图是动态变化的，新的节点类型随时可能出现。本节我们研究训练节点和测试节点标签域不同的图问题。一种可能的解决方案是利用迁移学习。具体来说，我们首先基于现有FL方法（如FedAvg[22]）在服务器上学习全局模型。然后，将全局模型作为初始模型，并用少量新标签域的带标签节点进行微调。最后，用微调后的模型预测新标签域测试节点的标签。然而，如我们的实验结果所示（见表2），这种基于迁移学习的方案效果并不理想。

我们设计了一种新颖的GraphFL方法，用于GraphSSC，能够泛化到新标签域的测试节点。具体来说，我们提出在FL框架下重构MAML，旨在在服务器上学习一个共享的全局模型，使每个客户端经过少量梯度更新后在其特定GraphSSC损失下表现良好。形式化地，我们定义目标函数如下：

$$
\min _ {\theta} \mathcal {L} (\theta) = \frac {1}{I} \sum_ {i = 1} ^ {I} \mathcal {L} _ {i} (\theta) = \frac {1}{I} \sum_ {i = 1} ^ {I} \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} (\theta - \alpha \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta)), \tag {9}
$$

其中 $\theta$ 是我们要学习的共享初始化；$\mathcal{L}_i(\theta)$ 是客户端 $C^{(i)}$ 上定义的损失。注意，我们的损失函数与现有FL方法[22, 41, 19]完全不同。

我们现在在FL框架下求解公式（9）。具体来说，我们首先基于定义的客户端损失更新本地模型，然后通过聚合本地模型更新全局模型。假设在第 $t$ 轮，服务器具有全局模型参数 $\theta_t$。然后，GraphFL包括以下步骤：

步骤I：服务器随机将 $\theta_t$ 发送给总客户端数的 $\rho$，记为 $\mathbb{C}_t$。

步骤II：每个参与客户端 $C^{(i)}$ 在 $\mathbb{C}_t$ 中通过最小化客户端损失 $\mathcal{L}_i(\theta_t)$ 学习本地模型参数 $\theta_t^{(i)}$。具体来说，基于全局模型参数 $\theta_t$，每个客户端损失通过梯度下降最小化。假设一步梯度下降，我们有：

$$
\theta_ {t} ^ {(i)} \leftarrow \theta_ {t} - \beta \cdot \nabla \mathcal {L} _ {i} \left(\theta_ {t}\right), \tag {10}
$$

其中 $\nabla \mathcal{L}_i(\theta_t)$ 计算如下：

$$
\nabla \mathcal {L} _ {i} (\theta_ {t}) = (\mathbb {I} - \alpha \cdot \nabla^ {2} \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta_ {t})) \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} (\theta_ {t} - \alpha \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta_ {t})).
$$

公式（10）可分两步求解。首先，客户端 $C^{(i)}$ 通过在带标签支持节点 $\mathbb{L}_S^{(i)}$ 上运行梯度下降获得中间模型参数 $\hat{\theta}_t^{(i)}$。假设一步梯度下降，我们有：

$$
\hat {\theta} _ {t} ^ {(i)} \leftarrow \theta_ {t} - \alpha \cdot \mathcal {L} _ {\mathbb {L} _ {S} ^ {(i)}} (\theta_ {t}). \tag {11}
$$

接下来，每个客户端 $C^{(i)}$ 用带标签查询节点 $\mathbb{L}_Q^{(i)}$ 更新本地模型参数 $\theta_t^{(i)}$。形式化地，

$$
\theta_ {t} ^ {(i)} \leftarrow \theta_ {t} - \beta (\mathbb {I} - \alpha \nabla^ {2} \mathcal {L} _ {\mathbb {L} _ {S}} (\theta_ {t}) \cdot \nabla \mathcal {L} _ {\mathbb {L} _ {Q} ^ {(i)}} (\hat {\theta} _ {t} ^ {(i)}). \tag {12}
$$

步骤III：服务器通过最小化参与客户端 $\mathbb{C}_t$ 的损失，使用梯度下降更新全局模型 $\theta_{t + 1}$。即，

$$
\theta_ {t + 1} \leftarrow \theta_ {t} - \frac {\beta}{| \mathbb {C} _ {t} |} \sum_ {C ^ {(i)} \in \mathbb {C} _ {t}} \nabla \mathcal {L} _ {i} \left(\theta_ {t}\right) = \frac {1}{| \mathbb {C} _ {t} |} \sum_ {C ^ {(i)} \in \mathbb {C} _ {t}} \theta_ {t} ^ {(i)}, \tag {13}
$$

其中我们在最后一个公式中代入公式（10）。公式（13）表明，服务器通过平均参与客户端的本地模型参数更新全局模型参数，该聚合规则与FedAvg[22]完全相同。

通过若干轮求解公式（9），我们在服务器上学习到一个全局半监督节点分类模型，能够快速适应新标签域的测试节点。具体来说，我们用全局模型作为初始模型，通过少量梯度下降用新标签域的少量带标签节点更新模型。然后，我们采用更新后的模型预测新标签域的测试节点标签。算法2详细描述了我们针对新标签域的联邦GraphSSC的GraphFL方法。

# 4.3 通过自训练利用未标注节点

现有FL方法主要用于监督学习，即只利用带标签数据。然而，实际图问题通常带标签节点有限，但未标注节点数量巨大。一个自然的问题是，如何利用未标注节点进一步提升GraphFL的性能？

我们提出了一种自训练方法以利用客户端图中的未标注节点。具体来说，给定一种基于图的半监督节点分类方法，我们首先在每个客户端用极少带标签节点训练本地模型。然后，在每个客户端用本地模型预测未标注节点，并选择预测最有信心的一部分未标注节点。将这些节点的预测标签作为伪标签，并将每个客户端选中的节点（及其伪标签）加入训练集。最后，我们在增强后的训练集上训练GraphFL方法，实现联邦半监督节点分类。我们的实验结果（见图5）表明，该方法简单但有效。

表1：数据集统计信息。  

<table><tr><td>数据集</td><td>Cora</td><td>Citeseer</td><td>Coauthor CS</td><td>Amazon2M</td></tr><tr><td>特征数</td><td>1,433</td><td>3,703</td><td>6,805</td><td>100</td></tr><tr><td>节点数</td><td>2,485</td><td>2,110</td><td>18,333</td><td>2,449,029</td></tr><tr><td>边数</td><td>5,069</td><td>3,668</td><td>81,894</td><td>61,859,140</td></tr><tr><td>类别数</td><td>7</td><td>6</td><td>15</td><td>47</td></tr></table>

# 5 实验评估

# 5.1 实验设置

# 5.1.1 数据集

我们选用四个基准图数据集：Cora、Citeseer、Coauthor CS和Amazon2M，这些数据集在以往节点分类工作中被广泛使用[16, 26, 5]。Cora和Citeseer[25]是引文图，每个节点表示一篇文档，两个文档间有边表示引用关系。每个节点的特征向量为对应文档的词袋特征，每个文档有一个标签。Coauthor CS[26]是基于KDD Cup 2016挑战赛的Microsoft Academic Graph构建的合作作者图。该图中每个节点表示一位作者，两个作者间有边表示他们共同发表过论文。每个节点的特征向量表示作者论文的关键词，节点标签表示作者最活跃的研究领域。Amazon2M是[5]构建的大规模图数据集，每个节点为一个商品，边表示两个商品被一起购买。每个节点的特征向量为商品描述的词袋特征。这些数据集的统计信息见表1。我们注意到这些数据集在图规模、平均节点度、类别数等方面各不相同。

# 5.1.2 对比学习方法

我们采用两种代表性GNN：GCN[16]和SGC[35]作为GraphSSC方法。我们将GraphFL与个体学习（IL）和FL[22]进行对比。

- 个体学习（IL）：该方法仅有客户端。每个客户端采用GraphSSC方法，用极少带标签节点训练本地节点分类模型。然后，每个本地模型用于分类测试节点并获得分类准确率。最终准确率为所有客户端的平均值。  
- 联邦学习（FL）[22]：该方法有中心服务器和多个客户端。每个客户端有极少带标签节点，服务器无法访问这些标签。在每一轮，服务器初始化全局模型并发送给选中的客户端；每个客户端用全局模型和极少带标签节点采用GraphSSC方法训练本地模型；服务器通过平均选中客户端的本地模型参数更新全局模型。若干轮后，服务器用最终全局模型对测试节点分类。

# 5.1.3 训练集与测试集

针对图数据非IID问题的实验，我们从每个数据集的每个类别随机采样80个节点组成训练集，并均匀分配给所有客户端（实验中为50个客户端）。每个客户端分配到的带标签节点极少（9到70个），这意味着客户端间带标签节点高度非IID。训练集进一步分为两半，第一部分用于对比方法的训练节点或我们方法的支持节点，第二部分用于对比方法的超参数调优或我们方法的查询节点。考虑到数据集图规模不同，我们在Cora和Citeseer中随机选取1000个节点，在Coauthor CS和Amazon2M中选取10000个节点作为测试集。训练集和测试集各采样5次，最终结果为平均分类准确率。

针对新标签域测试节点的实验，我们将每个图数据集的 $K$ 个类别节点分为两组。第一组包含前 $K-K_0$ 个类别的节点，第二组包含剩余 $K_0$ 个类别的节点。考虑到不同数据集类别数不同，我们在不同数据集设定不同的 $K_0$。每个客户端从第一组采样 $K_0$ 个类别，每类采样10个节点组成训练集。训练集进一步分为支持节点和查询节点（我们方法）或训练节点和验证节点（对比FL方法）。我们还从第二组为每个客户端采样 $K_0$ 个类别，每类采样与支持节点相同数量的节点用于快速适应/微调，并采样20个节点作为测试节点。

# 5.1.4 参数设置

在我们的方法中，总客户端数 $I=50$，轮数 $T=50$，本地梯度下降步数 $T_e=15$。全局模型 $\theta_0$ 随机初始化。在新标签域测试节点实验中，Cora和Citeseer设 $K_0=2$，Coauthor CS和Amazon2M设 $K_0=3$。有三个关键参数影响方法性能：每轮参与客户端比例、训练集带标签节点数、客户端图间重叠比例。默认情况下，每轮20%客户端参与，训练集每类80个带标签节点，所有客户端拥有完整图。研究某参数影响时，其余参数保持默认。

# 5.2 非IID图数据下的节点分类结果

本节我们评估并对比个体学习（IL）、FL和GraphFL在客户端间图数据非IID情况下的联邦GraphSSC表现。

![alt text](image.png)

图1：GCN在对比学习方法下，随每类带标签节点数变化的节点分类准确率。  

![alt text](image-1.png)

图2：SGC在对比学习方法下，随每类带标签节点数变化的节点分类准确率。

# 5.2.1 每类带标签节点数的影响

图1和图2分别展示了GCN和SGC在四个数据集上，采用对比学习方法时节点分类准确率随每类带标签节点数变化的情况。主要观察如下：首先，IL在所有情况下表现最差。这是因为IL中每个客户端只能利用自身带标签节点，而其他两种方法能利用所有客户端的标签信息。其次，GraphFL始终优于FL。这是因为GraphFL采用了MAML的训练方案，更好地处理了图数据的非IID问题。值得注意的是，当每类带标签节点数较少（如每类10个），即非IID更严重时，GraphFL与FL的准确率差距更大，约为20%。下文主要对比FL和GraphFL以简明起见。

# 5.2.2 每轮参与客户端比例的影响

图3展示了GCN和SGC在四个数据集上，采用FL和GraphFL时节点分类准确率随每轮参与客户端比例变化的情况。首先，随着每轮参与客户端比例增加，FL和GraphFL都能提升GCN和SGC的准确率。这是因为两种方法在更多客户端参与时能利用更多带标签节点。其次，GraphFL在所有数据集上均优于FL。此外，当参与客户端比例较小时，GraphFL的性能提升更大。原因可能是参与客户端较少时，客户端间带标签节点分布更非IID。

![alt text](image-2.png)

图3：GCN和SGC在FL和GraphFL下，随每轮参与客户端比例变化的节点分类准确率。

![alt text](image-3.png)

图4：GCN和SGC在FL和GraphFL下，随客户端图间重叠比例变化的节点分类准确率。

# 5.2.3 客户端图间重叠比例的影响

本实验模拟实际场景，每个客户端仅拥有大图的一个子图。为简化模拟，我们用客户端图间重叠节点比例作为指标。具体来说，给定一个大图和重叠比例 $\gamma$，我们将所有节点均匀分配给所有客户端，使相邻索引的两个客户端有 $\gamma$ 的重叠节点。每个客户端图由分配到的节点及其相关边组成，忽略与其他客户端节点相关的边。考虑到Cora和Citeseer图规模较小，我们仅在Coauthor CS和Amazon2M上进行实验。

图4展示了GCN和SGC在Coauthor CS和Amazon2M上，采用FL和GraphFL时节点分类准确率随客户端图间重叠比例变化的情况。主要观察如下：首先，随着图间重叠比例增加，FL和GraphFL在GCN和SGC上的节点分类准确率均提升。其次，GraphFL始终优于FL。此外，当客户端图无重叠时，GraphFL相较FL的性能提升最大。原因在于此时客户端间图数据最非IID。

表2：GCN和SGC在新标签域下采用FL和GraphSSC时，随每类带标签节点数变化的准确率。  

<table><tr><td>GCN</td><td colspan="3">Cora</td><td colspan="3">CiteSeer</td><td colspan="3">Coauthor CS</td><td colspan="3">Amazon2M</td></tr><tr><td>每类标签数</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td></tr><tr><td>FL+TL</td><td>0.527</td><td>0.623</td><td>0.667</td><td>0.500</td><td>0.503</td><td>0.527</td><td>0.664</td><td>0.667</td><td>0.711</td><td>0.614</td><td>0.626</td><td>0.637</td></tr><tr><td>GraphFL</td><td>0.667</td><td>0.767</td><td>0.843</td><td>0.620</td><td>0.630</td><td>0.670</td><td>0.774</td><td>0.835</td><td>0.889</td><td>0.699</td><td>0.706</td><td>0.764</td></tr><tr><td>SGC</td><td colspan="3">Cora</td><td colspan="3">CiteSeer</td><td colspan="3">Coauthor CS</td><td colspan="3">Amazon2M</td></tr><tr><td>每类标签数</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td><td>2</td><td>6</td><td>10</td></tr><tr><td>FL+TL</td><td>0.500</td><td>0.517</td><td>0.543</td><td>0.453</td><td>0.483</td><td>0.494</td><td>0.607</td><td>0.647</td><td>0.687</td><td>0.601</td><td>0.614</td><td>0.626</td></tr><tr><td>GraphFL</td><td>0.647</td><td>0.710</td><td>0.773</td><td>0.583</td><td>0.603</td><td>0.653</td><td>0.740</td><td>0.806</td><td>0.862</td><td>0.697</td><td>0.702</td><td>0.757</td></tr></table>

# 5.3 新标签域下的节点分类结果

本节我们评估GraphFL，并与FL对比在新标签域测试节点下的联邦半监督节点分类表现。由于现有FL无法处理新标签域节点，我们采用上述迁移学习（TL）方案。仅展示不同每类带标签节点数下的结果。我们在参与客户端比例和客户端图间重叠比例影响方面也有类似观察。

表2展示了GCN和SGC在四个数据集上，采用FL和GraphFL在新标签域下随每类带标签节点数变化的节点分类准确率。我们观察到，GraphFL始终优于FL，几乎所有情况下准确率提升至少10%。这表明迁移学习在联邦GraphSSC中处理新标签域测试节点效果有限。此外，随着每类带标签节点数增加，GraphSSL相较FL+TL的性能提升更大。

# 5.4 自训练下的节点分类结果

本实验研究采用自训练进一步提升GraphFL性能的有效性。

图5展示了GCN和SGC在GraphFL下，随每类自训练生成伪标签节点数变化的节点分类准确率（由于篇幅限制，未展示新标签域下结果，但有类似观察）。首先，采用伪标签节点训练时，所有数据集上的准确率均提升。这表明伪标签节点对联邦训练有益，能进一步提升半监督节点分类性能。原因在于大部分伪标签节点的预测标签与真实标签一致。其次，随着每类伪标签节点数增加，准确率先升后降。这是因为伪标签节点较少时，大部分标签预测正确；但数量过多时，错误标签也随之增多。表3还展示了GCN自训练在四个数据集上伪标签节点预测正确的比例，验证了上述结论。

![alt text](image-4.png)

图5：GCN和SGC在GraphFL下，随每类伪标签节点数变化的节点分类准确率。

表3：GCN自训练在四个数据集上伪标签节点预测正确的比例。  

<table><tr><td>伪标签节点数</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td>Cora</td><td>90%</td><td>85%</td><td>81%</td><td>69%</td></tr><tr><td>Citeseer</td><td>90%</td><td>80%</td><td>72%</td><td>64%</td></tr><tr><td>Coauthor CS</td><td>95%</td><td>90%</td><td>83%</td><td>74%</td></tr><tr><td>Amazon2M</td><td>83%</td><td>76%</td><td>73%</td><td>69%</td></tr></table>

# 5.5 小结

- 我们的方法在处理非IID图数据和泛化到新标签域测试节点方面，显著优于传统FL的联邦GraphSSC。  
- 通过自训练利用未标注节点，GraphFL性能可进一步提升。  
- 增加训练节点数、每轮参与客户端比例、客户端图规模或利用未标注节点，均可提升联邦图节点分类性能。

# 6 结论

我们研究了图上的联邦半监督节点分类，并提出了第一个联邦学习框架GraphFL。在FL环境下，基于图的半监督节点分类面临独特挑战，如1）客户端间图数据高度非IID；2）测试节点与训练节点标签域可能不同；3）客户端图中未标注节点众多。我们提出了两种受MAML启发的GraphFL方法，分别解决1）和2）问题。此外，我们设计了一种自训练方法解决3）问题。我们在两种代表性图神经网络上评估了GraphFL在联邦半监督节点分类中的表现。多数据集实验结果表明，我们的方法显著优于对比基线，结合自训练后性能进一步提升。

# 参考文献

[1] Leman Akoglu, Rishi Chandy, and Christos Faloutsos. Opinion fraud detection in online reviews by network effects. In ICWSM, 2013.  
[2] Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani. Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. JMLR, 2006.  
[3] Qiang Cao, Michael Sirivianos, Xiaowei Yang, and Tiago Pregueiro. Aiding the detection of fake accounts in large scale social online services. In NSDI, 2012.  
[4] Fei Chen, Mi Luo, Zhenhua Dong, Zhenguo Li, and Xiuqiang He. Federated meta-learning with fast convergence and efficient communication. arXiv, 2018.  
[5] Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gen: An efficient algorithm for training deep and large graph convolutional networks. In KDD, 2019.  
[6] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In NIPS, 2015.  
[7] Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. Personalized federated learning: A meta-learning approach. In NeurIPS, 2020.  
[8] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML, 2017.  
[9] Wolfgang Gatterbauer, Stephan Gunnemann, Danai Koutra, and Christos Faloutsos. Linearized and single-pass belief propagation. VLDB, 2015.  
[10] Neil Zhenqiang Gong, Mario Frank, and Prateek Mittal. Sybil belief: A semi-supervised learning approach for structure-based sybil detection. IEEE TIFS, 2014.  
[11] Neil Zhenqiang Gong and Bin Liu. You are who you know and how you behave: Attribute inference attacks via users' social friends and behaviors. In *Usenix Security*, 2016.  
[12] Neil Zhenqiang Gong and Bin Liu. Attribute inference attacks in online social networks. ACM TOPS, 2018.  
[13] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NIPS, 2017.  
[14] Jinyuan Jia, Binghui Wang, and Neil Zhenqiang Gong. Random walk based fake account detection in online social networks. In IEEE DSN, 2017.  
[15] Jinyuan Jia, Binghui Wang, Le Zhang, and Neil Zhenqiang Gong. Attriinfer: Inferring user attributes in online social networks using markov random fields. In WWW, 2017.  
[16] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2017.

[17] Jure Leskovec and Christos Faloutsos. Sampling from large graphs. In KDD, 2006.  
[18] Huayi Li, Zhiyuan Chen, Bing Liu, Xiaokai Wei, and Jidong Shao. Spotting fake reviews via collective positive-unlabeled learning. In ICDM, 2014.  
[19] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. arXiv, 2019.  
[20] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. In ICLR, 2020.  
[21] Qing Lu and Lise Getoor. Link-based classification. In ICML, 2003.  
[22] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017.  
[23] Thai Pham and Steven Lee. Anomaly detection in bitcoin network using unsupervised learning methods. arXiv, 2016.  
[24] Shebuti Rayana and Leman Akoglu. Collective opinion spam detection: Bridging review networks and metadata. In KDD, 2015.  
[25] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 2008.  
[26] Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Gunnemann. Pitfalls of graph neural network evaluation. In NeurIPS R2L Workshop, 2018.  
[27] Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. In NIPS, 2017.  
[28] Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In ICLR, 2018.  
[29] Binghui Wang, Neil Zhenqiang Gong, and Hao Fu. GANG: Detecting fraudulent users in online social networks via guilt-by-association on directed graphs. In ICDM, 2017.  
[30] Binghui Wang, Jinyuan Jia, and Neil Zhenqiang Gong. Graph-based security and privacy analytics via collective classification with joint weight learning and propagation. In  $NDSS$ , 2019.  
[31] Binghui Wang, Jinyuan Jia, Le Zhang, and Neil Zhenqiang Gong. Structure-based sybil detection in social networks via local rule-based propagation. IEEE TNSE, 2019.  
[32] Binghui Wang, Le Zhang, and Neil Zhenqiang Gong. Sybilcar: Sybil detection in online social networks via local rule based propagation. In INFOCOM, 2017.  
[33] Guan Wang, Sihong Xie, Bing Liu, and S Yu Philip. Review graph based online store review spammer detection. In ICDM, 2011.

[34] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom Robinson, and Charles E Leiserson. Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. In KDD Workshop, 2019.  
[35] Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr, Christopher Fifty, Tao Yu, and Kilian Q Weinberger. Simplifying graph convolutional networks. In ICML, 2019.  
[36] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In ICML, 2018.  
[37] Chao Yang, Robert Harkreader, Jialong Zhang, Seungwon Shin, and Guofei Gu. Analyzing spammers' social networks for fun and profit: a case study of cyber criminal ecosystem on twitter. In WWW, 2012.  
[38] Zhilin Yang, William W Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In ICML, 2016.  
[39] H. Yu, P. B. Gibbons, M. Kaminsky, and F. Xiao. SybilLimit: A near-optimal social network defense against Sybil attacks. In IEEE S & P, 2008.  
[40] H. Yu, M. Kaminsky, P. B. Gibbons, and A. Flaxman. SybilGuard
