
$$
\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)
$$


$$
\mathbf{H} = \left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}}\right)^{K} \mathbf{X} \mathbf{W}
$$

$w_{t+1}$ ← $∑_{k∈S_t} (n_k/n) w_{t+1}^k$



算法核心流程
text

服务器执行：
初始化全局模型 $w_0$  
for 每轮通信 t = 1, 2, ... do  
      随机选择一部分客户端 $S_t$  
      向每个客户端发送当前全局模型 $w_t$  
      并行执行：  
          for 每个客户端 k ∈ $S_t$ do  
              $w_{t+1}^k$ ← ClientUpdate(k, $w_t$)  
      服务器聚合：$w_{t+1}$ ← $∑_{k∈S_t} (n_k/n) w_{t+1}^k$


客户端 k 执行：  
输入：全局模型 w，本地数据 $D_k$  
本地模型初始化：$w_k$ ← w  
for 多个本地epoch do  
    在本地数据 $D_k$ 上执行SGD更新  
return $w_k$ 给服务器