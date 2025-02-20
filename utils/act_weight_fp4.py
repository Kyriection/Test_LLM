# import math
# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch.nn.parameter import Parameter
# import pdb


# def fp4_fake_quantize(
#     x: torch.Tensor,
#     e: int = 1, 
#     m: int = 2,
#     b: int = 1,
#     topk= 150,  # Top-K 绝对值最大元素单独处理的数量
#     group_size=256,
#     floor_t=True
# ): 
#     x = x.clone()
#     # group_size = int(x.shape[1])
#     if topk<1:
#         topk=group_size*topk
#     topk=int(topk)
#     x_shape = x.shape
#     x = x.view(-1,group_size) # group_size = 256
    
#     if topk > 0:
#         xabs = x.abs()      # 存储x中元素绝对值
#         xabs_topk_index = xabs.topk(topk, dim=-1).indices #找到topk元素的索引
#         topk_values = torch.gather(x, 1 , xabs_topk_index)#把topk元素提取出来
#         x[torch.arange(0, x.size(0), device=x.device)[:, None].expand(-1, topk), xabs_topk_index] = 0 # 把topk元素置为0
#     alpha = x.abs().max(dim=-1).values.clamp(min=1e-6) #############SCALE FACTOR？？？###################
#     q_max = alpha                                           #############SCALE FACTOR？？？###################
#     q_min = -q_max                                          #############SCALE FACTOR？？？###################
#     x_clamped = torch.clamp(x, q_min[:, None], q_max[:, None])
#     alpha_hat = alpha * (2**(-b))
#     b_hat = 2**e - torch.log2(q_max) + torch.log2(torch.tensor(2 - 2**(-m), dtype=torch.float32)) - 1
#     log_v = torch.floor(torch.log2(torch.abs(x_clamped) + 1e-8) + b_hat.unsqueeze(1))
#     v = torch.pow(2, torch.clamp(log_v - m, min=1-m))
#     if topk > 0:
#         if floor_t:
#             x = alpha_hat.unsqueeze(1) * v * torch.floor(x_clamped / (alpha_hat.unsqueeze(1) * v+1e-12) )
#         else:
#             x = alpha_hat.unsqueeze(1) * v * torch.round(x_clamped / (alpha_hat.unsqueeze(1) * v+1e-12) )

#         row_indices = torch.arange(0, x.size(0), device=x.device).view(-1, 1).expand_as(xabs_topk_index)  # [1024, 2]
#         x[row_indices, xabs_topk_index] = topk_values
#         # row_indices = (
#         #         torch.arange(x.size(0), device=x.device)
#         #         .view(-1, 1)
#         #         .expand(-1, topk)
#         #     )        
#         # # Restore the randomly selected elements
#         # x[row_indices, random_indices.unsqueeze(0).expand(x.size(0), -1)] = topk_values

#     x = x.view(*x_shape)
#     return x


# import torch

# # def fp4_fake_quantize(
# #     x: torch.Tensor,
# #     e: int = 1, 
# #     m: int = 2,
# #     b: int = 1,
# #     topk: int = 100  # Top-K 绝对值最大元素单独处理的数量
# # ): 
# #     # 复制一份原始 x，避免在原张量上做 in-place 操作
# #     x = x.clone()
# #     x_shape = x.shape
    
# #     # 展平到 (N, 256) 的形式，以便于后续 group-based 的操作
# #     # 这里假设 x.size 在最后一个维度是 256，也就是 group_size = 256
# #     x = x.view(-1, 256)
    
# #     if topk > 0:
# #         # 1. 找到绝对值最大的 topk 的索引（按行维度）
# #         xabs = x.abs()
# #         xabs_topk_index = xabs.topk(topk, dim=-1).indices
        
# #         # 2. 提取这些 topk 元素的值
# #         topk_values = torch.gather(x, 1, xabs_topk_index)
        
# #         # 3. 对非 Top-K 元素添加噪声
# #         #    这里按照“方差 = 0.1 * x”的理解，直接写成：noise = randn() * (0.1 * x)
# #         noise = torch.randn_like(x) * (0.25 * x.abs())
# #         x = x + noise
        
# #         # 4. 将 topk 元素“还原”，保持其不受噪声影响
# #         #    用 scatter_ 来把提取出来的 topk_values 放回去
# #         x.scatter_(1, xabs_topk_index, topk_values)
    
# #     # 还原回原来的形状
# #     x = x.view(*x_shape)
    
#     # return x


# def fp4_tensor_quantize(x: torch.Tensor,topk,group_size,floort=True):

#     return fp4_fake_quantize(x,topk=topk,group_size=group_size,floor_t=floort)


# # ----------------------
# # 2) 自定义 Linear 前后向
# # ----------------------
# class A8Linear(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor,topk,group_size):
#         # ---- debug
#         quant_x =fp4_tensor_quantize(x,topk,group_size,floort=True)
#         ctx.save_for_backward(quant_x,weight,bias)
#         output = quant_x @ weight.t()
#         if bias is not None:
#             output += bias
#         return output

#     @staticmethod
#     def backward(ctx, grad_output: Tensor):
#         quant_x, weight, bias = ctx.saved_tensors

#         grad_input =  grad_output @ weight

#         grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
#                                       quant_x.reshape(-1, quant_x.shape[-1])
        
#         if bias is not None:
#             out_features = bias.shape[0]
#             grad_bias = grad_output.reshape(-1, out_features).sum(0)
#         else:
#             grad_bias = None

#         return grad_input, grad_weight, grad_bias,None,None


# # ----------------------
# # 3) 带权重量化的 QLinear
# # ----------------------
# class Qfp4Linear(nn.Linear):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         device=None,
#         dtype=None,
#         weight_data=None,
#         bias_data=None,
#         num_bits=4,
#         group_size=256,
#         stochastic_round=True,
#         topk=100
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__(in_features, out_features, bias, **factory_kwargs)
#         self.num_bits = num_bits
#         self.group_size = group_size
#         self.stochastic_round = stochastic_round
#         if weight_data is not None:
#             self.weight.data.copy_(weight_data)
#         if bias_data is not None and bias:
#             self.bias.data.copy_(bias_data)
#         self.topk=topk

#     def forward(self, input: Tensor) -> Tensor:
#         # pdb.set_trace()
#         #qweight = fp4_tensor_quantize(self.weight,topk=self.topk,group_size=self.group_size,floort=False)
#         qweight=self.weight
#         quant_w = qweight.detach() + self.weight - self.weight.detach()
#         # --- debug --- pdb.settrace
#         # pdb.set_trace()
#         # self.group_size=int(input.shape[-1])
#         print(input.shape,self.group_size)
#         return A8Linear.apply(input, quant_w, self.bias,self.topk,self.group_size)

# def prepare_model_for_fp4_training_simulation_act_weight(model, args, target_module):

#     for name, module in reversed(model._modules.items()):
#         if len(list(module.children())) > 0:
#             model._modules[name] = prepare_model_for_fp4_training_simulation_act_weight(module, args, target_module)

#         if isinstance(module, nn.Linear):
#             if not name in target_module:
#                 print('Keep in original linear layer', name, module)
#                 continue
            
#             # NOTE(hanqing): no need to pass those stuffs
#             bias_data = module.bias.data if module.bias is not None else None
#             in_features = module.in_features
#             out_features = module.out_features
#             bias = module.bias is not None
#             weight_data = module.weight.data
#             new_layers = Qfp4Linear(in_features, out_features, bias=bias, device='cuda:0', 
#                 weight_data=weight_data, bias_data=bias_data, 
#                 num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round,topk=args.topk)

#             model._modules[name] = new_layers
#     return model



# # ----------------------
# # 4) 一个简单的测试
# # ----------------------
# if __name__ == "__main__":
#     # 测试一下前向后向是否能跑通
#     model = QLinear(4, 3, bias=True, device='cpu', dtype=torch.float32)
#     x = torch.randn(2, 4, requires_grad=True)

#     y = model(x)
#     loss = y.sum()
#     loss.backward()

#     print("Input x:", x)
#     print("Output y:", y)
#     print("Weight grad:", model.weight.grad)
#     print("Bias grad:", model.bias.grad)
import pdb
import math
import torch
import torch.nn as nn
from torch import Tensor


import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenwiseZScoreNormalizer(nn.Module):
    def __init__(self, input_dim, output_dim, window_size=31, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.window_size = window_size
        self.eps = eps

        # 创建用于局部均值计算的卷积核
        self.mean_kernel = torch.ones(1, 1, window_size) / window_size
        self.mean_kernel = self.mean_kernel.requires_grad_(False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # (B, S, D) -> (B, D, S) 适配 Conv1d x-mean/varaince
        x_perm = x.permute(0, 2, 1)

        mean_kernel = self.mean_kernel.to(x.device).to(x.dtype)
        mean_kernel = mean_kernel.expand(input_dim, 1, self.window_size)

        # 计算局部均值 (mean) 和方差 (variance) 基于 norm
        mean_vals = F.conv1d(x_perm, mean_kernel, padding=self.window_size // 2, groups=input_dim).permute(0, 2, 1)
        var_vals = F.conv1d((x_perm - mean_vals.permute(0, 2, 1)) ** 2, mean_kernel, padding=self.window_size // 2, groups=input_dim).permute(0, 2, 1)

        # 计算标准差 (std)
        std_vals = torch.sqrt(var_vals + self.eps)  # 避免除 0

        # Z-score 归一化
        x_norm = (x - mean_vals) / std_vals

        # 线性变换
        out = self.linear(x_norm)
        return out
# class TokenwiseZScoreNormalizer(nn.Module):
#     def __init__(self, input_dim, output_dim, window_size=31, eps=1e-6):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.window_size = window_size
#         self.eps = eps

#         # 创建用于局部均值计算的卷积核
#         self.mean_kernel = torch.ones(1, 1, window_size) / window_size
#         self.mean_kernel = self.mean_kernel.requires_grad_(False)

#     def forward(self, x):
#         """
#         x: (batch_size, seq_len, input_dim)
#         """
#         batch_size, seq_len, input_dim = x.shape

#         # 计算每个 embedding 的 L2 范数 (norm)
#         norms = torch.norm(x, p=2, dim=-1, keepdim=True)  # (B, S, 1)

#         # (B, S, 1) -> (B, 1, S) 适配 Conv1d
#         norms_perm = norms.permute(0, 2, 1)  # (B, 1, S)

#         # 将卷积核移到与输入张量相同的设备上，并匹配 dtype
#         mean_kernel = self.mean_kernel.to(x.device).to(x.dtype)

#         # 计算局部均值 (mean) 和方差 (variance) 基于 norm
#         mean_vals = F.conv1d(norms_perm, mean_kernel, padding=self.window_size // 2)
#         var_vals = F.conv1d((norms_perm - mean_vals) ** 2, mean_kernel, padding=self.window_size // 2)

#         # 计算标准差 (std)
#         std_vals = torch.sqrt(var_vals + self.eps)  # 避免除 0

#         # Z-score 归一化 on the norm
#         norm_zscore = (norms - mean_vals.permute(0, 2, 1)) / std_vals.permute(0, 2, 1)  # (B, S, 1)

#         # 用归一化后的 norm 调整原始 embedding 的大小 (rescale)
#         x_normalized = x * (norm_zscore / (norms + self.eps))

#         # 线性变换
#         out = self.linear(x_normalized)
#         return out


def fp4_fake_quantize(
    x: torch.Tensor,
    e: int = 1, 
    m: int = 2,
    b: int = 1,
    topk: float = 150,  
    # If topk < 1, it is interpreted as a fraction of group_size.
    group_size: int = 256,
    use_floor: bool = True
) -> torch.Tensor:
    """
    Simulate FP4 quantization on the input tensor.
    
    The tensor is reshaped into groups of size `group_size`. Within each group, 
    the top-k elements (by absolute value) are extracted and later restored to avoid
    quantization on these values.
    
    Parameters:
        x         : Input tensor.
        e, m, b   : Quantization parameters.
        topk      : If topk < 1, it is interpreted as a fraction of group_size; 
                    otherwise, as the number of top elements to treat separately.
        group_size: Number of elements per group.
        use_floor : If True, uses floor rounding; otherwise, uses round.
    
    Returns:
        Quantized tensor with the same shape as x.
    """
    x = x.clone()
    # Determine the actual topk count (if topk is a fraction, convert to an integer count)
    # group_size=768
    if group_size>x.shape[-1]:
        group_size=x.shape[-1]
    if topk < 1:
        topk = int(group_size * topk)
    else:
        topk = int(topk)

    original_shape = x.shape
    
    x = x.view(-1, group_size)
    # x = x.view(x.shape[0]*x.shape[1],)

    if topk > 0:
        x_abs = x.abs()
        topk_indices = x_abs.topk(topk, dim=-1).indices  # Indices of top-k absolute values
        topk_values = torch.gather(x, 1, topk_indices)
        # Temporarily zero out the top-k values
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_indices)
        x[batch_indices, topk_indices] = 0

    # Compute scaling factors using the remaining values
    alpha = x.abs().max(dim=-1).values.clamp(min=1e-6)
    q_max = alpha
    q_min = -q_max
    x_clamped = torch.clamp(x, q_min.unsqueeze(1), q_max.unsqueeze(1))
    alpha_hat = alpha * (2 ** (-b))
    b_hat = 2 ** e - torch.log2(q_max) + torch.log2(torch.tensor(2 - 2 ** (-m), dtype=torch.float32)) - 1

    # Compute quantization levels
    log_v = torch.floor(torch.log2(torch.abs(x_clamped) + 1e-8) + b_hat.unsqueeze(1))
    v = torch.pow(2, torch.clamp(log_v - m, min=1 - m))
    if use_floor:
        x_quant = alpha_hat.unsqueeze(1) * v * torch.floor(x_clamped / (alpha_hat.unsqueeze(1) * v + 1e-12))
    else:
        x_quant = alpha_hat.unsqueeze(1) * v * torch.round(x_clamped / (alpha_hat.unsqueeze(1) * v + 1e-12))
    # print(x_clamped / (alpha_hat.unsqueeze(1) * v + 1e-12))
    # print(torch.floor(x_clamped / (alpha_hat.unsqueeze(1) * v + 1e-12)))
    # print(torch.round(x_clamped / (alpha_hat.unsqueeze(1) * v + 1e-12)))
    # pdb.set_trace()  # Sets a breakpoint
    # Restore the top-k values
    if topk > 0:
        x_quant[batch_indices, topk_indices] = topk_values

    return x_quant.view(*original_shape)


def fp4_tensor_quantize(x: torch.Tensor, topk: float, group_size: int, use_floor: bool = True) -> torch.Tensor:
    return fp4_fake_quantize(x, topk=topk, group_size=group_size, use_floor=use_floor)


class A8Linear(torch.autograd.Function):
    """
    Custom autograd function applying FP4 activation quantization on the input,
    followed by a linear operation.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, topk: float, group_size: int,training) -> Tensor:
        # if training:
        quant_x = fp4_tensor_quantize(x, topk, group_size, use_floor=True)
        # else:
            # quant_x=x
        ctx.save_for_backward(quant_x, weight, bias)
        output = quant_x @ weight.t()
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        quant_x, weight, bias = ctx.saved_tensors

        grad_input = grad_output @ weight
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                      quant_x.reshape(-1, quant_x.shape[-1])
        grad_bias = grad_output.reshape(-1, bias.shape[0]).sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None,None


class Qfp4Linear(nn.Linear):
    """
    A custom linear layer that simulates FP4 quantization on input activations.
    The weights remain in full precision.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_data: torch.Tensor = None,
        bias_data: torch.Tensor = None,
        num_bits: int = 4,  # (Currently not used explicitly in the quantization function)
        group_size: int = 256,
        topk: float = 100
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.topk = topk

        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias:
            self.bias.data.copy_(bias_data)

    def forward(self, input: Tensor) -> Tensor:
        # The weight remains in full precision.
        quant_weight = self.weight       
        return A8Linear.apply(input, quant_weight, self.bias, self.topk, self.group_size,self.training)


def prepare_model_for_fp4_training_simulation_act_weight(model: nn.Module, args, target_module):
    """
    Recursively replace target nn.Linear modules in the model with Qfp4Linear modules.
    
    Parameters:
        model         : The neural network model (nn.Module).
        args          : An object with attributes: weight_bits, weight_group_size, topk.
        target_module : A list of module names to be replaced.
    
    Returns:
        The modified model.
    """
    for name, module in list(model.named_children()):
        if len(list(module.children())) > 0:
            setattr(model, name, prepare_model_for_fp4_training_simulation_act_weight(module, args, target_module))

        if isinstance(module, nn.Linear):
            if name not in target_module:
                print('Keep original linear layer:', name)
                continue

            weight_data = module.weight.data
            bias_data = module.bias.data if module.bias is not None else None
            new_layer=TokenwiseZScoreNormalizer(module.in_features,module.out_features)
            # new_layer = Qfp4Linear(
            #     in_features=module.in_features,
            #     out_features=module.out_features,
            #     bias=(module.bias is not None),
            #     device=module.weight.device,
            #     dtype=module.weight.dtype,
            #     weight_data=weight_data,
            #     bias_data=bias_data,
            #     num_bits=args.weight_bits,
            #     group_size=args.weight_group_size,
            #     topk=args.topk
            # )
            setattr(model, name, new_layer)
    return model


# ----------------------
# A simple test to verify forward and backward passes
# ----------------------
if __name__ == "__main__":
    # Create a small Qfp4Linear layer for testing.
    model = Qfp4Linear(4, 3, bias=True, device='cpu', dtype=torch.float32, topk=0.5, group_size=4)
    x = torch.randn(2, 4, requires_grad=True)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("Input x:", x)
    print("Output y:", y)
    print("Weight grad:", model.weight.grad)
    print("Bias grad:", model.bias.grad)


# ----------------------
# A simple test to verify forward and backward passes
# ----------------------
if __name__ == "__main__":
    # Create a small Qfp4Linear layer for testing.
    model = Qfp4Linear(4, 3, bias=True, device='cpu', dtype=torch.float32, topk=0.5, group_size=4)
    x = torch.randn(2, 4, requires_grad=True)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("Input x:", x)
    print("Output y:", y)
    print("Weight grad:", model.weight.grad)
    print("Bias grad:", model.bias.grad)
