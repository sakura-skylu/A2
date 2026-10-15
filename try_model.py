"""小实验：调用参考 Transformer，观察输入与输出的形状。"""

import torch

from cs336_basics.model import BasicsTransformerLM

def main():
    # TODO: 创建一个小型 BasicsTransformerLM。
    # TODO: 准备形状为 [2, 16] 的整数 token ID 输入。
    # TODO: 调用模型并打印输出形状。
    
    MyLM = BasicsTransformerLM(vocab_size = 100,
                               context_length = 16,
                               d_model = 64,
                               num_layers = 2,
                               num_heads = 4,
                               d_ff = 128,
    )

    x = torch.randint(low = 0, high = 100, size = (2,16))
    print(x.shape)
    print(x)
    x_lm = MyLM.forward(x)
    print(x_lm.shape)
    print(x_lm)

    #获取训练数据
    y = torch.randint(low= 0,high= 100,size= (2,17))
    input_tensor = y[:, :16]
    result_tensor = y[:, 1:17]
    input_tensor_lm = MyLM.forward(input_tensor)
    result_tensor_lm = MyLM.forward(result_tensor)

    print(input_tensor)
    print(result_tensor)
    print(input_tensor_lm.shape)
    print(result_tensor_lm.shape)

    #修改尺寸为 [32,100],
    input_tensor_lm = input_tensor_lm.reshape(32,100)
    result_tensor = result_tensor.reshape(32)
    print(input_tensor_lm.shape)
    print(result_tensor.shape)

    #使用 交叉熵
    loss = torch.nn.functional.cross_entropy(input_tensor_lm,result_tensor)
    print(loss)

    loss.backward()

    optimizer = torch.optim.AdamW(
        MyLM.parameters(),
        lr = 1e-3
    )

    for name,param in MyLM.named_parameters():
        print(name, param.grad.norm().item())

    optimizer.step()

    pass


if __name__ == "__main__":
    main()
