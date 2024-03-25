    # 计算损失
                loss = target - output
                
                # 反向传播更新参数
                delta_output = -loss
                # 更新输出层权重
                self.output_weights -= lr * np.outer(delta_output, hidden_outputs)
                # 更新输出层偏置
                self.output_biases -= lr * delta_output