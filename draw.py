import numpy as np
import matplotlib.pyplot as plt

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


if __name__ == "__main__":
    train_loss_path = "trainacc.txt"  # 存储文件路径

    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(0,100*len(y_train_loss),100)  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iter')
plt.ylabel('acc')


plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train acc",color = 'red')
plt.legend()
plt.title('Acc curve')

plt.grid(axis='y')
plt.show()
