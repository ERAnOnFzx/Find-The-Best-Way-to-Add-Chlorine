# 读入inp文件
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib
from matplotlib.pyplot import plot
from epyt import epanet

# 读入inp文件
G = epanet('。/test1.inp')  # 将这里的路径更改为inp文件即可
G.plot(nodesID=True)  # 画出管网拓扑结构图
G.plot(linksID=True)

# inp文件中的点的序号和真实序号不同，进行互相转换，name为真实序号，index为在inp中的序号
name2index = {}
index2name = {}
for i in range(G.getNodeCount()):
    Name = G.getNodeNameID(str(i + 1))
    Index = i + 1
    name2index[Name] = Index
    index2name[Index] = Name


# 计算结束绘图
# 可以设置参数：xdiff为浓度标签左偏数，ydiff为浓度标签下偏移数，fontsize为浓度标签字体大小，
# color为浓度标签颜色，savefig为保存图片地址
def new_plot(xdiff=1.5, ydiff=0, fontsize=4, color='red', savefig=None): 
    G.plot(figure=False, nodesID=True, linksID=True)  
    loc = G.getNodeCoordinates()  
    x = loc['x']  
    y = loc['y']  
    keyx = list(x.keys())
    keyy = list(x.keys())

    for i in range(len(keyx)):  # 通过修改{:.2f}中的数字控制输出保留小数点位数，数字改成几就控制小数点后几位
        matplotlib.pyplot.text(x[keyx[i]] - xdiff, y[keyy[i]] - ydiff, "{:.2f}".format(G.getNodeActualQuality(i + 1)),
                               fontsize=fontsize, color=color)

    if savefig is not None:
        matplotlib.pyplot.savefig(savefig)

    matplotlib.pyplot.show()


def work(cj=None, cs=None, testnode=None, aim=None, link_aim=None, inition=None, bounds=None, boundj=None,
         population_size=500,
         max_num_iteration=3000, tolerance=1e-6, parents_portion=0.1, mutation_probability=0.3,  # 具体参数意义可以看代码中的注释
         crossover_probability=0.8, elit_ratio=0.01, max_iteration_without_improv=100, crossover_type='two_point',
         xdiff=1.5, ydiff=0, fontsize=4, color='red', savefig=None):
    # 小案例使用的参数
    if cj is None:
        cj = [30, 45, 138, 188, 436]  # 加氯点中的中途加氯点
    if cs is None:
        cs = [1, 26]  # 加氯点中的水源点
    if testnode is None:
        testnode = []  # 监测点
    if aim is None:
        aim = 0.05  # 目标浓度
    Aim = []   #如果传入的目标浓度 aim 不是列表类型，则假设目标浓度是一个常数值，将其复制到 Aim 列表中的每个元素。
    if type(aim) != list:
        for i in range(len(testnode)):
            Aim.append(aim)
    else:         #如果传入的目标浓度 aim 是列表类型，则直接将其赋值给 Aim
        Aim = aim
    if inition is None:
        inition = 0.5  # 遗传算法第一次迭代使用的浓度
    if boundj is None:
        boundj = [0, 4.0]  # 中途加氯点的浓度上下界
    if bounds is None:
        bounds = [0.3, 4.0]  # 水源点的浓度上下界
    if link_aim is None:
        link_aim = 0.05  # 默认管道浓度

    # 设置变量上下界，传入参数bounds=[下界,上界], boundj=[下界,上界] 闭区间,
    varbound = []  
    for i in range(G.getNodeCount()):   #从 0 到 G.getNodeCount()（节点数量）减 1。
        if i + 1 in cs:                 #索引值加 1 在水源加氯点列表 cs 中，则将水源加氯点的浓度上下界 bounds 添加到 varbound 列表中。
            varbound.append(bounds)
        if i + 1 in cj:
            varbound.append(boundj)
    varbound = np.array(varbound)          #将 varbound 转换为 NumPy 数组，以便后续在遗传算法中使用

    # 设置第一次迭代使用的浓度，传入参数inition=浓度
    # 设置遗传算法使用的个体数，数量越多计算越慢但算出更优解的概率越大，
    # 可以自行设置，传入参数population_size = 个体数
    # population_size = 100
    init_matrix = []
    for i in range(population_size):
        temp = []
        for j in range(len(cs) + len(cj)):
            temp.append(inition)
        init_matrix.append(temp)

    # 设置遗传算法参数
    algorithm_param = {
        'max_num_iteration': max_num_iteration,  # 设置最大迭代次数，
        'convergence_curve': True,   #收敛曲线
        'population_size': population_size,  # 设置个体数，
        'tolerance': tolerance,  # 设置精度，
        'parents_portion': parents_portion,  # 设置每次迭代保留上代个体数量，
        'mutation_probability': mutation_probability,  # 设置遗传过程中的变异率，
        'crossover_probability': crossover_probability,  # 设置遗传过程中的基因交换概率，
        'elit_ratio': elit_ratio,  # 设置每次迭代保留精英个体率，
        'crossover_type': crossover_type,  # 设置基因交换方式，
        'max_iteration_without_improv': max_iteration_without_improv,  #多次没有提升后提前终止迭代，设置终止次数
        'initial_population_matrix': init_matrix  # 设置初始矩阵
    }

    def eval_func(x):  # 评估函数，用于在遗传算法中评价个体优劣
        G.setQualityType('Chlorine', 'mg/L')  # 此行为设置化学物品类型，如果要使用其他类型建议在inp文件中设置并在此行前面加上#注释掉这行

        # 设置初始状态
        k = 0
        for i in range(G.getNodeCount()):
            if i + 1 in cj:  # 对中途加氯点设置其SourceQuality
                G.setNodeSourceQuality(name2index[str(i + 1)], x[k])
                G.setNodeInitialQuality(name2index[str(i + 1)], 0)
		G.setNodeSourceType(name2index[str(i+1)],'SETPOINT')
                k += 1
            elif i + 1 in cs:  # 对水源加氯点设置其InitialQuality
                G.setNodeInitialQuality(name2index[str(i + 1)], x[k])
                G.setNodeSourceQuality(name2index[str(i + 1)], 0)
                k += 1
            else:  # 将其余设置为0
                G.setNodeSourceQuality(name2index[str(i + 1)], 0)
                G.setNodeInitialQuality(name2index[str(i + 1)], 0)

        # 水力与水质分析
        G.solveCompleteHydraulics()
        G.solveCompleteQuality()

        # 评估个体优劣
        Sum = 0
        for i in range(len(testnode)):
            if G.getNodeActualQuality(name2index[str(testnode[i])]) < Aim[i]:
                Sum += 999999
            else:
                Sum += (G.getNodeActualQuality(name2index[str(testnode[i])]) - Aim[i])  # 因为有多个监测点，这里选择的计算方式是让每个点
                # 与aim的浓度差之和尽量小。
        for i in range(G.getLinkCount()):
            if G.getLinkActualQuality(i + 1) < link_aim:
                Sum += 999999
        return Sum

    # 启动遗传算法
    model = ga(function=eval_func, dimension=len(cs) + len(cj), variable_type='real', variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)
    model.run()

    # 输出最佳结果
    x = model.best_variable  # 保存最佳的自变量数组然后重新进行一次模拟
    print("")
    print("加氯点浓度:")
    k = 0
    for i in range(G.getNodeCount()):
        if i + 1 in cj:      #中途加氯点
            print(i + 1, ":", x[k])
            G.setNodeSourceQuality(name2index[str(i + 1)], x[k])
            G.setNodeInitialQuality(name2index[str(i + 1)], 0)
	    G.setNodeSourceType(name2index[str(i+1)],'SETPOINT')
            k += 1
        elif i + 1 in cs:         #水源点
            print(i + 1, ":", x[k])
            G.setNodeInitialQuality(name2index[str(i + 1)], x[k])
            G.setNodeSourceQuality(name2index[str(i + 1)], 0)
            k += 1
        else:
            G.setNodeSourceQuality(name2index[str(i + 1)], 0)
            G.setNodeInitialQuality(name2index[str(i + 1)], 0)
    G.solveCompleteHydraulics()
    G.solveCompleteQuality()

    print("余氯浓度:")
    for i in range(len(testnode)):
        print(str(testnode[i]), ":", G.getNodeActualQuality(name2index[str(testnode[i])]))

    # 绘图
    new_plot(xdiff=xdiff, ydiff=ydiff, fontsize=fontsize, color=color, savefig=savefig)


# 运行
testnode1=[]
for i in range(441):
    testnode1.append(i+1)
work(population_size=500, testnode=testnode1,savefig="D:\quality.png")
