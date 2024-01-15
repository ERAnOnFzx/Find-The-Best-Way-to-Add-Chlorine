# Find-The-Best-Way-to-Add-Chlorine
1.	使用代码时所有符号都要在英文模式下输入<br>
   Type punctuation in English keyboard.
2.	不要用python默认的idle运行代码，否则遗传算法会非常卡<br>
    DO NOT run these code in IDLE, since the genetic algorithm in IDLE would make the code very slow.
3.	修改inp文件：在第8行的G=epanet('./test2.inp')处修改inp文件地址，将单引号中的内容修改为需要输入的inp文件的路径就可以了<br>
    replace the code in the 8th line """ G=epanet('./test2.inp') """ while changing new input files.
## Run
1.  除了修改地址以外，本程序的使用只需要修改代码最底下的work函数的参数就行<br>
    execute the function "work" can run the code "SearchChlorineQuality.py" easily.
2.	Work函数参数如下，声明时使用了默认参数，调用的时候不写就会用默认值<br>
    The parameters of function "work" are as followings, a call without giving parameters will lead to a call with default parameters.
![image](https://github.com/ERAnOnFzx/Find-The-Best-Way-to-Add-Chlorine/assets/129355970/da43b52b-5dd6-4112-9da6-695e714bb151)
3.  cj为普通加氯点，例：将点3、4设置为加氯点时传入cj=[3, 4]<br>
    cj is the list of the normal vertexs in the graph.
4.	cs为水源加氯点，例：将点3、4设置为加氯点时传入cs=[3, 4]<br>
    cs is the list of the source vertexs in the graph.
5.	testnode为监测点，例：将点3、4设置为监测点时传入testnode=[3,4]<br>
    testnode is the list of vertex that you monitor the quality of Chlorine.
6.	aim为目标浓度，有两种使用方式，当所有监测点目标浓度相同时（假设为0.2），可以直接aim=0.2，否则需要一一对应传入每个监测点目标浓度，例如：监测点1，2，4，5，8目标浓度分别为0.2，0.5，0.25，0.3，0.1则传入testnode=[1, 2, 4, 5, 8],aim=[0.2, 0.5, 0.25, 0.3, 0.1]<br>
    aim is the quality that you asked. There are 2 ways to use the parameter aim. #1 When you have a same aim for every testnode, for example 0.2, you can just set aim=0.2. #2 When there are some testnode which aim for different qualities. Set aim Equals a list which includes the aim of every testnode. For example, when the aims of test node 1, 2, 4, 5, 8 are 0.2, 0.5, 0.25, 0.3, 0.1 set testnode=[1, 2, 4, 5, 8], aim=[0.2, 0.5, 0.25, 0.3, 0.1]
7.	link_aim为管道目标浓度，只能全部设置为相同值（假设为0.2），传入方式为link_aim=0.2<br>
    link_aim is the aim of Chlorine quality in the pipes. Diiferent from the parameter "aim", you can only set all the pipes aim for the same quality. 
8.	inition为遗传算法迭代初始浓度，默认值为0.5，正常情况下随便设一个或者不管即可，传入参数为inition=0.5<br>
    inition is the initial quality that the genetic algorithm starts its search. Most time this parameters would not influence the result, so just leave it or give it any value that you want in boundaries.
9.	bounds为水源点上下限，例如水源点浓度上下限为0.3-0.4时，传入bounds=[0.3,0.4]<br>
    bounds is the boundary of the Chlorine quality of the source vertex. 
10.	boundj为普通加氯点上下限，例如普通加氯点浓度上下限为0.2-4时，传入boundj=[0.2,4]<br>
    boundj is the boundary of the Chlorine quality of the normal vertex.
11.	population_size为遗传算法使用的个体数，越大越容易找到更好的解，但是越大越耗时，使用时如果不写这个参数就取默认值100，传入方式为population_size=100<br>
    population_size is the individual that the genetic algorithm executes. Usually, the larger of this parameter will give you a better results. But it may take a longer time to execute.
12.	max_num_iteration为遗传算法最大迭代次数，默认值为1000但一般用不完，只有迭代1000次还不提前结束时才需要更改此项，传入方式为max_num_iteration=2000<br>
    max_num_iteration is the individuals that the genetic algorithm uses. The larger of this parameter will give you a better results. But it may take a longer time to execute.The default value is 1000, but most time it won't execute 1000 times because of the early stop.
13.	tolerance为精度，传入方式为tolerance=1e-x，其中x为计算时具体到小数点后几位（受遗传算法编码方式影响，输出时仍会带一长串小数点，但计算时只会考虑你要求的精度范围的变化）<br>
    The minimum of improvement that the algorithm accept. When it is 0.001, set tolerace=1e-3.
14.	parents_portion为遗传算法自然选择时直接保留的父代比例，默认值为0.1，如果计算效果不好可以考虑调整这个数值，尽量不要超过0.5，调大调小都可能对计算有帮助（但也可能是上下限问题导致无解）。传入方式为parents_portion=0.1<br>
    parents_portion is the portion of the parents generation that the algorithm keep in every step.
15.	mutation_probability为遗传算法传递基因时的基因突变概率，默认值为0.3，计算效果不好也可以考虑调整这个值，同样不要超过0.5，传入方式为mutation_probability=0.3<br>
    mutation_probability is the probability of mutation between steps.
16.	crossover_probability为遗传算法传递基因时基因交换概率，默认值为0.8，计算效果不好也可以考虑调整这个值，但不要小于0.5，传入方式为crossover_probability=0.8<br>
    crossover_probability is the probability that the gene crossover when creating sons generation.
17.	elit_ratio为遗传算法自然选择时的精英率，0.01相当于前1%不用选择直接留下。尽量不要调整这个参数，传入方式为elit_ratio=0.01<br>
    elit_ratio is the portion of the best individuals that the algorithm keep without executing Roulette selection. Please don't change this parameters unless it's necessary, since the algorithm would be easily to trapped by a locally optimal solution if you set the value of this parameter to large.
18.	max_iteration_without_improv为遗传算法的无提升提前终止代数，默认值为100，即在要求的精度下连续100代无提升就提前终止迭代。<br>
    max_iteration is the
19.	crossover_type为基因交换方式，常用的为one_point和two_point，分别表示单点交换和双点交换。传参方式为crossover_type=‘one_point’。<br>
    crossover_type of the crossover. Switch it by crossover_type='one-point' or crossover_type='two_point'.
20.	xdiff为画图时浓度数据距离点位置左偏移像素数，传入方式为xdiff=1.5<br>
    ydiff为画图时浓度数据距离点位置下偏移像素数，传入方式为ydiff=1.5<br>
    xdiff and ydiff are the pixels of left shift and down shift that the numbers of the quality shift from the points on the image while drawing the results on an image.
21.	fontsize为画图时浓度数据字体大小，传入方式为fontsize=4<br>
    fontsize is the size of the numbers of quality while drawing the results on an image.
23.	color为画图时浓度数据字体颜色，传入方式为color=’red’或color=’blue’之类颜色名<br>
    color is the color of the numbers of quality while drawing the results on an image. For example, color='red' or color='blue'
24.	savefig为画出的浓度图保存地址，假设需要将浓度图保存为和代码同路径的quality.png文件，则传入savefig='./quality.png'<br>
    savefig is the path of the output images while drawing the results on an image.
