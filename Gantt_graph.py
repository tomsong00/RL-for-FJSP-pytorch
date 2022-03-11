import matplotlib.pyplot as plt
import numpy as np

#注：此处的Machine和AGV分别表示Machine类列表和AGV类列表
def Gantt(Task):
    #需要先指定图片大小，比例
    plt.figure(figsize=(20, 6), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
         'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
         'navajowhite','navy', 'sandybrown', 'moccasin']
    Job_text=['J'+str(i+1) for i in range(np.shape(Task)[0])]
    #Job_text=np.array(Job_text)
    #0-id	1-所属大任务  2-内部编号  3-允许执行的机器类型编号	4-开始时间 5-结束时间 6-机器编号
    t = 0
    font = {'family': 'Times New Roman', 'weight': 'normal','size': 16}
    #plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title('Scheduling result', font)
    plt.xlabel("Time", font)
    plt.ylabel("Resource", font)

    plt.tick_params(labelsize=13)
    my_y_ticks = np.arange(1,len(Task), 1)
    plt.yticks(my_y_ticks)
    max_x=max(Task[:,5])
    my_x_ticks = np.arange(0,max_x, 30)
    plt.xticks(my_x_ticks)
    for i in range(np.shape(Task)[0]):
        if Task[i][5] - Task[i][4] != 0:
            plt.barh(Task[i][8], width=np.int(Task[i][5] - Task[i][4]),
                     height=0.8, left=Task[i][4],
                     color=M[np.int(Task[i][1]%22-1)],
                     edgecolor='black')
            plt.text(x=np.float(Task[i][4]+(Task[i][5] - Task[i][4])/2 - 4),
                     y=Task[i][8]-0.1,
                     s="{}-{}".format(np.int(Task[i][1]), np.int(Task[i][7])),
                     fontsize=13)
        if Task[i][5]>t:
            t=Task[i][5]
    #plt.figure(figsize=(10, 8), dpi=300)
    #plt.grid()
    #plt.rcParams['figure.figsize'] = (18,-1)
    #plt.xlim(0,np.int(max(Task[:][5])))
    #plt.plot((1,2,3),(8,5,-1))
    #id = np.where(==Task[:][5])
    #fig = plt.figure(figsize=(20, 8))
    #plth = fig.add_subplot(111)
    #plth.plot()
    #plt.xlim(0,np.int(max(Task[:][5])))
    #plt.ylim(0,np.int(max(Task[:][6])))
    #plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./pic/test.png',bbox_inches='tight')
    plt.show()


