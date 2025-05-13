from scipy.spatial.distance import cdist
import numpy as np
import datetime
import csv
from math import cos,pi
import matplotlib.pyplot as plt
import pygad
import os
csv.field_size_limit(1024*1024*100)
def read_cover(file, minpop=0):
    cid_cover = {}  # 即格子覆盖人数，用minpop筛选
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            if len(eval(row[1])) >= minpop:
                cid_cover[eval(row[0])] = eval(row[1])
    print('cover read')
    return cid_cover
def cid_to_xy(cid, ynum):
    xid = int((cid - 1) / ynum)
    yid = cid - 1 - xid * ynum
    return [xid, yid]
def solve_grav_median(cids,nloc,cid_pop,seed,lam,dirs,ynum):
    #cids是cover的key，即所有有轨迹经过的点，即备选点
    #nloc是选址点数量
    #cid_pop是基于居住的人口，其key是cover的子集
    #lam是重力模型距离幂函数参数
    # solving the gravitational median location choice problem with pygad
    # cids are potential locations
    # nloc is the number of expected locations, which is equal to the number of current stores
    # cid_pop is the residential population in each cell, which is analyzed from the trajectory data
    # seed is for setting the random state
    # lam is the distance power parameter in the gravity model
    # dirs is for saving the results
    # ynum is used to calculate the geographic information by cell ids
    p=nloc
    cids=np.array(cids)
    cid_xy = {}
    xys0=[]
    for i in cids:
        cid_xy[i] = cid_to_xy(i, ynum)
        xys0.append(cid_to_xy(i, ynum))
    xys0=np.array(xys0)

    dmat = cdist(xys0, xys0)
    print('dist ready')
    def Tfunc(sol):

        T=0
        for i in range(len(cids)):
            cid0=cids[i]
            if cid0 in cid_pop:
                wi = cid_pop[cid0]

                sum1, sum2 = 0, 0
                flag=0
                for tag in sol:
                    dij = dmat[i][tag]
                    if dij != 0:
                        sum1 += dij ** (1 - lam)
                        sum2 += dij ** (-lam)

                    else:
                        flag+=1
                if flag==0:
                    T += wi * sum1 / sum2
        return T


    # 遗传算法核心配置
    def fitness_func(ga_instance, solution, solution_idx):
        """适应度函数：最大化最小距离 + 修复约束"""
        selected = np.where(solution == 1)[0]
        if len(selected) != nloc:
            return -np.inf  # 无效解惩罚
        return -Tfunc(selected)

    def repair_solution(solution, p):
        """修复解：确保恰好选中p个点"""
        selected = np.where(solution == 1)[0]
        if len(selected) == p:
            return solution
        # 随机调整至p个点
        solution = np.zeros_like(solution)
        solution[np.random.choice(len(solution), p, replace=False)] = 1
        return solution

    def on_crossover(ga_instance, offspring_crossover):
        """交叉后修复解"""
        for i in range(offspring_crossover.shape[0]):
            offspring_crossover[i] = repair_solution(offspring_crossover[i], p)
        return offspring_crossover

    def on_mutation(ga_instance, offspring_mutation):
        """变异后修复解"""
        for i in range(offspring_mutation.shape[0]):
            offspring_mutation[i] = repair_solution(offspring_mutation[i], p)
        return offspring_mutation

    # 初始化种群（确保合法解）
    def init_population(num_solutions, num_genes, p, fixed_solutions=None, all_fixed_same=False):
        """生成初始种群：每个解恰好有p个1"""
        population = np.zeros((num_solutions, num_genes), dtype=int)
        if fixed_solutions is not None:
            if not all_fixed_same:
                # 1. 注入预设解
                for i, sol in enumerate(fixed_solutions[:num_solutions]):
                    if isinstance(sol, (list, np.ndarray)):
                        if len(sol) != num_genes:   # 如果输入是索引列表
                            population[i, sol] = 1
                        else:  # 如果输入已经是二进制形式
                            population[i] = sol
                # 2. 剩余位置随机生成合法解
                if len(fixed_solutions) >= num_solutions:
                    for i in range(len(fixed_solutions), num_solutions):
                        population[i, np.random.choice(
                            num_genes, p, replace=False)] = 1
            else:
                if len(fixed_solutions) != num_genes:  # 如果输入是索引列表
                    binary_solution = np.zeros(num_genes, dtype=int)
                    binary_solution[fixed_solutions] = 1
                else:  # 如果输入已经是二进制形式
                    binary_solution = fixed_solutions
                # 所有个体设置为相同解
                population[:] = binary_solution

        else:
            for i in range(num_solutions):
                population[i, np.random.choice(
                    num_genes, p, replace=False)] = 1
        return population

    def on_generation(ga_instance):
        population_fitness = ga_instance.last_generation_fitness
        best_fit = np.max(population_fitness)
        mean_fit = np.mean(population_fitness)
        print(f"Gen {ga_instance.generations_completed}: Best = {best_fit:.4f}, Mean = {mean_fit:.4f}")
    # 运行遗传算法
    ga = pygad.GA(
        num_generations=200,
        num_parents_mating=80,
        fitness_func=fitness_func,
        initial_population=init_population(200, len(cids), p),  # 种群大小100
        gene_type=int,
        gene_space=[0, 1],
        mutation_probability=0.20,
        crossover_type="uniform",
        parent_selection_type="rank",  # 稳态选择
        crossover_probability=0.8,
        keep_parents=5,
        keep_elitism=2,
        on_crossover=on_crossover,
        on_mutation=on_mutation,
        parallel_processing=['thread', 32],  # 使用4线程并行
        stop_criteria="saturate_20",
        suppress_warnings=True,
        on_generation=on_generation
    )

    start_time = datetime.datetime.now()
    print('start')
    ga.run()
    print('end')
    end_time = datetime.datetime.now()
    print(f"GA Optimization time: {end_time - start_time}s")

    # 结果输出与可视化
    best_solution, best_fitness, _ = ga.best_solution()
    selected_indices = np.where(best_solution == 1)[0]
    selected_nodes = np.array(cids)[selected_indices].tolist()
    selected_coords = xys0[selected_indices]
    print(best_fitness)
    # print("选中的点坐标：", selected_coords)
    np.save(r'{dirs}/sol.npy'.format(dirs=dirs),selected_nodes)
    np.save(r'{dirs}/sol_xy.npy'.format(dirs=dirs), selected_coords)
    return selected_nodes,selected_coords, best_fitness
if __name__ == '__main__':
    xmin, xmax, ymin, ymax = 113.67561783007596, 114.60880792079337, \
        22.28129833936937, 22.852485545898546  # 深圳最大最小经纬度
    r = 6371 * 1000
    ymid = (ymin + ymax) / 2
    r1 = r * cos(ymid / 180 * pi)
    scale = 150
    xgap = scale / r1 / pi * 180
    ygap = scale / r / pi * 180
    xnum = int((xmax - xmin) / xgap) + 1
    ynum = int((ymax - ymin) / ygap) + 1
    print(ynum)
    nloc = 136
    cover = read_cover(r'E:\基础数据\轨迹\深圳百度20191202\stay_150m_home_t1800_cover.csv', minpop=100)
    print(len(cover))
    cid_xy = {}
    for cid in cover:
        cid_xy[cid] = cid_to_xy(cid, ynum)


    seed = 0
    mingradprop = 0.1
    #lam = 0.5  # huff说不同商品的参数不一样，之后看情况多跑几组
    cid_pop = {}

    with open(r'E:\基础数据\轨迹\深圳百度20191202\user_home.csv', 'r') as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            cid=int(row[1])
            if cid in cid_pop:
                cid_pop[cid]+=1
            else:
                cid_pop[cid]=1
    print('xy and pop ready')
    cids = list(cid_xy.keys())
    for lam in [10]:
        dirs = r'E:\门店选址\20250428grav_median\sz\{lam}'.format(lam=lam)
        if not os.path.exists(dirs):
            os.makedirs(dirs)


        res, xys, perf = solve_grav_median(cids, nloc, cid_pop, seed, lam, dirs, ynum)
        print(len(xys))
        plt.plot(xys[:, 0], xys[:, 1], 'o')
        plt.savefig(r'{dirs}/plot.png'.format(dirs=dirs),dpi=150)
        plt.show()