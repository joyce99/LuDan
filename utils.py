import numpy as np
import scipy.sparse as sp
import math
import random

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append([head,r+1,tail])
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def get_matrix(triples,entity,rel,symmetry):

        ent_size = max(entity)+1
        rel_size = (max(rel) + 1)
        print(ent_size, rel_size)
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        rel_mark = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))

        # adj_feature 主对角线全为1
        for i in range(max(entity)+1):
            adj_features[i, i] = 1

        # for h,r,t in triples:
        #     adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
        #     adj_features[h,t] = 1; adj_features[t,h] = 1;
        #     radj.append([h,t,r]); radj.append([t,h,r+rel_size]);
        #     rel_out[h][r] += 1; rel_in[t][r] += 1

        for i in range(len(triples)):
            h = triples[i][0]
            r = triples[i][1]
            t = triples[i][2]
            if symmetry[h][t] == 1:
                rel_mark[h, t] = 1
                adj_matrix[h, t] = 2
                adj_matrix[t, h] = 2
                adj_features[h, t] = 1
                adj_features[t, h] = 1
                adj_features[h, t] *= math.exp(1/2)
                adj_features[t, h] *= math.exp(1/2)
                radj.append([h, t, r])
                radj.append([t, h, r + rel_size])
                rel_out[h][r] += 1
                rel_in[t][r] += 1
            else:
                adj_matrix[h, t] = 1
                adj_matrix[t, h] = 1
                adj_features[h, t] = 1
                adj_features[t, h] = 1
                adj_features[h, t] *= math.exp(- 1 / 2)
                adj_features[t, h] *= math.exp(- 1 / 2)
                radj.append([h, t, r])
                radj.append([t, h, r + rel_size])
                rel_out[h][r] += 1
                rel_in[t][r] += 1
            # 上面需要的！！

        count = -1
        s = set()
        d = {}
        e_index, e_val = [], []
        r_index, r_val = [], []  # r_index表示第几组关系和他对应的关系名
        conRelNum = {}
        ratioNum = []
        ratio = []
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                if symmetry[h][t] == 1:
                    r_val.append(2)
                else:
                    r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                conRelNum[count] = 0
                r_index.append([count,r])
                if symmetry[h][t] == 1:
                    r_val.append(2)
                else:
                # 上面需要的！！
                    r_val.append(1)
            if symmetry[h][t] == 1:
                conRelNum[count] += 1  # 得到{0:2,1:4}这样同构的
            if conRelNum[count] < 1:
                conRelNum[count] = 0
            # 上面需要的！！

        relNum = 0
        rel = 0
        r_sum = []
        for i in range(len(r_index)):
            relNum += 1
            r_val[i] = 1 / d[r_index[i][0]]
            if r_val[i] == 2:
                r_val[i] *= math.exp(1/2)
            else:
                r_val[i] *= math.exp(-1/2)
            # 上面需要的！！！


            # ratioNum.append(0)
            # if conRelNum[r_index[i][0]] != 0 and conRelNum[r_index[i][0]] and conRelNum[r_index[i][0]] != d[r_index[i][0]]:
            #     ratioNum[i] = conRelNum[r_index[i][0]] / d[r_index[i][0]]
            #     # 第二步的实现，同构关系的权重值
            #     ratio[i] = math.log((1 - ratioNum[i]) / ratioNum[i]) / 2
            #     r_val[i] *= math.exp(ratio[i])
            # 上面不需要

            rel += r_val[i]
            # 每个关系中同构关系占比的最终值
            # 先归一化
            if relNum == d[r_index[i][0]]:
                relNum = 0
                for j in range(d[r_index[i][0]]):
                    r_sum.append(rel)
                rel = 0
            # 上面需要的！！！

        for i in range(len(r_index)):
            r_val[i] = r_val[i]/r_sum[i]

        # 第一步：构建em,求出同构的关系的总数
        # 第二步：求出同构关系占所有关系（这个数量=d[r_index[i][0]]）的比值
        # 第三步： 求出新的增加后的同构关系权重
        # 第四步： 进行归一化
        # 如果ratio在（0,1），则 r_val[i] *= math.exp(ratio[i])

        rel_features = np.concatenate([rel_in,rel_out],axis=1)

        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))
        return adj_matrix,r_index,r_val,adj_features,rel_features
def  load_data(lang,train_ratio = 0.3):
    entity1,rel1,triples1 = load_triples(lang + 'triples_1')
    entity2,rel2,triples2 = load_triples(lang + 'triples_2')

    # 寻找连接
    # def findNeighbor(index,n):
    #     if n==1:
    #         return 0
    #     for i in triples2:
    #         if i[0] == index and i[2] == 37747:
    #             print(i,index)
    #             return i
    #         else:
    #             a= findNeighbor(i[2],n+1)
    #             # print(index)
    #         if i[2] == index and i[0] == 37747:
    #             print(i,index)
    #             return i
    #         else:
    #             a= findNeighbor(i[0],n+1)
    #             # print(index)
    #     return 0
    #
    # for index in [36051]:
    #     aaa = findNeighbor(index,0)


    # if "_en" in lang:
    #     alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
    #     np.random.shuffle(alignment_pair)
    #     train_pair,test_pair = alignment_pair[0:int(len(alignment_pair)*train_ratio)],alignment_pair[int(len(alignment_pair)*train_ratio):]
    # else:
    train_pair = load_alignment_pair(lang + 'sup_ent_ids')
    del train_pair[1500: 4500]

    test_pair = load_alignment_pair(lang + 'ref_ent_ids')
    # train_pair = train_pair[0:450]
    symmetry_len = max(entity1.union(entity2))+1
    symmetry = np.zeros((symmetry_len,symmetry_len))

    # is_pair = []
    # 找两个对齐实体且他们之间有关系联通
    # for a1 in range(len(train_pair)):
    #     for a2 in range(a1+1,len(train_pair)):
    #         for i in range(len(triples1)):
    #             if train_pair[a1][0] == triples1[i][0] and train_pair[a2][0] == triples1[i][2]:
    #                 for j in range(len(triples2)):
    #                     if train_pair[a1][1] == triples2[j][0] and train_pair[a2][1] == triples2[j][2]:
    #                         is_pair.append((int(train_pair[a1][0]), int(train_pair[a1][1])))
    #                         is_pair.append((int(train_pair[a2][0]), int(train_pair[a2][1])))
    # for i in is_pair:
    #     print(is_pair[i])
    rel_pair = []


    # 过滤部分同构
    # flag = 0
    # str1 = triples1[0][0]
    # triples3 = []
    # for e1, e2 in train_pair:
    #     for t1, r, t2 in triples1:
    #         if e1 == t1:
    #             for e3, e4 in train_pair:
    #                 if e3 == t2:
    #                     for t3, r1, t4 in triples2:
    #                         if (e2 == t3 and e4 == t4) or (e2 == t4 and e4 == t3):
    #                             print("[", t3, ",", r1, ",", t4, "]")
    #                             triples3.append([t3, r1, t4])


    # 总边
    triples4 = triples2
    # delNum = int(len(triples3) * 0.05)
    # for i in range(delNum):
    #     # delLoc = random.randint(0, len(triples3) - 1)
    #     delLoc = random.choice(triples3)
    #     print(delLoc)
    #     print(triples2[0])
    #     triples2.remove(delLoc)
    #     triples3.remove(delLoc)
        #
        # print(triples3[delLoc])
        # del triples3[delLoc]
    #
    # for i in range(len(triples3)):
    #     if triples3[i] in triples4:
    #         triples4.remove(triples3[i])

    # with open("forty.txt", "w") as f:
    #     for h,r,t in triples4:
    #         f.write(str(h) + "\t" + str(r-1) + "\t" + str(t) + "\n")

    # 注释开始！！！！！！
    for a1 in range(len(train_pair)):
        for i in range(len(triples1)):
            if triples1[i][0] == train_pair[a1][0]:
                for a2 in range(len(train_pair)):
                    if train_pair[a2][0] == triples1[i][2]:
                        for j in range(len(triples2)):
                            if triples2[j][0] == train_pair[a1][1] and triples2[j][2] == train_pair[a2][1]:
                                rel_pair.append([triples1[i][1], triples2[j][1]])

    # print("关系对齐个数：", flag)





    num = 0
    more = 0
    nnum = 0  # 原有同构的个数
    allnum = 0  # 总的个数
    ocNum = 0
    ocNumOld = 0
    addA=0
    oldA = 0
    addB = 0
    oldB = 0
    while more < 5:
        more += 1
        for a1 in range(len(train_pair)):
            ngbNum = 0
            sameNum = 0
            oldN = 0
            oldS = 0
            for i in range(len(triples1)):
                if train_pair[a1][0] == triples1[i][0]:
                    ngbNum += 1
                    oldN += 1
                    for a2 in range(len(train_pair)):
                        if train_pair[a2][0] == triples1[i][2]:
                            flag = 0
                            for j in range(len(triples2)):
                                leftTriples2 = triples2[j][0]
                                rightTriples2 = triples2[j][2]
                                leftTrain = train_pair[a1][1]
                                rightTrain = train_pair[a2][1]
                                if (leftTriples2 == leftTrain) and (rightTriples2 == rightTrain):
                                    flag = 1
                                    if symmetry[triples1[i][0], triples1[i][2]] != 1:
                                        nnum += 1
                                        allnum += 1
                                    symmetry[triples1[i][0], triples1[i][2]] = 1
                                    symmetry[triples1[i][2], triples1[i][0]] = 1
                                    symmetry[triples2[j][0], triples2[j][2]] = 1
                                    symmetry[triples2[j][2], triples2[j][0]] = 1
                                    sameNum+=1
                                    oldS +=1

                                if (rightTriples2 == leftTrain) and (leftTriples2 == rightTrain):
                                    flag = 1
                                    if symmetry[triples1[i][0], triples1[i][2]] != 1:
                                        nnum += 1
                                        allnum += 1
                                    symmetry[triples1[i][0], triples1[i][2]] = 1
                                    symmetry[triples1[i][2], triples1[i][0]] = 1
                                    symmetry[triples2[j][0], triples2[j][2]] = 1
                                    symmetry[triples2[j][2], triples2[j][0]] = 1
                                    sameNum += 1
                                    ngbNum += 1
                                    oldS += 1

                            if flag == 0:
                                for increase in range(len(rel_pair)):
                                    if triples1[i][1] == rel_pair[increase][0]:
                                        triples2.append([int(triples2[j][0]), int(rel_pair[increase][1]), int(triples2[j][2])])
                                        num += 1
                                        addB += 1
                                        if symmetry[triples2[j][0], triples2[j][2]] != 1:
                                            allnum += 1

                                        symmetry[triples1[i][0], triples1[i][2]] = 1
                                        symmetry[triples1[i][2], triples1[i][0]] = 1
                                        symmetry[triples2[j][2], triples2[j][0]] = 1
                                        symmetry[triples2[j][0], triples2[j][2]] = 1
                                        print("increase ", num, ": ", int(triples2[j][1]))

                                        sameNum += 1
                                        ngbNum += 1
                                        break

            for i in range(len(triples2)):
                if train_pair[a1][1] == triples2[i][0]:
                    ngbNum += 1
                    oldN += 1
                    for a2 in range(len(train_pair)):
                        # a1 + 1,
                        if train_pair[a2][1] == triples2[i][2]:
                            flag = 0
                            for j in range(len(triples1)):
                                if triples1[j][0] == train_pair[a1][0] and triples1[j][2] == train_pair[a2][1]:
                                    flag = 1
                                    if symmetry[triples1[j][0], triples1[j][2]] != 1:
                                        nnum += 1
                                        allnum += 1
                                    symmetry[triples1[j][0], triples1[j][2]] = 1
                                    symmetry[triples1[j][2], triples1[j][0]] = 1
                                    symmetry[triples2[i][0], triples2[i][2]] = 1
                                    symmetry[triples2[i][2], triples2[i][0]] = 1
                                    sameNum += 1
                                    oldS+=1

                                if triples1[j][2] == train_pair[a1][0] and triples1[j][0] == train_pair[a2][1]:

                                    flag = 1
                                    if symmetry[triples1[j][0], triples1[j][2]] != 1:
                                        nnum += 1
                                        allnum += 1
                                    symmetry[triples1[j][0], triples1[j][2]] = 1
                                    symmetry[triples1[j][2], triples1[j][0]] = 1
                                    symmetry[triples2[i][0], triples2[i][2]] = 1
                                    symmetry[triples2[i][2], triples2[i][0]] = 1
                                    sameNum += 1
                                    oldS+=1

                            if flag == 0:
                                for increase in range(len(rel_pair)):
                                    if triples2[i][1] == rel_pair[increase][1]:
                                        triples1.append([int(triples1[j][0]), int(rel_pair[increase][0]), int(triples1[j][2])])
                                        num += 1
                                        addA += 1
                                        if symmetry[triples1[j][0], triples1[j][2]] != 1:
                                            allnum += 1

                                        symmetry[triples1[j][0], triples1[j][2]] = 1
                                        symmetry[triples1[j][2], triples1[j][0]] = 1
                                        symmetry[triples2[i][0], triples2[i][2]] = 1
                                        symmetry[triples2[i][2], triples2[i][0]] = 1
                                        print("increase ", num, ": ", int(triples1[j][1]))
                                        sameNum += 1
                                        ngbNum += 1
                                        break
            ocNum += sameNum / ngbNum
            ocNumOld += oldS / oldN
    ocNum = ocNum / (len(train_pair)*5)
    ocNumOld = ocNumOld / (len(train_pair) * 5)
    print("ocNum:",ocNum,"++++ocNumOld:",ocNumOld)
    print("addA:",addA,"++++addB:",addB)
    # for en in range(len(train_pair)):
    #     ngbNum = 0
    #     sameNum = 0
    #     for i in range(len(triples1)):
    #         if triples1[i][0]==train_pair[en][0] or  triples1[i][2]==train_pair[en][0]:

    # 注释结束！！！！！！
    # 计算重叠系数
    ocSum = 0
    # for e1 in range(len(train_pair)):
    #     ngbSet = set()
    #     ocNum = 0
    #     for i in range(len(triples1)):
    #         if train_pair[e1][0] == triples1[i][0]:
    #             if triples1[i][2] not in ngbSet:
    #                 ngbSet.add(triples1[i][2])
    #                 for e2 in range(len(train_pair)):
    #                     if triples1[i][2] == train_pair[e2][0]:
    #                         for j in range(len(triples2)):
    #                             if (triples2[j][0] == train_pair[e1][1] and triples2[j][2] ==
    #                                 train_pair[e2][1]) or (triples2[j][2] == train_pair[e1][1] and triples2[j][0] ==
    #                                                        train_pair[e2][1]):
    #                                 ocNum += 1
    #         elif train_pair[e1][0] == triples1[i][2]:
    #             if triples1[i][0] not in ngbSet:
    #                 ngbSet.add(triples1[i][0])
    #                 for e2 in range(len(train_pair)):
    #                     if triples1[i][0] == train_pair[e2][0]:
    #                         for j in range(len(triples2)):
    #                             if (triples2[j][0] == train_pair[e1][1] and triples2[j][2] == train_pair[e2][1]) or (
    #                                     triples2[j][2] == train_pair[e1][1] and triples2[j][0] == train_pair[e2][1]):
    #                                 ocNum += 1
    #     ocSum += ocNum / len(ngbSet)

    # ocSum /= len(train_pair)
    # print("ocSum:" , ocSum)
    print("nnum: ", nnum, " and allnum: ", allnum)
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2), symmetry)

    return np.array(train_pair),np.array(test_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features

def get_hits(vec, test_pair, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])  # 左边的测试实体组成的矩阵
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])  # 右边的测试实体组成的矩阵

    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)  # 单位化向量
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)  # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y(-1就是默认吧里面所有的一阶向量全部从小到大排序)
    # 有左向量和右向量相乘最小的那个值就是他们最近的位置，这里就把每一个左边每一个实体，到右边每一个实体的距离排出来了，然后和他一样的值得那个就是他的对应值
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]  # 就是取sim第i行的向量
        rank_index = np.where(rank == i)[0][0]  # 寻找正确的那个在哪个位置比如i=4,【101，78，213,4】去找4在哪个位置
        MRR_lr += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:,i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1/(rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))
