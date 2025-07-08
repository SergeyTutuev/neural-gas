import numpy as np
import math as mh
import plotly.express as px
import plotly.graph_objs as go
from sklearn.utils import shuffle

# Параметры алгоритма
dim = 2 # Размерность пространства
epsilon1 = 0.2 # Параметр для смещения ближайшего нерона
epsilon2 = 0.005 # Параметр для смещения соседей ближайшего нейрона
max_age = 25 # Максимальный возраст рёбер
epochs = 35 # количество эпох обучения
count_iter = 80 # Количество итераций, для вставки нового нейрона
err_alpha = 0.5 # Уменьшение максимальной локальной ошибки
err_betta = 0.954 # Уменьшение максимальной локальной ошибки у нового нейрона

# Составляем тест
X1 = np.random.randint(1500,3500, (420, dim))
X22 = np.random.randint(0,5000, (5000, dim))
X2=[]
for x in X22:
    if ((x[0]-2500)**2+(x[1]-2500)**2)<=2000**2 and ((x[0]-2500)**2+(x[1]-2500)**2)>=1500**2:
        X2.append(x)
X=np.array(list(X1)+X2)
X=shuffle(X)

df={"x":[i[0] for i in X],"y": [i[1] for i in X]}
fig=px.scatter(df,x="x",y="y")
fig.show()
X=X.astype("float32")
X/=100000

# Начвльные 2 нейрона в произвольных точках рассматриваемого пространства
neirons={0:{"vector":np.random.randint(0,5000,dim), "vertexes":[[1,0]], "Lockal_error": 0},1:{"vector":np.random.randint(0,5000,dim), "vertexes":[[0,0]],"Lockal_error": 0}}

for i in (0,1):
    neirons[i]["vector"] = neirons[i]["vector"].astype("float32")
    neirons[i]["vector"] /= 100000


for epoch in range(epochs):

    for i in range(len(X)):

        min_dist = 10**20
        number_great_neiron_1 = 0
        number_great_neiron_2 = 0

        # Выбор ближайшего нейрона
        for j in list(neirons.keys()): 
            dist = np.sqrt(np.sum(np.square(X[i]-neirons[j]["vector"])))
            if dist < min_dist:
                min_dist = dist
                number_great_neiron_1 = j

        # Выбор ближайшего к number_great_neiron_1 нейрона 
        min_dist = 10**20
        for j in list(neirons.keys()): 
            dist = np.sqrt(np.sum(np.square(X[i]-neirons[j]["vector"])))
            if dist < min_dist and j != number_great_neiron_1:
                min_dist = dist
                number_great_neiron_2 = j
                
        # Уменьшаем расстояние между данной точкой и двумя рассматриваемыми нейронами, а также их соседями и данной точкой в заданное число раз
        neirons[number_great_neiron_1]["Lockal_error"] += np.sum(np.square(X[i] - neirons[number_great_neiron_1]["vector"]))
        neirons[number_great_neiron_1]["vector"] += epsilon1*(X[i] - neirons[number_great_neiron_1]["vector"])

        for k in neirons[number_great_neiron_1]["vertexes"]:
            neirons[k[0]]["Lockal_error"]+=np.sum(np.square(X[i]-neirons[k[0]]["vector"]))
            neirons[k[0]]["vector"]+=epsilon2*(X[i]-neirons[k[0]]["vector"])
            k[1]+=1

            for l in neirons[k[0]]["vertexes"]:
                if l[0]==number_great_neiron_1:
                    l[1]+=1 # Также обновляем возраст у рёбер, соединяющих 2 ближайших нейрона и их соседей
                    break
                    
        # Также обновляем возраст у рёбер, соединяющих 2 ближайших нейрона и их соседей
        flag1=True
        for  k in neirons[number_great_neiron_1]["vertexes"]:
            if number_great_neiron_2 == k[0]:
                k[1]=0
                flag1=False
                break

        flag2=True
        for  k in neirons[number_great_neiron_2]["vertexes"]:
            if number_great_neiron_1 == k[0]:
                k[1]=0
                flag2=False
                break
                
        # Добавляем ребро между двумя рассматриваемыми нейронами
        if (flag1 and flag2):
            neirons[number_great_neiron_1]["vertexes"].append([number_great_neiron_2,0])
            neirons[number_great_neiron_2]["vertexes"].append([number_great_neiron_1,0])
        
        # Удаление рёбер с очень большим возрастом
        
        for j in neirons:
            new_ver=[]
            for k in neirons[j]["vertexes"]:
                if k[1] <= max_age:
                   new_ver.append(k)
            neirons[j]["vertexes"]=new_ver
        
        new_neirons={}
        keys=list(neirons.keys())
        for j in keys:
            if len(neirons[j]["vertexes"]):
                new_neirons[j]=neirons[j]

        neirons = new_neirons

        # Добавляем новый нейрон между тем нейроном, у которого самая большая локальная ошибка (сумма квадратов всех перемещений данного нейрона) и соседа данного нейрона с максимальной локальной ошибкой
        if (epoch*len(X)+i+1) % count_iter == 0:
            max_err=-1
            neiron_err_1=0
        
            for j in (list(neirons.keys())):
                if neirons[j]["Lockal_error"] > max_err:
                    max_err=neirons[j]["Lockal_error"]
                    neiron_err_1=j

            max_err=-1
            neiron_err_2=0
            for j in neirons[neiron_err_1]["vertexes"]:
                if neirons[j[0]]["Lockal_error"]>max_err:
                    max_err=neirons[j[0]]["Lockal_error"]
                    neiron_err_2=j[0]

            # Установка параметров нового нейрона
            new_neiron = max(list(neirons.keys())) + 1
            neirons[new_neiron] = {}
            neirons[new_neiron]["vector"] = (neirons[neiron_err_1]["vector"] + neirons[neiron_err_2]["vector"])/2
            neirons[new_neiron]["Lockal_error"] = neirons[neiron_err_1]["Lockal_error"]*err_betta
            neirons[new_neiron]["vertexes"] = [[neiron_err_1,0],[neiron_err_2,0]]

            neirons[neiron_err_1]["Lockal_error"]*=err_alpha
            neirons[neiron_err_2]["Lockal_error"]*=err_alpha

            for j in neirons[neiron_err_1]["vertexes"]:
                if j[0]==neiron_err_2:
                    neirons[new_neiron]["vertexes"][0][1],neirons[new_neiron]["vertexes"][1][1]=j[1],j[1]
                    j[0]=new_neiron
                    break

            for j in neirons[neiron_err_2]["vertexes"]:
                if j[0]==neiron_err_1:
                    j[0]=new_neiron
                    break
    print(epoch+1)
            #print(len(neirons))

"""max_key=max(list(neirons.keys()))
matrix = [[0]*(max_key+1) for i in range(max_key+1)]
for i in neirons:
    for k in neirons[i]["vertexes"]:
        matrix[i][k[0]]=k[1]

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if matrix[i][j]!=matrix[j][i] and i!=j:
        print("*")"""

# Раздление полученного графа на компоненты связности поиском в ширину
sp_sm={}
labs = {}
for i in list(neirons.keys()):
    sp_sm[i]=[k[0] for k in neirons[i]["vertexes"]]
    labs[i]=1

components={}
count_comp=0
while 1 in labs.values():
    for i in list(labs.keys()):
        if labs[i]:
            memory=[i]
            break

    component=[]
    while len(memory)>0:
        if labs[memory[0]]:
            component.append(memory[0])
            for i in sp_sm[memory[0]]:
                if labs[i]:
                    memory.append(i)

        labs[memory[0]]=0
        del memory[0] 

    components[count_comp]=list(component)
    count_comp+=1

# Формируем кластеры
classes=[[] for i in range(len(components))]
for x in X:
    min_dist=10**20
    class_number=0
    for i in components:
        for j in components[i]:
            dist = np.sqrt(np.sum(np.square(x-neirons[j]["vector"])))
            if dist<min_dist:
                min_dist=dist
                class_number=i
    classes[class_number].append(x*100000)

fig2=go.Figure()
for i in classes:
    df={"x":[k[0] for k in i],"y":[k[1] for k in i]}
    fig2.add_trace(go.Scatter(x=df["x"],y=df["y"],mode="markers"))
fig2.show()

