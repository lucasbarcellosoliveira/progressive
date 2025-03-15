#run with uvicorn main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import itertools
from sklearn.cluster import KMeans, DBSCAN, OPTICS, BisectingKMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import scipy
import random
from datetime import timedelta
from io import StringIO

#100.000.000 é limite
quant=100000
varNormal=scipy.stats.multivariate_normal.rvs(mean=[75,25],cov=20,size=quant)
#varSkewed=scipy.stats.skewnorm.rvs(20,loc=5,scale=50,size=quant)
varUniform=scipy.stats.uniform.rvs(loc=0,scale=100,size=quant)
#varPareto=scipy.stats.genpareto.rvs(20,loc=0,scale=100,size=quant)
#varWishart=scipy.stats.wishart.rvs(df=3,scale=1,size=quant)
#synth=pd.concat([pd.DataFrame(varNormal,columns=["N1","N2"]),pd.DataFrame(varSkewed,columns=["S1"]),pd.DataFrame(varUniform,columns=["U1"]),pd.DataFrame(varPareto,columns=["P1"]),pd.DataFrame(varWishart,columns=["W1"])],axis=1)
dates=pd.DataFrame(scipy.stats.uniform.rvs(loc=0,scale=366,size=quant),columns=["date"])
dates["date"]=pd.to_datetime(pd.Series("01/01/2012",index=range(quant)))+dates["date"].astype("timedelta64[D]")
#synth=pd.concat([dates,pd.DataFrame(varNormal,columns=["N1","N2"]),pd.DataFrame(varSkewed,columns=["S1"]),pd.DataFrame(varUniform,columns=["U1"])],axis=1)
synth=pd.concat([dates,pd.DataFrame(varNormal,columns=["N1","N2"]),pd.DataFrame(varUniform,columns=["U1"])],axis=1)
synth["G1"]=5+synth["date"].dt.day*4+synth["date"].dt.month*2#+np.random.randint(-30,30,size=(quant,1))
synth["G1"]=synth["G1"].apply(lambda x: x+random.random()*60-30)
synth["S1"]=35+30*np.sin(synth["date"].dt.day*2*np.pi/31)+synth["date"].dt.month*3
synth["S1"]=synth["S1"].apply(lambda x: x+random.random()*60-30)
#synth["G1"]=5+synth["date"].dt.day*2+synth["date"].dt.month*8+random.random()*30
#print(synth.describe())
#mean, var, skew, kurt=scipy.stats.skewnorm.stats(20,moments='mvsk')
#print(mean, var, skew, kurt)bisec

groups=synth.groupby(synth.date)

synth["dateFloat"]=synth.date.apply(lambda x: x.timestamp())
clustered=KMeans(n_clusters=128,n_init=10,max_iter=5000).fit_predict(synth.drop(columns=["date"]))

clusteredDBSCAN=DBSCAN(eps=500).fit_predict(synth.drop(columns=["date"]))

#clusteredOPTICS=OPTICS().fit_predict(synth.drop(columns=["date"]))

clusteredBisecting=BisectingKMeans(n_clusters=128,n_init=10,max_iter=5000).fit_predict(synth.drop(columns=["date"]))

poly=PolynomialFeatures(degree=5,include_bias=False)
reg=SVR(degree=20,C=2000000)

synth["combined"]=list(zip(synth["N1"],synth["N2"],synth["S1"],synth["U1"]))
combined_weight=synth["combined"].value_counts(normalize=True)
synth["combined_weight"]=synth.combined.apply(lambda x: combined_weight[x])

class Teste(torch.nn.Module):
    def __init__(self):
        super(Teste, self).__init__()
        self.c1=torch.nn.Linear(5,256) #entradas: 2 binarias para variavel, 2 binarias para variavel filtro, 1 para valor filtro
        self.c2=torch.nn.Linear(256,256)
        self.c3=torch.nn.Linear(256,366)
        self.relu=torch.nn.ReLU()
    
    def forward(self,x):
        x=self.relu(self.c1(x))
        x=self.relu(self.c2(x))
        x=self.c3(x)
        return x
    
model=Teste()
model.to("cuda")
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
optimizer.zero_grad()

#vars=["N1","N2","S1","U1","G1"]
vars=["N2","S1","U1","G1"]
#vars=["N2","G1"]

model.train()
for i in range(10000):
    train_var=1 #random.randrange(4)
    train_filter=random.randrange(4)
    train_n=random.random()*50
    out=model(torch.tensor([[train_var//2,train_var%2,train_filter//2,train_filter%2,train_n]],device="cuda"))
    loss=criterion(out.float(),torch.tensor(np.array([synth[["date",vars[train_var]]].groupby("date").agg("mean")[vars[train_var]].values]),device="cuda").float())
    loss.backward()
    optimizer.step()
model.eval()


class Teste2(torch.nn.Module):
    def __init__(self):
        super(Teste2, self).__init__()
        self.c1=torch.nn.Linear(3,256) #entradas: 2 binarias para variavel, 1 para dia
        self.c2=torch.nn.Linear(256,256)
        self.c3=torch.nn.Linear(256,1)
        self.relu=torch.nn.ReLU()
    
    def forward(self,x):
        x=self.relu(self.c1(x))
        x=self.relu(self.c2(x))
        x=self.c3(x)
        return x

model2=Teste2()
model2.to("cuda")
criterion2=torch.nn.MSELoss()
optimizer2=torch.optim.Adam(model.parameters(),lr=0.1)
optimizer2.zero_grad()

#vars=["N1","N2","S1","U1","G1"]
vars2=["N2","S1","U1","G1"]
#vars=["N2","G1"]

model2.train()
for i in range(10000):
    train_day2=random.randrange(366)
    train_var2=3 #random.randrange(4)
    out2=model2(torch.tensor([[train_var2//2,train_var2%2,float(train_day2)]],device="cuda"))
    date2=(pd.to_datetime("01/01/2012")+timedelta(days=train_day2)).strftime('%Y-%m-%d')
    loss2=criterion2(out2.float(),torch.tensor(np.array([[synth.query("date=='"+date2+"'")[vars[train_var2]].agg("mean")]]),device="cuda").float())
    loss2.backward()
    optimizer2.step()
model2.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def delta(full, attribute, date, n):
    return (full[full["date"]==date][attribute]-n)**2

@app.get("/sample/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=synth.sample(frac=precision/100).query(query)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

def threadExec(group,attribute,precision,agg):
    return pd.DataFrame({"date":group[0],attribute:group[1].sample(frac=precision/100)[attribute].agg(agg)},index=[group[0]])

@app.get("/groups/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExec,groups,itertools.repeat(attribute,len(groups)),itertools.repeat(precision,len(groups)),itertools.repeat(agg,len(groups)))
    ret=pd.concat(t).reset_index()[["date",attribute]].to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

def threadExecCluster(group,attribute,precision,agg):
    return group[1].sample(frac=precision/100)

@app.get("/knn/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=synth.groupby(clustered)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/dbscan/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=synth.groupby(clusteredDBSCAN)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/optics/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=synth.groupby(clusteredOPTICS)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/bisecting/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=synth.groupby(clusteredBisecting)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/representative/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=synth[["date",attribute]].sample(frac=precision/100,weights=synth["combined_weight"]).groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/fft/{attribute}/{agg}/{coefficients}")
async def root(attribute:str,agg:str,coefficients:int):
    start=time.process_time()
    ret=synth[["date",attribute]].groupby("date").agg(agg).reset_index()
    #transformado=np.fft.rfft(ret[attribute])
    #np.put(transformado,range(coefficients*2,transformado.size),0)
    #reconstruido=np.fft.irfft(transformado)
    #ret[attribute]=reconstruido.real
    #ret=ret.to_json(orient="records")
    transformado=np.fft.fft(ret[attribute])[:coefficients]
    serial=list(zip(transformado.real,transformado.imag))
    print(time.process_time()-start)
    return [ret["date"].to_json(orient="records"),serial]

@app.get("/samplereg/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=synth[["date",attribute]].sample(frac=precision/100).groupby("date").agg(agg).reset_index()
    reg.fit(poly.fit_transform(ret["date"].values.reshape(-1,1)),ret[attribute])
    ret[attribute]=reg.predict(poly.fit_transform(ret["date"].values.reshape(-1,1)))
    ret=ret.to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(StringIO(ret))
    full=synth.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/nn/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    start=time.process_time()
    q1,q2=query.split("<")
    ret=model(torch.tensor([[vars.index(attribute)//2,vars.index(attribute)%2,vars.index(q1)//2,vars.index(q1)%2,int(q2)]],device="cuda").to(torch.float32))
    print("\nTime")
    print(time.process_time()-start)
    p=synth.query(query)[["date",attribute]].groupby("date").agg(agg).reset_index()
    ref=p[attribute]
    p[attribute]=ret.to("cpu").detach().numpy()[0]
    p=p.to_json(orient="records")
    mse=((ref-ret.to("cpu").detach().numpy()[0])**2).mean()
    print("\nMSE:")
    print(mse)
    return p

@app.get("/nn2/{attribute}/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    start=time.process_time()
    values=[]
    for i in range(366):
        ret=model2(torch.tensor([[vars2.index(attribute)//2,vars2.index(attribute)%2,i]],device="cuda").to(torch.float32))
        values+=[ret.to("cpu").detach().numpy()[0][0]]
    print("\nTime")
    print(time.process_time()-start)
    p=synth.query(query)[["date",attribute]].groupby("date").agg(agg).reset_index()
    ref=p[attribute]
    p[attribute]=pd.Series(np.array(values))
    p=p.to_json(orient="records")
    mse=((ref-pd.Series(np.array(values)))**2).mean()
    print("\nMSE:")
    print(mse)
    return p

@app.get("/test/{experiments}/sample/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        ret=synth.sample(frac=precision/100).query(query)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/representative/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        ret=synth[["date",attribute]].sample(frac=precision/100,weights=synth["combined_weight"]).groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/groups/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExec,groups,itertools.repeat(attribute,len(groups)),itertools.repeat(precision,len(groups)),itertools.repeat(agg,len(groups)))
        ret=pd.concat(t).reset_index()[["date",attribute]].to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/knn/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        clusters=synth.groupby(clustered)
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
        ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/dbscan/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        clusters=synth.groupby(clusteredDBSCAN)
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
        ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/optics/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        clusters=synth.groupby(clusteredOPTICS)
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
        ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/bisecting/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        clusters=synth.groupby(clusteredBisecting)
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
        ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/bisecting/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        clusters=synth.groupby(clusteredBisecting)
        with ThreadPoolExecutor() as executor:
            t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
        ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

@app.get("/test/{experiments}/samplereg/{attribute}/{agg}/{precision}/{query}")
async def root(experiments:int,attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    times,coverage,mse=[],[],[]
    for i in range(experiments):
        start=time.process_time()
        ret=synth[["date",attribute]].sample(frac=precision/100).groupby("date").agg(agg).reset_index()
        reg.fit(poly.fit_transform(ret["date"].values.reshape(-1,1)),ret[attribute])
        ret[attribute]=reg.predict(poly.fit_transform(ret["date"].values.reshape(-1,1)))
        ret=ret.to_json(orient="records")
        times+=[time.process_time()-start]
        computed=pd.read_json(StringIO(ret))
        full=synth.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
        coverage+=[len(computed)/len(full)]
        diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
        mse+=[np.sum(diff)/len(diff)]
    return [np.mean(times),np.std(times),np.mean(coverage),np.std(coverage),np.mean(mse),np.std(mse)]

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app")