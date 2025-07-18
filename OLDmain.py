#run with uvicorn main:app --reload
#dataset https://www.kaggle.com/datasets/dgawlik/nyse?select=fundamentals.csv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#import json
#import base64
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
import pywt

#100.000.000 é limite
quant=100000
varNormal=scipy.stats.multivariate_normal.rvs(mean=[75,25],cov=20,size=quant)
varSkewed=scipy.stats.skewnorm.rvs(20,loc=5,scale=50,size=quant)
varUniform=scipy.stats.uniform.rvs(loc=0,scale=100,size=quant)
#varPareto=scipy.stats.genpareto.rvs(20,loc=0,scale=100,size=quant)
#varWishart=scipy.stats.wishart.rvs(df=3,scale=1,size=quant)
#synth=pd.concat([pd.DataFrame(varNormal,columns=["N1","N2"]),pd.DataFrame(varSkewed,columns=["S1"]),pd.DataFrame(varUniform,columns=["U1"]),pd.DataFrame(varPareto,columns=["P1"]),pd.DataFrame(varWishart,columns=["W1"])],axis=1)
dates=pd.DataFrame(scipy.stats.uniform.rvs(loc=0,scale=366,size=quant),columns=["date"])
dates["date"]=pd.to_datetime(pd.Series("01/01/2012",index=range(quant)))+dates["date"].astype("timedelta64[D]")
synth=pd.concat([dates,pd.DataFrame(varNormal,columns=["N1","N2"]),pd.DataFrame(varSkewed,columns=["S1"]),pd.DataFrame(varUniform,columns=["U1"])],axis=1)
synth["combined"]=list(zip(synth["N1"],synth["N2"],synth["S1"],synth["U1"]))
combined_weight=synth["combined"].value_counts(normalize=True)
synth["combined_weight"]=synth.combined.apply(lambda x: combined_weight[x])
synth["G1"]=5+synth["date"].dt.day*4+synth["date"].dt.month*2+random.random()*30
#synth["G1"]=5+synth["date"].dt.day*2+synth["date"].dt.month*8+random.random()*30
print(synth.describe())
#mean, var, skew, kurt=scipy.stats.skewnorm.stats(20,moments='mvsk')
#print(mean, var, skew, kurt)

prices = pd.read_csv("archive/prices.csv")
prices["date"]=pd.to_datetime(prices["date"],format="ISO8601")

groups=prices.groupby(prices.date)

prices["dateFloat"]=prices.date.apply(lambda x: x.timestamp())
clustered=KMeans(n_clusters=16,n_init=10,max_iter=5000).fit_predict(prices.drop(columns=["date","symbol"]))

#clusteredDBSCAN=DBSCAN(eps=0.2).fit_predict(prices.drop(columns=["date","symbol"]))

#clusteredOPTICS=OPTICS().fit_predict(prices.drop(columns=["date","symbol"]))

#clusteredBisecting=BisectingKMeans(n_clusters=16,n_init=10,max_iter=5000).fit_predict(prices.drop(columns=["date","symbol"]))

prices["combined"]=list(zip(prices["open"],prices["close"],prices["low"],prices["high"],prices["volume"]))
combined_weight=prices["combined"].value_counts(normalize=True)
prices["combined_weight"]=prices.combined.apply(lambda x: combined_weight[x])

poly=PolynomialFeatures(degree=5,include_bias=False)
reg=SVR(degree=20,C=2000000)

#HERE: switch to synthetic data!
#prices=synth

#symbols=prices["symbol"].unique()
#symbolsEnc=OneHotEncoder()
#symbolsEnc.fit(symbols.reshape(-1,1))

#for i in range(10):
#    X=np.random.choice(symbols,5)
#    X=symbolsEnc.transform(X)
#    print(X)
#    y=prices[prices["symbol"].isin(X)].groupby("date").mean(numeric_only=True)

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
    train_var=random.randrange(4)
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

@app.get("/prices/sample/{attribute}/all/{agg}/{precision}/{query}")
async def root(attribute:str,agg:str,precision:float,query:str):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=prices.sample(frac=precision/100).query(query)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.query(query).groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

def threadExec(group,attribute,precision,agg):
    return pd.DataFrame({"date":group[0],attribute:group[1].sample(frac=precision/100)[attribute].agg(agg)},index=[group[0]])

@app.get("/prices/groups/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
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
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

def threadExecCluster(group,attribute,precision,agg):
    return group[1].sample(frac=precision/100)

@app.get("/prices/knn/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=prices.groupby(clustered)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/dbscan/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=prices.groupby(clusteredDBSCAN)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/optics/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=prices.groupby(clusteredOPTICS)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/bisecting/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    clusters=prices.groupby(clusteredBisecting)
    with ThreadPoolExecutor() as executor:
        t=executor.map(threadExecCluster,clusters,itertools.repeat(attribute,len(clusters)),itertools.repeat(precision,len(clusters)),itertools.repeat(agg,len(clusters)))
    ret=pd.concat(t)[["date",attribute]].groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/representative/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=prices[["date",attribute]].sample(frac=precision/100,weights=prices["combined_weight"]).groupby("date").agg(agg).reset_index().to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/{attribute}/{symbol}/avg/full")
async def root(attribute:str,symbol:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    start=time.process_time()
    ret=prices[prices[["date",attribute]].symbol==symbol].groupby("date").mean(attribute).reset_index().to_json(orient="records")
    print(time.process_time()-start)
    return ret

@app.get("/prices/{attribute}/{symbol}/avg/{precision}")
async def root(attribute:str,symbol:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    start=time.process_time()
    ret=prices[prices.symbol==symbol][["date",attribute]].sample(frac=precision/100).groupby("date").mean(attribute).reset_index().to_json(orient="records")
    print(time.process_time()-start)
    return ret

@app.get("/fft/prices/{attribute}/{symbol}/{agg}/{coefficients}")
async def root(attribute:str,symbol:str,agg:str,coefficients:int):
    start=time.process_time()
    if symbol=="all":
        ret=prices[["date",attribute]].groupby("date").agg(agg).reset_index()
    else:
        ret=prices[prices.symbol==symbol][["date",attribute]].groupby("date").agg(agg).reset_index()
    #transformado=np.fft.rfft(ret[attribute])
    #np.put(transformado,range(coefficients*2,transformado.size),0)
    #reconstruido=np.fft.irfft(transformado)
    #ret[attribute]=reconstruido.real
    #ret=ret.to_json(orient="records")
    transformado=np.fft.fft(ret[attribute])[:coefficients]
    serial=list(zip(transformado.real,transformado.imag))
    print(time.process_time()-start)
    return [ret["date"].to_json(orient="records"),serial]

@app.get("/prices/samplereg/{attribute}/all/{agg}/{precision}")
async def root(attribute:str,agg:str,precision:float):
    if precision<=0 or precision>100:
        return "Selected precision "+str(precision)+" is invalid"
    print("Precision %")
    print(precision)
    start=time.process_time()
    ret=prices[["date",attribute]].sample(frac=precision/100).groupby("date").agg(agg).reset_index()
    reg.fit(poly.fit_transform(ret["date"].values.reshape(-1,1)),ret[attribute])
    ret[attribute]=reg.predict(poly.fit_transform(ret["date"].values.reshape(-1,1)))
    ret=ret.to_json(orient="records")
    print("\nTime")
    print(time.process_time()-start)
    computed=pd.read_json(ret)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    print("\nCoverage")
    print(len(computed)/len(full))
    diff=[delta(full, attribute, date, n) for date, n in zip(computed["date"],computed[attribute])]
    print("\nMSE:")
    print(np.sum(diff)/len(diff))
    return ret

@app.get("/prices/nn/{attribute}/all/{agg}/{precision}/{query}")
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

@app.get("/prices/nn2/{attribute}/all/{agg}/{precision}/{query}")
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

@app.get("/prices/wavelet/{attribute}/{wavelet}/{agg}/{coefficients}")
async def root(attribute:str,wavelet:str,agg:str,coefficients:int):
    start=time.process_time()
    ret=prices[["date",attribute]].groupby("date").agg(agg).reset_index()
    #(cA,cD)=pywt.dwt(ret[attribute],wavelet)
    #np.put(cA,range(10,cA.size),0)
    #ret[attribute]=pywt.idwt(cA,None,wavelet)
    coeffs=pywt.wavedec(ret[attribute],wavelet)
    for i in range(coefficients,len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    ret[attribute]=pywt.waverec(coeffs,wavelet)
    full=prices.groupby("date").mean(attribute).reset_index()[["date",attribute]]
    diff=[delta(full, attribute, date, n) for date, n in zip(ret["date"],ret[attribute])]
    mse=np.sum(diff)/len(diff)
    print(mse)
    ret=ret.to_json(orient="records")
    print(time.process_time()-start)
    return ret

if __name__=="__main__":
    import uvicorn
    uvicorn.run("OLDmain:app")
