import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from keras.models import Sequential
from keras.layers.core import Dense
img = cv.imread('cubos.png')
plt.imshow(img)
plt.show(block=False)
cv.imwrite("localizador2.png",img)
print(len(img[0,1]))
renglones, columnas, profundidad = img.shape
#216*281*3
X = np.zeros((60696,3))
p = 0
for i in range(renglones):
    for j in range(columnas):
        X[p] = img[i,j]
        p = p + 1

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(X)

print(kmeans.labels_)
kmeans.predict([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])
canal = X[0:60696]
canal = canal.reshape(60696,-1)
b = kmeans.predict(canal[0:60696])

b = b.reshape(60696,-1)
c = np.zeros((60696,2))

c[b[:,0]==0,0] = 1
c[b[:,0]==0,1] = 1

c[b[:,0]==1,0] = 1
c[b[:,0]==1,1] = 0

c[b[:,0]==2,0] = 0
c[b[:,0]==2,1] = 0

c[b[:,0]==3,0] = 0
c[b[:,0]==3,1] = 1

print(canal.shape)

datos_entrenamiento = canal[0:42487,0:3]
datos_salida = c[0:42487,0:2]


modelo = Sequential()
modelo.add(Dense(400,input_dim=3,activation='relu'))
modelo.add(Dense(2,activation='sigmoid'))

modelo.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['binary_accuracy'])
modelo.fit(datos_entrenamiento,datos_salida,batch_size=100,epochs=100,verbose=0)

scores = modelo.evaluate(datos_entrenamiento,datos_salida)

print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))
print(modelo.predict(datos_entrenamiento).round())

scores = modelo.evaluate(canal[42487:60696,0:3], c[42487:60696,0:2])
print("\n%s: %.2f%%" % (modelo.metrics_names[1], scores[1]*100))
print(modelo.predict(canal[42487:60696,0:3]).round())

ventanas = np.zeros((24,31,3))
puntero = 0
puntero1 = 0
rojo = 0
verde = 0
azul = 0

for i in range (0,24):
    for j in range (0,31):
        rojo = (img[puntero:puntero+9,puntero1:puntero1+9,0].sum())
        verde = (img[puntero:puntero+9,puntero1:puntero1+9,1].sum())
        azul = (img[puntero:puntero+9,puntero1:puntero1+9,2].sum())
        puntero1  = puntero1 + 9
        ventanas[i,j,0] = round(rojo/81)
        ventanas[i,j,1] = round(verde/81)
        ventanas[i,j,2] = round(azul/81)
    puntero = puntero + 9
    puntero1 = 0

font = cv.FONT_HERSHEY_SIMPLEX
img1 = cv.imread('lienzo.png',1)
img2 = cv.imread('localizador2.png')
img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
for i in range (0,24):
    for j in range (0,31):
        pred = modelo.predict(np.array([ventanas[i,j]])).round()
        if (pred[0,0] == 0 and pred[0,1] == 0):
            cv.rectangle(img1,(j*9,i*9),((j*9)+9,(i*9)+9),(255,0,0),-1)
            cv.rectangle(img2,(j*9,i*9),((j*9)+9,(i*9)+9),(255,255,255),-1)

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
cv.imwrite("fase1.png",img1)
cv.imwrite("fase1p2.png",img2)

grises = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
bordes = cv.Canny(grises,100,200)
ctns, _ = cv.findContours(bordes, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
for i in range(0,len(ctns)):
    area = cv.contourArea(ctns[i])
    if (area>324):
        cv.drawContours(img1, ctns, i, (0,0,0), 1)
        M = cv.moments(ctns[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        xf,yf,wf,hf = cv.boundingRect(ctns[i])
        cv.rectangle(img,(xf,yf),(xf+wf,yf+hf),(255,0,0),3)
        cv.putText(img,"Rojo",(cx+10,cy+10), font, 0.5,(255,255,255),1)

img1 = cv.imread('lienzo.png')
img2 = cv.imread('localizador2.png')
img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
for i in range (0,24):
    for j in range (0,31):
        pred = modelo.predict(np.array([ventanas[i,j]])).round()
        if (pred[0,0] == 0 and pred[0,1] == 1):
            cv.rectangle(img1,(j*9,i*9),((j*9)+9,(i*9)+9),(0,255,0),-1)
            cv.rectangle(img2,(j*9,i*9),((j*9)+9,(i*9)+9),(255,255,255),-1)
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
cv.imwrite("fase2.png",img1)
cv.imwrite("fase2p2.png",img2)
