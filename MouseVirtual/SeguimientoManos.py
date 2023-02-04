#-----------Importamos las librerias
import math
import cv2
import mediapipe as mp
import time

#-----------Creamos una clase
class detectormanos():
    #-----------Inicializamos los parametros de la deteccion
    def __init__(self, mode=False, modelComplexity = 1, maxManos = 2, Confdeteccion = 0.5, Confsegui = 0.5):
        self.mode=mode          #Creamos el objeto que tendra su propia variable
        self.maxManos=maxManos  #N° manos
        self.Confdeteccion= Confdeteccion
        self.Confsegui= Confsegui
        self.modelComplex = modelComplexity

        #-----------Creamos los objetos que detectan las manos y dibujaran
        self.mpmanos= mp.solutions.hands
        self.manos= self.mpmanos.Hands(static_image_mode=self.mode,
                                    max_num_hands=self.maxManos,
                                    min_detection_confidence=self.Confdeteccion,
                                    model_complexity = self.modelComplex,
                                    min_tracking_confidence = self.Confsegui)
        self.dibujo= mp.solutions.drawing_utils
        self.tip=[4,8,12,16,20]

    #-----------Funcion para encontrar las manos
    def encontrarmanos(self, frame, dibujar=True):
        imgcolor= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados= self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame,mano,self.mpmanos.HAND_CONNECTIONS)
        return frame

    #-----------Funcion para encontrar la posicion
    def encontrarposicion(self, frame,ManoNum=0,dibujar=False):
        xlista=[]
        ylista=[]
        bbox=[]
        self.lista=[]
        if self.resultados.multi_hand_landmarks:
            miMano= self.resultados.multi_hand_landmarks[ManoNum]
            for id,lm in enumerate(miMano.landmark):
                alto,ancho,c= frame.shape   #Extraemos las dimenciones de la fps
                cx,cy= int(lm.x * ancho),int(lm.y*alto) #convertimos la informacion en pixeles
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id,cx,cy])
                if dibujar:
                    cv2.circle(frame,(cx,cy),5,(0,0,0),cv2.FILLED)  #Dibujamos un circulo

            xmin,xmax= min(xlista),max(xlista)
            ymin,ymax= min(ylista),max(ylista)
            bbox= xmin, ymin, xmax, ymax
            if dibujar:
                cv2.rectangle(frame,(xmin-20, ymin-20),(xmax+20,ymax+20),(0,255,0),2)

        return self.lista, bbox

    #-----------Function para detectar y dibujar los dedos arriba
    def dedosarriba(self):
        dedos=[]
        if self.lista[self.tip[0]][1]>self.lista[self.tip[0]-1][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        
        for id in range(1,5):
            if self.lista[self.tip[id]][2]<self.lista[self.tip[id]-2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos
    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1= self.lista[p1][1:]
        x2, y2= self.lista[p2][1:]
        cx, cy= (x1+x2)//2, (y1+y2)//2

        xd0,yd0= self.lista[4][1:]
        xd1,yd1= self.lista[8][1:]
        xd2,yd2= self.lista[12][1:]
        xd3,yd3= self.lista[16][1:]
        xd4,yd4= self.lista[20][1:]
        if dibujar:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),t)
            cv2.line(frame,(cx,cy),(xd2,yd2),(0,0,255),t)
            cv2.circle(frame, (x1,y1),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (x2,y2),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (cx,cy),r,(0,0,255),cv2.FILLED)

            cv2.circle(frame, (xd0,yd0),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (xd1,yd1),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (xd2,yd2),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (xd3,yd3),r,(0,0,255),cv2.FILLED)
            cv2.circle(frame, (xd4,yd4),r,(0,0,255),cv2.FILLED)
        length= math.hypot(x2-x1,y2-y1)

        return length, frame, [x1,y1,x2,y2,cx,cy]

#-----------Funcion principal
def main():
    ptiempo=0
    ctiempo=0

    #-----------Leemos la camara web
    cap= cv2.VideoCapture(0)
    #-----------Crearemos el objeto
    detector= detectormanos()
    #-----------Realizamos la deteccion de manos
    while True:
        ret, frame= cap.read()
        #Una vez que obtengamos la imagen la enviaremos
        frame= detector.encontrarmanos(frame)
        lista,bbox= detector.encontrarposicion(frame)
        if len(lista)!=0:
            print(lista[4])
        #-----------Mostramos los fps
        ctiempo= time.time()
        fps=1/(ctiempo-ptiempo)
        ptiempo= ctiempo

        cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Manos",frame)
        k=cv2.waitKey(1)

        if k==27:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()