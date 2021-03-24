# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:33:53 2021

@author: Ender
"""


import numpy as np
import cv2

#import dlib_align
import sys




from tensorflow.keras import backend as K 
# Re neuronal 

def cnn(file_name,model):
    
   

    img = cv2.imread(file_name)
    if img is None:
        print("Error al leer imagen")
        sys.exit()
    imagen=img
    #rostros,rects= dlib_align.dlib_alignment(img, 96, 96)
    
#    results=[]
  #  for img in rostros:
    
    img = np.array(img)
    img= cv2.resize(img,(96,96))
    img=img.astype("uint8")# / 255.0 
    img = np.reshape(img,[1,96,96,3])
    #tiempo_inicial=time()
    y_pred=model.predict(img)
    #tiempo_final=time()
    #y_class=np.argmax(y_pred[2],axis=1)
   #results.append(y_pred)
    results=y_pred

# Do some code, e.g. train and save model

    K.clear_session()
        #images.append(img)
        #tiempo_ejecucion = tiempo_final - tiempo_inicial
       # print('Tiempo de ejecución', tiempo_ejecucion)
    
    
    
    #images=np.array(images)
    #y_pred=model.predict(images)
    

        
    # for img in rostros:
    #     plt.imshow(img) 
    #     plt.show()
        
    
    
    # Decodificación de las etiquetas
    for i in  range(len(results)):
       
        # género
        if np.argmax(results[i][0]) == 0:
            results[i][0]=str('M')
        else:
            results[i][0]=str('F')
            
        # raza
        if np.argmax(results[i][1]) == 0:
            results[i][1]=str('Caucasian')
            
        if np.argmax(results[i][1]) == 1:
            results[i][1]=str('African-Afroamerican')
            
        if np.argmax(results[i][1]) == 2:  
            results[i][1]=str('Asian')
        elif np.argmax(results[i][1]) == 3:
            results[i][1]=str('Latino/M.East/Indian')
        # Edad
        if np.argmax(results[i][2]) == 0:  
            results[i][2]=str('0-10')
        if np.argmax(results[i][2]) == 1:  
            results[i][2]=str('11-20')
        if np.argmax(results[i][2]) == 2:  
            results[i][2]=str('21-35')
        if np.argmax(results[i][2]) == 3:  
            results[i][2]=str('36-50')
        if np.argmax(results[i][2]) == 4:  
            results[i][2]=str('51-65')
        elif  np.argmax(results[i][2]) == 5: 
            results[i][2]=str('66+')
        
    
    rects=results    
    # Características finales
    
    print("Rostros encontrados :", len(rects))
    for i in range(len(results)):
        
            print("Resultados del rostro %i :" % i) 
            print("Género: ",results[i][0])
            print("Raza: ", results[i][1])
            print("Edad: ", results[i][2])
           
    
    font_size=0
    w,h=imagen.shape[0],imagen.shape[1]
    if w <500 and len(rects)>=3:
       font_size=0.3
       
    elif w>=500 and len(rects)>=3:
       font_size=0.6
     
    elif w>=500 and len(rects)<=2:
        font_size=0.75
      
    else:
        font_size=0.5
       
    
    
    # Mostrar detección y clasificación en la imagen original
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    for i in range(0,len(rects)):
        
         edad=' age: '+ results[i][2]
         etnia = ' race: ' + results[i][1]
         genero = ' gender: ' +results[i][0]
        
                        # left top   right bottom
         cv2.rectangle(imagen,(rects[i].left(),rects[i].top()),(rects[i].right(),
                                       rects[i].bottom()),(255,0,0),3)
         
         #etiquetas= str(edad+etnia+genero)
         #y = rects[i].top() - 15 if rects[i].top() - 15 > 15 else rects[i].top() + 15
         
         
         y = rects[i].top() - 15 if rects[i].top() - 15 > 15 else rects[i].top() + 15
        
         cv2.putText(imagen,genero+edad,(rects[i].left()-10,y-15), 
                     font, font_size,(0,255,0), 1,cv2.LINE_AA  )
         # cv2.putText(imagen,edad,(rects[i].left()-m1,rects[i].top()-m1), 
         #             font, font_size,(0,255,0), 1,cv2.LINE_AA)
        # y = rects[i].top() - 15 if rects[i].top() - 15 > 15 else rects[i].top() + 15
         cv2.putText(imagen,etnia,(rects[i].left()-10,y+5), 
                     font, font_size,(0,255,0), 1,cv2.LINE_AA )
    
   # plt.imshow(imagen) 
   # plt.show()
    
    
  #  cv2.imshow("Rostros", imagen)
   # cv2.waitKey()
   # cv2.destroyAllWindows()
    cv2.imwrite(file_name,imagen)
    
  
    return 
    
    


