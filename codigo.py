import cv2 as cv
import mediapipe as mp
import autopy
import numpy as np
import math
import time
import pynput.keyboard
from pynput.mouse import Controller

# Class Rectangle that stores the coordinates of the top-left and bottom-right corners of the rectangle containing the hand
class Rectangulo:
    def __init__(self, vertice1, vertice2):
        self.x1, self.y1 = vertice1
        self.x2, self.y2 = vertice2


def main():
    # Initialization of libraries
    mp_hands = mp.solutions.hands               
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Inicialización de la cámara
    cap = cv.VideoCapture(0)

    # Dedos utilizados para ciertas acciones
    tip = [4,8,12,16,20]     

    # Coordenadas del cursor    
    pubix, pubiy = 0,0
    cubix, cubiy = 0,0

    # Estados de los clics del mouse
    leftClick, rightClick, middleClick = False, False, False

    # Estados de las funciones de copiar, cortar, pegar, eliminar, retroceder y avanzar
    paste, cut, delete, back, ahead = False, False, False, False, False

    # Dimensiones de la pantalla y la cámara
    widthScreen, heightScreen = autopy.screen.size()
    widthCam, heightCam = int(cap.get(3)), int(cap.get(4))

    # Tamaño del cuadro y márgenes para el movimiento del cursor
    square = 100
    margin = square+((widthCam-2*square) // 3)

    # Umbral de separacion entre dedos
    threshold = 24

    while cap.isOpened():
        ret, frame = cap.read() # Lectur de la camara 
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_RGB) # Obtenemos los resultados de la detección hecha por mediapipe

        # Vector para cada mano donde se almacena si el dedo está levantado (1), dedo recogido (0) o no detectado (-1)
        fingersR = []
        fingersL = []
        matrix_fingerDistance = []
        
        if results.multi_hand_landmarks: 
            
            # Detección de las manos y visualización en la pantalla
            matrix_pixels, num_hands = detect_hands(frame,results,mp_hands,mp_drawing)
                

            # Extracción de información relevante de las manos
            fingersR, fingersL, handRight, handLeft = extract_information(matrix_pixels,num_hands)


            # Interacción con el ratón
            if fingersL != [-1, -1, -1, -1, -1] and fingersL != [0,0,0,0,0]:
                pubix, pubiy = move_mouse(matrix_pixels[handLeft],pubix,pubiy,widthScreen,heightScreen,widthCam,heightCam,square,margin) 

            if fingersR != [-1, -1, -1, -1, -1]:
                for i in range(1,3):                                                                   
                    fingerDistance = distance(tip[0], tip[i], matrix_pixels[handRight])
                    matrix_fingerDistance.append(fingerDistance)

                # Acciones de copiar, cortar, pegar, eliminar, retroceder y avanzar
                perform_control_c(fingersR)
                fingerDistance = distance(tip[1], tip[2], matrix_pixels[handRight])     
                cut = perform_control_x(fingerDistance, fingersR,cut,threshold)
                paste = perform_control_v(fingersR, paste)
                back = perform_control_z(fingersR, fingersL, back)
                ahead = perform_control_y(fingersR, fingersL, ahead)
                fingerDistance = distance(tip[0], tip[4], matrix_pixels[handRight])     
                delete = perform_delete(fingerDistance,delete,threshold)

                # Acciones de clic izquierdo y derecho
                leftClick = perform_click_left(matrix_fingerDistance, leftClick, threshold)
                rightClick = perform_click_right(matrix_fingerDistance, fingersR, rightClick, threshold)

                # Simulación de la rueda del ratón
                simulate_mouse_wheel(fingersR, pubiy,heightScreen,threshold)                               
                
                # Función de zoom
                perform_zoom(fingersR, pubiy,heightScreen)
        
        else: 
            fingersR = [-1,-1,-1,-1,-1]
            fingersL = [-1,-1,-1,-1,-1]  
        
        # Mostrar el video con el seguimiento de las manos
        cv.imshow('Hand Tracking', frame)
        k = cv.waitKey(1)

        if k == 27:
            break
        
    cap.release()
    cv.destroyAllWindows()

# Función para obtener el rectángulo que contiene la mano
def get_hand_rectangle(pixels):
    if not pixels:
        return None  # Si no hay puntos, retorna None

    # Inicializa los valores mínimos y máximos con el primer punto
    x_min, y_min = x_max, y_max = pixels[0][1:]

    # Encuentra las coordenadas extremas que contienen todos los puntos
    for _, cx, cy in pixels:
        x_min = min(x_min, cx)
        y_min = min(y_min, cy)
        x_max = max(x_max, cx)
        y_max = max(y_max, cy)

    # Crea y devuelve un objeto Rectangulo con las coordenadas extremas
    return Rectangulo((x_min, y_min), (x_max, y_max))

# Función para calcular la distancia entre dos puntos en la imagen
def distance(p1, p2, pixels):
        x1, y1 = pixels[p1][1:]
        x2, y2 = pixels[p2][1:]

        length = math.hypot(x2-x1, y2-y1)

        return length

# Función para calcular el punto medio del rectángulo de la mano
def calculate_midpoint(rect):
    return ((rect.x1 + rect.x2) // 2, (rect.y1 + rect.y2) // 2)

# Función para verificar si el punto medio está dentro del rectángulo
def is_midpoint_inside(midpoint, rect):
    return (rect.x1 <= midpoint[0] <= rect.x2) and (rect.y1 <= midpoint[1] <= rect.y2)

def extract_information(matrix_pixels, num_hands):
    fingersR = []
    fingersL = []
    tip = [4,8,12,16,20]      # landmarks o puntos de referencia de la punta de los dedos 
    hand_right = False        # Variables booleanas para conocer si la mano derecha o izquierda ha sido detectada
    hand_left = False
    handRight = 0             # Indice de la mano, es decir, conocer que vector dentro de la matriz corresponde con la mano derecha 
    handLeft = 0

    for i in range(0,num_hands):
        if matrix_pixels[i][tip[0]][1] > matrix_pixels[i][tip[4]][1]:
            # Si el pulgar se encuenta más alejado del origen de coordenadas, sobre el eje de abscisas, se trata de la mano derecha 
            # Recordar que la camara está en modo espejo, lo que hace que la mano derecha se vea como la izquierda en el frame
            handRight = i
            fingers = fingersR
            hand_right = True
            # Al tratarse de la mano derecha, si la punta del pulgar está mas alejado, sobre el eje de abcisas, en comparación con el nudillo del mismo dedo, el dedo esta extendido 
            # En caso contrario, el dedo se considera recogido 
            fingers.append(1 if matrix_pixels[i][tip[0]][1] > matrix_pixels[i][tip[0] - 1][1] else 0)
        else:
            handLeft = i
            fingers = fingersL
            hand_left = True
            fingers.append(1 if matrix_pixels[i][tip[0]][1] < matrix_pixels[i][tip[0] - 1][1] else 0)

        for id in range(1, 5):
            # Para el resto de dedos, si la coordenada en Y está mas proximo al origen que los nudillos del propio dedos, está extendido   
            fingers.append(1 if matrix_pixels[i][tip[id]][2] < matrix_pixels[i][tip[id] - 2][2] else 0)
    
    # Para cuando se detecte solo una mano, indicar que mano no se ha encontrado
    if num_hands == 1 and not hand_right:
        fingersR = [-1, -1, -1, -1, -1]
    elif num_hands == 1 and not hand_left:
        fingersL = [-1, -1, -1, -1, -1]

    return fingersR, fingersL, handRight, handLeft

def detect_hands(frame, results, mp_hands, mp_drawing):
    matrix_pixels = []
    rectangulos_detectados = []
    num_hands = 0 # Número de manos existentes (no almacena detección de manos incorrectas)

    if results.multi_hand_landmarks:
        for id, landmarks in enumerate(results.multi_hand_landmarks):
            # Dibujamos los puntos así como los conectores en las manos 
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 205, 100),
                                                                                    circle_radius=4),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                      thickness=2))

        for hand in results.multi_hand_landmarks:
            pixels = [] # Vector que almacena el id, coordenada en X e Y de cada punto de la mano 
            num_hands += 1

            for id, landmarks in enumerate(hand.landmark):
                # Extraemos y guardamos en el vector: id, coordenada en X e Y en el vector pixels 
                heigh, width, c = frame.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * heigh)
                pixels.append([id, cx, cy])

            # Obtenemos el rectangulo que contiene los puntos, lo guardamos y calculamos su punto medio
            hand_rectangle = get_hand_rectangle(pixels)
            rectangulos_detectados.append(hand_rectangle)
            midpoint_rect = calculate_midpoint(hand_rectangle)

            if (num_hands == 1 or not is_midpoint_inside(midpoint_rect, rectangulos_detectados[0])):
                # Guardamos la inforación cuando se ha detectado solo una mano o el punto medio de la segunda mano no está contenido por el primer rectangulo
                # Solución a posibles fallos de detección de dos manos en una misma 
                matrix_pixels.append(pixels)
                cv.rectangle(frame, (hand_rectangle.x1, hand_rectangle.y1),
                             (hand_rectangle.x2, hand_rectangle.y2), (0, 0, 255), 2)
            else:
                num_hands -= 1

    return matrix_pixels, num_hands

def move_mouse(pixels,pubix,pubiy,widthScreen,heighScreen,widthCam,heighCam,square,margin):
    sua = 5 # Suavizado (favorece movimiento más fluido del ratón)

    x1, y1 = pixels[9][1:]  # Coordenadas (X,Y) del nudillo inferior del dedo medio/corazon de la mano izquierda      
    x3 = np.interp(x1, (margin,widthCam-square), (0,widthScreen)) # Se definen los margenes del frame que corresponde con los de la pantalla
    y3 = np.interp(y1, (square, heighCam-square), (0, heighScreen))

    cubix = pubix + (x3 - pubix) / sua # Se calcula la nueva posición del ratón 
    cubiy = pubiy + (y3 - pubiy) / sua
    autopy.mouse.move(widthScreen - cubix, cubiy) # Movemos el ratón a la posición deseada             
    return cubix, cubiy

def perform_click_left(matrix_finger_distance, click, threshold):
    
    if matrix_finger_distance[0] < threshold and matrix_finger_distance[1] > threshold and not click:
        # Si la distancia entre la punta del pulgar y del índice es menor a un umbral y no se ha hecho click izquierdo en el instante inmediantamente anterior 
        autopy.mouse.toggle(autopy.mouse.Button.LEFT, True) # Click izquierdo pulsado
        click = True # Click izquierdo hecho, llave que nos permite hacer un click o mantener pulsado sin llegar a hacer click izquierdo todo el tiempo que los dedos se toquen 
    elif matrix_finger_distance[0] > threshold and click:
        # Si la distancia entre los dedos supera el umbral y se ha hecho click anteriormente
        autopy.mouse.toggle(autopy.mouse.Button.LEFT, False) # Click izquierdo despulsado 
        click = False

    return click

def perform_click_right(matrix_finger_distance, fingersR, click, threshold):

    if matrix_finger_distance[1] < threshold and matrix_finger_distance[0] > threshold and fingersR[3:] == [1,1] and not click:
        # Si la distancia entre la punta del pulgar y del corazón/medio es menor a un umbral y no se ha hecho click derecho en el instante inmediantamente anterior 
        autopy.mouse.click(autopy.mouse.Button.RIGHT, 0.1) # Click derecho pulsado
        click = True # Click derecho hecho, llave que nos permite hacer un solo click independientemente del tiempo que se toquen ambos dedos seguidamente 
    elif matrix_finger_distance[1] > threshold and click:
        # Si la distancia entre los dedos supera el umbral y se ha hecho click anteriormente
        click = False

    return click

# Calculos de los pasos a realizar en scroll en función de la distancia del cursos al centro de la pantalla
def calculate_steps(cursor_y, screen_height):
    distance_to_center = abs(cursor_y - screen_height / 2)

    max_distance = screen_height / 2
    steps = int(distance_to_center / max_distance * 10)          

    return steps

# Simulación de la rueda del ratón 
def simulate_mouse_wheel(fingersR, cursor_y, screen_height, threshold):
    if fingersR == [0,1,0,0,0]: # Solo Indice levantado 
        mouse = Controller()

        direction = 1 if cursor_y < screen_height / 2 else -1 # Si el cursor está en la mitad superior de la pantalla nos desplazamos hacia arriba y en caso contrario, hacia abajo
        steps = calculate_steps(cursor_y, screen_height)

        mouse.scroll(0, direction * steps)

def perform_control_c(fingersR):
    if fingersR == [1,0,0,0,1]: # pulgar y meñique levantados y el resto recogidos 
        keyboard = pynput.keyboard.Controller()

        with keyboard.pressed(pynput.keyboard.Key.ctrl):
            keyboard.press('c')
            keyboard.release('c')

def perform_control_v(fingersR, paste):
    if fingersR == [0,0,0,0,0] and not paste:  # Mano derecha cerrada, todos los dedos recogidos  
        keyboard = pynput.keyboard.Controller()
    
        with keyboard.pressed(pynput.keyboard.Key.ctrl):
            keyboard.press('v')
            keyboard.release('v')
    
        paste = True # Variable booleanda que permite realizar el comando solo una vez por cada vez que realizamos el gesto

    elif fingersR != [0,0,0,0,0] and paste:
        paste = False # Una vez la mano deje de estar cerada, se podrá hacer otro "control v"

    return paste

def perform_control_x(fingerDistance, fingersR, cut,threshold): 
    if fingerDistance < threshold and not cut and fingersR == [0,1,1,0,0]:
        # Si la distancia entre los dedos indice y corazon es menor al umbral, el resto están recogidos y no se ha cortado inmediantamente antes 
        keyboard = pynput.keyboard.Controller()
    
        with keyboard.pressed(pynput.keyboard.Key.ctrl):
            keyboard.press('x')
            keyboard.release('x')
        cut = True  # Variable booleanda que permite realizar el comando solo una vez por cada vez que realizamos el gesto
    elif fingerDistance > threshold and cut:
        cut = False # Una vez finalice el gesto, se podrá hacer otra vez el comando

    return cut

def perform_delete(fingerDistance, delete,threshold): 
    if fingerDistance < threshold and not delete:
        # Si la distancia entre el pulgar y meñique es menor al umbral y no se ha eliminado inmediantamente antes 
        keyboard = pynput.keyboard.Controller()
    
        keyboard.press(pynput.keyboard.Key.delete)
        keyboard.release(pynput.keyboard.Key.delete)
        delete = True # Variable booleanda que permite realizar el comando solo una vez por cada vez que realizamos el gesto
    elif fingerDistance > threshold and delete:
        delete = False # Una vez finalice el gesto, se podrá hacer otra vez el comando

    return delete

def perform_zoom (fingersR, cursor_y, screen_height):
    if fingersR == [0,1,0,0,1]: # Indice y meñique levantados 
        umbral1 = screen_height / 3
        umbral2 = 2*umbral1
        if (cursor_y < umbral1 or cursor_y > umbral2):
            keyboard = pynput.keyboard.Controller()
            
            if cursor_y < screen_height / 3: direction = '+'
            elif cursor_y > 2*(screen_height/3): direction = '-'
            
            # Si la coordenada en Y del cursos se encuentra en el tercio superior de la pantalla, zoom in 
            # Si la coordenada en Y del cursos se encuentra en el tercio inferior de la pantalla, zoom out
            with keyboard.pressed(pynput.keyboard.Key.ctrl):
                keyboard.press(direction)
                keyboard.release(direction)
            time.sleep(0.075)

def perform_control_z(fingersR, fingersL, back): 
    if fingersR == [1,0,0,0,0] and fingersL == [0,1,0,0,0] and not back:
        # Si el pulgar de la mano derecha levantado e indice de la mano izquierda y no se ha vuelto inmediantamente antes 
        keyboard = pynput.keyboard.Controller()
    
        with keyboard.pressed(pynput.keyboard.Key.ctrl):
            keyboard.press('z')
            keyboard.release('z')
        back = True # Variable booleanda que permite realizar el comando solo una vez por cada vez que realizamos el gesto
    elif (fingersR != [1,0,0,0,0] or fingersL != [0,1,0,0,0]) and back:
        back = False # Una vez finalice el gesto, se podrá hacer otra vez el comando

    return back

def perform_control_y(fingersR, fingersL, ahead): 
    if fingersR == [1,0,0,0,0] and fingersL == [1,0,0,0,1] and not ahead:
        # Si el pulgar de la mano derecha e izquierda levantado ademas del meñique y no se ha adelantado inmediantamente antes 
        keyboard = pynput.keyboard.Controller()
    
        with keyboard.pressed(pynput.keyboard.Key.ctrl):
            keyboard.press('y')
            keyboard.release('y')
        ahead = True # Variable booleanda que permite realizar el comando solo una vez por cada vez que realizamos el gesto
    elif (fingersR != [1,0,0,0,0] or fingersL != [1,0,0,0,1]) and ahead:
        ahead = False # Una vez finalice el gesto, se podrá hacer otra vez el comando

    return ahead


if __name__ == "__main__":
    main()

