#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import os
#import gym
import random
import warnings
import keyboard
import time


# In[2]:


def hold_W (hold_time):
    import time, pyautogui
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.press('w')

def hold_S (hold_time):
    import time, pyautogui
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.press('s')


# In[3]:


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
time.sleep(10)
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hand_landmarks = None
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    image2 = cv2.flip(image, 1)
    cv2.imshow('MediaPipe Hands', image2)
    if cv2.waitKey(5) & 0xFF == 27:
      break

    if hand_landmarks:
      h, w, _ = image.shape
      t = mp_hands.HandLandmark.INDEX_FINGER_TIP
      x = hand_landmarks.landmark[t].x
      y = hand_landmarks.landmark[t].y
    else:
      y = 0.5

    #env.render()
    r = random.random()
    if y<0.3:
      keyboard.write("w")
    elif y>0.7:
      keyboard.write("s")
    else:
      if r>=y:
        keyboard.write("w")
      else:
        keyboard.write("s")
    #observation, reward, done, info = env.step(action)

    #if abs(observation[0])>3:
      #observation = env.reset()
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#env.close()    
cap.release()

