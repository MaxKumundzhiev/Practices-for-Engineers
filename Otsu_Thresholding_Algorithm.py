# # ------------------------------------------
# #
# # Program created by Maksim Kumundzhiev
# #
# #
# # email: kumundzhievmaxim@gmail.com
# # github: https://github.com/KumundzhievMaxim
# # -------------------------------------------
#
# #lINK - https://habr.com/ru/post/112079/
# #https://habr.com/ru/post/129898/
# #https://habr.com/ru/post/128768/
#https://www.meccanismocomplesso.org/en/opencv-python-the-otsus-binarization-for-thresholding/


# #######################################STEPS####################################################
# # 1. Загружаем в программу изображение RGB
# # 2. Переводим изображение из RGB в полутоновое (с помощью формулы I = 0.2125 R + 0.7154 G + 0.0721 B), сумма коэфицентов RGB в данной формуле не должна превышать 1
# # 3. Вычисляем максимальные и минимальные значений яркости пикселей для нашего полутонового изображения, проходя в цикле все пикселы нашего изображения.
# # 4. Строим гистограмму яркости пикселей нашего изображения????? (что такое эта гистограмма касаемо нашей программы и как ее стоить ???)
# # 5. Устанавливаем порог t = 1,
# # (далее идут непонятки)...a1 и a2 и w1 и w2 что это такое и откуда оно взялось - надо описать для чего и откуда это
# # Тут вычисление в1в2
# # 6. Проходим через всю гистограмму (значений яркости пикселей ), на каждом шаге пересчитывая дисперсию σb(t). (что эта за формула - скрин ее в студию и как она относится к нашей программе)
# # По формуле максимизации межклассовой дисперсии
# # В этой формуле a1 и a2 — средние арифметические значения для каждого из классов, w1 и w2 — вероятности первого и второго классов соответственно.
# # Если на каком-то из шагов дисперсия оказалась больше (как она оказалась больше, что такое дисперсия в этой программе) максимума, то обновляем дисперсию и t=T (откуда т и Т ???).
# # 7. Искомый порог равен T.
# # 8. Применяем значение порога яркости к нашему полутоновому изображению (каким образом???)
# # 9. Выполнение бинаризации полутонового изображения с найденным порогом Т (каким образом??)
# # 10. Визуализация результирующего изображения
#
# #############################
# #Метод Оцу (Otsu's Method) использует гистограмму изображения для расчета порога.
# # Напомню, что гистограмма — это набор бинов, каждый из которых характеризует количество попаданий в него элементов выборки
#
# import cv2
#
#
# def main():
#     windowName = ['Binary', 'Binary Inv', 'Zero', 'Zero Inv', 'Trunc']
#
#     cap = cv2.VideoCapture (0)
#
#     if cap.isOpened ():
#         ret, frame = cap.read ()
#     else:
#         ret = False
#
#     while ret:
#
#         ret, frame = cap.read ()
#
#         th = 127
#         max_val = 255
#
#         ret, o1 = cv2.threshold (frame, th, max_val, cv2.THRESH_BINARY)
#         ret, o2 = cv2.threshold (frame, th, max_val, cv2.THRESH_BINARY_INV)
#         ret, o3 = cv2.threshold (frame, th, max_val, cv2.THRESH_TOZERO)
#         ret, o4 = cv2.threshold (frame, th, max_val, cv2.THRESH_TOZERO_INV)
#         ret, o5 = cv2.threshold (frame, th, max_val, cv2.THRESH_TRUNC)
#
#         cv2.imshow (windowName[0], o1)
#         cv2.imshow (windowName[1], o2)
#         cv2.imshow (windowName[2], o3)
#         cv2.imshow (windowName[3], o4)
#         cv2.imshow (windowName[4], o5)
#         if cv2.waitKey (1) == 27:
#             break
#
#     cv2.destroyAllWindows ()
#     cap.release ()
#
#
# if __name__ == "__main__":
#     main ()


#OTSU IMPLEMANTATION

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/macbook/PycharmProjects/Images/barbara.png',0)

ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize = (8,12))
plt.subplot(3,1,1), plt.imshow(img,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2), plt.hist(img.ravel(), 256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.imshow(imgf,cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.show()
print(ret)




# img = cv.imread('building.png',0)
# blur = cv.GaussianBlur(img,(5,5),0)
# # find normalized_histogram, and its cumulative distribution function
# hist = cv.calcHist([blur],[0],None,[256],[0,256])
# hist_norm = hist.ravel()/hist.max()
# Q = hist_norm.cumsum()
# bins = np.arange(256)
# fn_min = np.inf
# thresh = -1
# for i in range(1,256):
#     p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
#     q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
#     b1,b2 = np.hsplit(bins,[i]) # weights
#     # finding means and variances
#     m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
#     v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
#     # calculates the minimization function
#     fn = v1*q1 + v2*q2
#     if fn < fn_min:
#         fn_min = fn
#         thresh = i
# # find otsu's threshold value with OpenCV function
# ret, otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# print( "{} {}".format(thresh,ret) )