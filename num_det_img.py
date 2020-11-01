import numpy as np
import cv2
import onr
import stfilter


num = ""
accuracy = ""


img = cv2.imread("realtest1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resolution = np.flip(np.asarray(gray.shape))
thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
median = cv2.medianBlur(thresh, 5)
stats = cv2.connectedComponentsWithStats(median, connectivity=8)[2][1:]
sorted_stats = stfilter.filter(stats, resolution)

for i in sorted_stats:
    x, y, w, h, size = i
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    sgmt = 255 - median[y:y+h, x:x+w]
    data = onr.detect(sgmt)
    num += str(data[0])
    acc = str(data[1])
    accuracy += acc + " "

print("The number is:", num, "\naccuracy: ", accuracy, "%")
cv2.imshow("median", median)
cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()
