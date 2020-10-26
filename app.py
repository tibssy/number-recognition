import numpy as np
import cv2
import num_det as nd


num = ""

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
median = cv2.medianBlur(thresh, 5)
stats = cv2.connectedComponentsWithStats(median, connectivity=8)[2][1:]
sorted_stats = stats[np.argsort(stats[:, 0])]

for i in sorted_stats:
    x, y, w, h, size = i
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    sgmt = 255 - median[y:y+h, x:x+w]
    data = nd.detect(sgmt)
    num += str(data[0])

print("The number is:", num)
cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()
