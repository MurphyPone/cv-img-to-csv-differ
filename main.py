import cv2
import numpy as np
from imutils import contours
import pytesseract
import pandas as pd
import csv

# Load image, grayscale, Otsu's threshold
def txt_from_spreadsheet_img(_input):
    image = cv2.imread(_input)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove text characters with morph open and contour filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)

    # Repair table lines, sort contours, and extract ROI
    close = 255 - cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 25000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), -1)
            ROI = original[y:y+h, x:x+w]

    result = pytesseract.image_to_string(image)
    # print(result)
    return result

# converts txt to a csv file
def txt_to_csv(txt, file):
    raw = open(f"./working/{file}.txt", "w")
    raw.write(txt)
    raw.close()

    with open(f"./working/{file}.txt") as fin, open(f"./working/{file}.csv", 'w') as fout:
        o = csv.writer(fout)
        for line in fin:
            o.writerow(line.split())

# # a, b are csv files
def compare(a, b):
    with open(a, 'r') as f1, open(b, 'r') as f2, open("./output/diff.txt", "w") as out:
        line_num = 0
        for line1 in f1:
            for line2 in f2:
                for row1 in line1.split("\",\""):
                    for row2 in line2:
                        print(row1, row2)
                        # if(line1[row1] != line2[row2]):
                        #     out.write((f"{line_num}[{row1}]: \t{line1[row1]}\n\t{line2[row2]}\n"))
                line_num += 1
                break
    

# txt1 = txt_from_spreadsheet_img("./inputs/input_a.png")
# txt2 = txt_from_spreadsheet_img("./inputs/input_b.png")

# txt_to_csv(txt1, "raw_input_a")
# txt_to_csv(txt2, "raw_input_b")

# compare("./working/raw_input_a.csv", "./working/raw_input_b.csv")

df_a = pd.read_csv('./working/raw_input_a.csv')
df_b = pd.read_csv('./working/raw_input_b.csv')
	
diff = pd.concat([df_a, df_b]).drop_duplicates(keep=False)
print(diff)

# for row in df_a.iterrows():
#     for col in df_a.itercols():
#     print(row[col])
