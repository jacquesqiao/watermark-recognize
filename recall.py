import sys
#file_name = "train_result"
file_name = sys.argv[1]

TP = 0
TN = 0
FP = 0
FN = 0

with open(file_name, "r") as f:
  for line in f:
    items = line.split(" ")
    if int(items[1]) == 1 and int(items[0]) == 1:
      TP = TP + 1
    if int(items[1]) == 0 and int(items[0]) == 0:
      TN = TN + 1
    if int(items[1]) == 1 and int(items[0]) == 0:
      FP = FP + 1
    if int(items[1]) == 0 and int(items[0]) == 1:
      FN = FN + 1


print "acc:" + str(float(TP)/(TP+FP))
print "recall:" + str(float(TP)/(TP+FN)) 
