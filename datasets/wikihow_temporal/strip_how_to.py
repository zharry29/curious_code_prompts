with open("order_train.csv") as f, open("stripped.csv",'w') as fw:
    for line in f:
        if line.startswith("How to "):
            line = line[7:]
        fw.write(line)