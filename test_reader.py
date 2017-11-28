import create_dataset

for i in create_dataset.train_reader()():
    print i[0]
    print i[1]