import pandas as pd
from sklearn.model_selection import train_test_split

def read_data():
    path_to_dir = 'data/'
    # Read train
    train_data = pd.read_csv(path_to_dir + 'Train.csv', encoding = "ISO-8859-1")
    
    # Shuffle the data
    train_data = train_data.sample(frac=1)
    
    # Split the data into CV and train
    train_split = train_test_split(train_data, test_size=0.3)
    train = train_split[0]
    cv= train_split[1]
    
    test_data = pd.read_csv(path_to_dir + 'Test.csv', encoding = "ISO-8859-1")
    
    return train, cv, test_data

def main():
    train, cv, test = read_data()
    return train, cv, test
    
if __name__ == "__main__":
    main()
    