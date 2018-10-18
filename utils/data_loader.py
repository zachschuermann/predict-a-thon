import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(amount_to_cv):
    path_to_dir = 'data/'
    # Read train
    train_data = pd.read_csv(path_to_dir + 'Train.csv', encoding = "ISO-8859-1")
    
    # Shuffle the data
    train_data = train_data.sample(frac=1)
    
    # Convert visitStartTime to hour
    train['datetime'] = pd.to_datetime(train['visitStartTime'], unit='s')
    train['hour'] = train['datetime'].dt.hour
    train.drop(columns = ['datetime', 'visitStartTime', ''], inplace=True)
    
    # Split the data into CV and train
    train_split = train_test_split(train_data, test_size = amount_to_cv)
    
    train = train_split[0]
    cv= train_split[1]
    
    # Read the test data
    test_data = pd.read_csv(path_to_dir + 'Test.csv', encoding = "ISO-8859-1")
    
    # Shuffle the test data
    test_data = test_data.sample(frac=1)
    
    # Convert visitStartTime to hour
    test_data['datetime'] = pd.to_datetime(test_data['visitStartTime'], unit='s')
    test_data['hour'] = test_data['datetime'].dt.hour
    test_data.drop(columns = ['datetime', 'visitStartTime', ''], inplace=True)

    return train, cv, test

def main():
    train, cv, test = read_data()
    return train, cv, test
    
if __name__ == "__main__":
    main()
    