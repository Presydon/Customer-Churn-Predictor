import pickle as pk
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """ Constructor """
    def __init__(self):
        pass


    def get_data(self, data):
        """ This function reads the csv file from the user to the website, using pandas to read the csv file """
        try: 
            df = pd.read_csv(data)
            return df
        except Exception as e:
            return str(e)


    def one_hotencoding(self, dataset, column):
        """ This function uses the one hot encoding to convert the dataset into encoded values, then concatenates the dataset with the new columns. """
        sample = pd.get_dummies(dataset[column]).iloc[:, :-1]
        df = pd.concat([dataset, sample], axis=1).drop(column, axis=1)
        return df


    def bin_dataset(self, dataset, col_name, bin_edges, bin_labels):
        """ This function converts the dataset into a binned dataset, collecting parameters like the dataset, the new column name, bin edges and labels. """ 
        dataset[f'{col_name}_Category'] = pd.cut(dataset.pop(col_name), bins=bin_edges, labels=bin_labels)
        return dataset


    def label_encoder(self, dataset):
        """ this uses a label encoder, converting the object and category values into numerical values"""
        le = LabelEncoder()
        string_col = dataset.select_dtypes(include=['object', "category"])
        for col in string_col:
            dataset[col] = le.fit_transform(dataset[col])
        return dataset


    def scaler(self, dataset):
        """ This functions creates a scaler that normalises the Balance and Estimated Salary to values between 0 and 1 """
        ss = StandardScaler()
        num_ft = ['Balance', 'EstimatedSalary']
        dataset[num_ft] = ss.fit_transform(dataset[num_ft])
        return dataset



class ChurnProcessor:
    def __init__(self):
        """ Constructor """
        self.CLASSIFIER = pk.load(open('models/classifier_model.pkl', 'rb'))
        self.CLUSTER = pk.load(open('models/cluster_model.pkl', 'rb'))
        self.RESULT_MAPPING = {0: 'Moderate', 1: 'High', 2: 'Low'}
        self.CHURN_MAPPING = {1: 'Churn', 0: 'Not Churn'}


    def feature_creator(self, data, alpha, beta):
        """ This function creates the churn and churn probability score """
        data['Churn_Result'] = alpha
        data['Probability'] = beta[:, 1]
        data['Result'] = self.CLUSTER.predict(data['Probability'].values.reshape(-1, 1))
        data['Result'] = data['Result'].map(self.RESULT_MAPPING)
        data['Churn_Result'] = data['Churn_Result'].map(self.CHURN_MAPPING)
        data = data[data['Churn_Result'] == 'Churn']
        return data
        

    def customer_result(self, data):
        """ This function creates the result for the churn, which includes both the customer id, the churn, the churn probability and the total number of churn customers. """
        dataset = zip(data['Customer ID'], data['Result'])
        churn_scoring = []
        for index, (c_id, level) in enumerate(dataset, start=1):
            entry = f'{index}, Customer {c_id} has a {level} probability of Churning'
            churn_scoring.append(entry)
        churn_scoring.append(f"There are {len(churn_scoring)} of churn customers")
        return churn_scoring