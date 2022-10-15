from collections import OrderedDict
import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class NoShowModel():
    def __init__(self) -> None:
        self._model_file = r'.\noshow_model_objects.pkl'
        self._data_file = r'.\Registered_Patients_DB.csv'
        self.model = None
        self.encoders = None
        self.scaler = None
        self.patient_data = pd.DataFrame()
        self.session_data = OrderedDict()
        self.time_slots = []
        self._feature_ls = [
            'Gender',
            'Age',
            'Hypertension',
            'Diabetes',
            'Alcoholism',
            'Handicap',
            'Employment',
            'Location',
            'Clinic_Location',
            'Scheduled_Day',
            'Appointment_Day',
            'Appointment_Type',
            'Channel',
            'Day_Difference',
            'Appointment_Hour'
        ]
        self.final_result = OrderedDict()
        print("Object Created")

    def load_model(self):
        with open(self._model_file, 'rb') as fp:
            model_objects = pickle.load(fp)
        
        self.model = model_objects.get('model')
        self.encoders = model_objects.get('encoders')
        self.scaler = model_objects.get('scaler')
        
        print('Loading Model Complete')

    def _load_patient_data(self, patient_id):
        patient_ds = pd.read_csv(self._data_file)
        self.patient_data = patient_ds[patient_ds['ID'] == patient_id]
        # print(self.patient_data)
        print('loading Data complete')

    def predict_time_slots(self, session_data, time_slots):
        self.session_data = session_data
        self.time_slots = time_slots
        self._load_patient_data(self.session_data['ID'])
        self._prepare_patient_data()
        self._preprocess_data()
        print(self.session_data.shape, self.patient_data.shape)
        self.model_result = self.model.predict_proba(self.session_data_processed)
        self._transform_result()
        return self.final_result

    
    def _prepare_patient_data(self):
        # print(self.patient_data)
        self.session_data = pd.DataFrame(self.session_data, index= [0])
        # print(self.session_data)
        self.session_data = pd.merge(self.patient_data, self.session_data, on= 'ID')
        
        for feature in self._feature_ls:
            if feature not in self.session_data.columns.tolist():
                self.session_data[feature] = np.nan
        
        self.session_data['Scheduled_Day'] = self.session_data['Scheduled_Date'].dt.day_of_week
        self.session_data['Appointment_Day'] = self.session_data['Appointment_Date'].dt.day_of_week
        self.session_data['Day_Difference'] = self.session_data['Appointment_Date'] - self.session_data['Scheduled_Date']
        self.session_data['Day_Difference'] = self.session_data['Day_Difference'].dt.days
        self.session_data = pd.concat([self.session_data] * (len(self.time_slots)), ignore_index= True)

        self.time_slots = pd.to_datetime(pd.Series(self.time_slots))
        self.time_slots = self.time_slots.dt.hour
        self.session_data['Appointment_Hour'] = pd.Series(self.time_slots)

        self.session_data = self.session_data[self._feature_ls]
        print('Preparing Data Complete')

    def _preprocess_data(self):
        # lenc = LabelEncoder()
        # scaler = StandardScaler()
        cat_cols = self.session_data.select_dtypes(include=['object']).columns.tolist()
        # print(self.session_data)
        for cat in cat_cols:
            self.session_data[cat] = self.encoders[cat].transform(self.session_data[cat])
        # self.session_data[cat_cols] = self.session_data[cat_cols].apply(lenc.fit_transform)
        # print(self.session_data.info())
        self.session_data_processed = self.scaler.transform(self.session_data)
        # print(self.session_data)
        # print(self.scaler.mean_)
        print('Preprocessing Data Complete')

    def _transform_result(self):
        time_slots = self.model_result[:,0]
        self.session_data.loc[:,'time_slots'] = time_slots
        self.session_data.sort_values(['time_slots'], ascending = False, inplace = True)
        self.session_data['Appointment_Hour'] = pd.to_datetime(self.session_data['Appointment_Hour'], format = '%H').dt.strftime('%I:%M %p')
        for id, row in enumerate(self.session_data.head(3)['Appointment_Hour']):
            self.final_result[id] = row
        
        # print(self.session_data)
        print('Transforming Result Complete')

if __name__ == '__main__':
    # testObj = NoShowModel()
    bot_session_data = {
        'ID': 'PPNQ006',
        'Clinic_Location': 'Coimbatore',
        'Scheduled_Date': datetime.datetime.today(),
        'Appointment_Date': datetime.datetime.today(),
        'Appointment_Type': 'New Patient',
        'Channel': 'Chatbot'
    }
    bot_session_data_2 = {
        'ID': 'PCHN001',
        'Clinic_Location': 'Chennai',
        'Scheduled_Date': datetime.datetime.today(),
        'Appointment_Date': datetime.datetime.today(),
        'Appointment_Type': 'Follow-up Visit',
        'Channel': 'Chatbot'
    }
    time_slots = ['10:00 AM', '01:00 PM', '07:00 PM', '08:00 PM', '09:00 PM']

    # print(bot_session_data, '\n', time_slots)
    print("Patient 1 =====>>>")
    patient_1 = NoShowModel()
    patient_1.load_model()
    result = patient_1.predict_time_slots(bot_session_data, time_slots)
    print("Recomendation Result \n\n")
    print(result)
    print("Patient 2 =====>>>")
    patient_2 = NoShowModel()
    patient_2.load_model()
    result_2 = patient_2.predict_time_slots(bot_session_data_2, time_slots)
    print("Recomendation Result \n\n")
    print(result_2)
