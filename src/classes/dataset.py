'''
This module contains the definition of the Dataset class and some methods to create dates
'''

from datetime import date, timedelta
import json
import os
from random import choice

import pandas as pd
from tqdm import tqdm

from names_dataset import NameDataset

from classes.entities import Condition, Therapy
from classes.patient import Patient, PCondition
from classes.trial import Trial

from extractor import extract_condition_webpage, extract_therapy_webpage

def str_date_to_iso(str_date: str='20010101'):
    '''
    This function converst a string date to the ISO format.
    Input string must have the date in the format YYYYMMDD
    '''

    iso_str = f'{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}'
    date_iso = date.fromisoformat(iso_str)
    return date_iso

def random_date(start_date='20010101', end_date='20010101'):
    '''
    This function generates a random date between
    start_date and end_date. Both parameters must be
    strings in the format YYYYMMDD default is 20010101
    1st January 2001
    '''

    if end_date is None or end_date == 'None':
        end_date = date.today()
    if isinstance(start_date, str):
        start_date = str_date_to_iso(start_date)
    if isinstance(end_date, str):
        end_date = str_date_to_iso(end_date)

    delta = end_date - start_date
    rand_delta = choice(range(delta.days+1))
    rand_date = start_date + timedelta(days=rand_delta)
    return rand_date.strftime('%Y%m%d')

class Dataset():
    '''
    Class used to represent the dataset containing the Patients,
    Conditions and Therapies.

    Attributes
    ----------
    Conditions: list of Condition
        Conditions known in the dataset.
    Therapies: list of Therapy
        Therapies known in the dataset
    Patients: list of Patient
        Patient entries in the dataset
    '''

    def __init__(self):
        self.conditions = []
        self.therapies = []
        self.patients = []

    def make_utility_matrix(self, condition):
        '''
        This method creates the utility matrix using the data inside the dataset
        '''

        utility_matrix = pd.DataFrame(columns=[x.id for x in self.therapies])
        with tqdm(total=len(self.patients)) as pbar:
            pbar.set_description('Analyzing patients medical_history')
            for patient_entry in self.patients:  # For all patients in dataset
                medical_history = {}
                pcondition = patient_entry.get_pcondition(condition)
                if len(pcondition) > 0:  # Patient had that condition
                    for trial in patient_entry.trials:
                        if trial.condition == pcondition:
                            success = float(trial.successful.replace('%', ''))
                            if trial.therapy not in medical_history:
                                medical_history[trial.therapy] = success
                            else:
                                medical_history[trial.therapy] += success
                                medical_history[trial.therapy] /= 2
                    utility_matrix.loc[patient_entry.id] = medical_history
                pbar.update(1)
        pbar.close()

        # Drop empty columns -> avoid wasting space
        utility_matrix.dropna(how='all', axis=1, inplace=True)

        if 0 in utility_matrix.shape:
            print('No query matrix created, too less data.')
            return

        # Therapies weighted by their response over the patients
        for therapy in utility_matrix.columns:
            patients = utility_matrix.loc[:][therapy].copy()
            normalized_mean_therapy_success = patients.mean() / 100
            utility_matrix.loc[:][therapy] *= normalized_mean_therapy_success

        # Normalize rows -> subtract row mean from each patient score
        for patient_row in utility_matrix.index:
            utility_matrix.loc[patient_row] -= utility_matrix.loc[patient_row].mean()

        # Force maximum in the utility matrix to be 1
        utility_matrix /= max(utility_matrix.max())

        print(f'Utility matrix has size: {utility_matrix.shape}')
        return utility_matrix

    def get_condition(self, condition_id) -> str:
        '''
        This method is used to return the name of a condition given
        its id.

        Arguments
        ---------
        condition_id: str
            Is the unique identifier of the condition.

        Returns
        -------
        str:
            Is the name of the condition.
        '''

        for c in self.conditions:
            if c.id == condition_id:
                return c.name
        return ''

    def get_therapy(self, therapy_id):
        '''
        This method is used to return the name and type of a therapy given
        its id.

        Arguments
        ---------
        therapy_id: str
            Is the unique identifier of the therapy.

        Returns
        -------
        list of str:
            It contains the name and type of therapy in the first and second
            position respectively.
        '''

        for therapy in self.therapies:
            if therapy.id == therapy_id:
                return [therapy.name, therapy.type]
        return []

    def get_patient(self, patient_id):
        '''
        This method is used to retrive the patient with a specific id from the
        dataset.

        Arguments
        ---------
        patient_id: str
            Unique identifier of the patient in the dataset
        '''

        i = 0
        while self.patients[i].id != patient_id:
            i += 1
            if i == len(self.patients):
                print(f'No patient with ID {patient_id} in the dataset')
                return None
            else:
                return self.patients[i]

    def add_entry(self, entry):
        '''
        Adds an entry to the dataset.

        entry: Condition / Therapy / Patient
        '''

        if not isinstance(entry, (Condition, Therapy, Patient)):
            print('Invalid entry type.')
            return

        if isinstance(entry, Condition):
            self.conditions.append(entry)
        elif isinstance(entry, Therapy):
            self.therapies.append(entry)
        else:
            self.patients.append(entry)

    def save_in_json(self, file_path=os.path.dirname(os.path.dirname(__file__)),
                     name_file='Dataset.json'):
        '''
        Method used to save the class data into a .json file.

        Arguments
        ---------
        file_path: str
            Path in which to save the json file. If None the local one is
            assumed.
        name_file: str
            Is the name under which the file is saved.
        '''

        # data = {x: None for x in self.__dict__.keys()}
        data = {}
        data['Conditions'] = [x.to_dict() for x in self.conditions]
        data['Therapies'] = [x.to_dict() for x in self.therapies]
        data['Patients'] = [x.to_dict() for x in self.patients]

        if os.sep not in file_path:
            file_path = os.path.dirname(os.path.dirname(__file__)) + os.sep + file_path
        if file_path[-1] != os.sep:
            file_path += os.sep
        with open(file_path + name_file, 'w', encoding="utf-8") as json_stream_out:
            json.dump(data, json_stream_out, indent=4)
        json_stream_out.close()

    def from_json(self, file_path=os.path.dirname(os.path.dirname(__file__)),
                  name_file='Dataset.json'):
        '''
        Loads a json file data into the Dataset class.

        file_path: str
            Path from which the file is taken.
        name_file: str
            Name of the file from which to load
        '''

        if file_path is None or file_path == '':
            file_path = os.path.dirname(os.path.dirname(__file__))

        if file_path[-1] != os.sep:
            file_path += os.sep

        with open(file_path + name_file, 'r', encoding="utf-8") as json_stream_in:
            loaded_dataset = json.load(json_stream_in)
        json_stream_in.close()

        for key, value in loaded_dataset.items():
            if key == 'Conditions':
                for condition_json in value:
                    temp_c = Condition()
                    temp_c.from_dict(condition_json)
                    self.conditions.append(temp_c)
            elif key == 'Therapies':
                for therapy_json in value:
                    temp_t = Therapy()
                    temp_t.from_dict(therapy_json)
                    self.therapies.append(temp_t)
            elif key == 'Patients':
                for patient_json in value:
                    temp_p = Patient(patient_json['id'], patient_json['name'])
                    for c in patient_json['conditions']:
                        temp_pc = PCondition(c['id'], c['diagnosed'],
                                             c['cured'], c['kind'])
                        temp_p.conditions.append(temp_pc)
                    for t in patient_json['trials']:
                        temp_t = Trial(t['id'], t['start'], t['end'],
                                       t['condition'], t['therapy'],
                                       t['successful'])
                        temp_p.trials.append(temp_t)
                    self.patients.append(temp_p)

        tot_entries = sum(len(x) for x in loaded_dataset.values())
        print(f'Added {tot_entries} entries to the dataset')

    def random_fill(self, result_save_path, patients_num=0,
                    conditions_url='https://www.nhsinform.scot/illnesses-and-conditions/a-to-z',
                    therapies_url='https://en.wikipedia.org/wiki/List_of_therapies'):
        '''
        Method to randommly fill a Dataset class with random data.
        It uses a pool for the conditions and therapies names.
        The pools are generated by downloading the respective reference
        websites and process the webpages. For the download and
        processing the extractor.py script is used.
        Dataset is filled with random data and using a random
        number of Conditions, Therapies and patients. Whenever the
        Conditions and therapies pools run out the remaining part of
        the dataset is filled by Patient entries.

        patients_num: int
            Is the number of patients that must be inside the dataset
            together with all the available conditions and therapies.
        '''

        target_dir = os.path.join(result_save_path, 'temp')

        conditions_file = target_dir + os.sep + 'Conditions.html'
        conditions_name_pool = extract_condition_webpage(conditions_url, False, conditions_file)

        therapies_file = target_dir + os.sep + 'Therapies.html'
        therapies_name_pool = extract_therapy_webpage(therapies_url, False, therapies_file)

        names_pool = NameDataset()
        first_n_pool = list(names_pool.first_names)
        last_n_pool = list(names_pool.last_names)
        name_id_pool = list(range(1, len(first_n_pool)))

        # Add all conditions available in the dataset
        with tqdm(total=len(conditions_name_pool)) as pbar:
            pbar.set_description('Adding conditions to dataset')
            for condition in conditions_name_pool:
                cond_index = conditions_name_pool.index(condition)
                cond_id = f'Cond{cond_index + 1}'
                temp_cond = Condition(cond_id, condition[0], condition[1])
                self.add_entry(temp_cond)
                pbar.update(1)
        pbar.close()

        # Add all therapies available in the dataset
        with tqdm(total=len(therapies_name_pool)) as pbar:
            pbar.set_description('Adding therapies to dataset')
            for therapy_index, [therapy_name, therapy_kind] in enumerate(therapies_name_pool):
                therapy_id = f'Th{therapy_index+1}'
                temporary_therapy = Therapy(therapy_id, therapy_name, therapy_kind)
                self.add_entry(temporary_therapy)
                pbar.update(1)
        pbar.close()

        with tqdm(total=patients_num) as pbar:
            pbar.set_description('Adding patients to dataset')

            # Add the requested amount of patients
            for _ in range(patients_num):
                # Name and ID generation --------------------------------------
                first_name = choice(first_n_pool)
                last_name = choice(last_n_pool)
                name_id = choice(name_id_pool)
                name_id_pool.remove(name_id)  # Ensures uniqueness of ID

                # Patient class instance --------------------------------------
                rand_patient = Patient(name_id, last_name + ' ' + first_name)

                # Age and ranges determination --------------------------------
                max_livable_years = 100  # Humans live around 100 years

                age = choice(range(max_livable_years + 1))

                birth_date = date.today()
                life_delta = timedelta(days=365*age)
                birth_date -= life_delta

                death = choice(range((max_livable_years+1)-age))
                death_date = birth_date + timedelta(days=(365*death)+365)

                # Fill medical history of conditions -> max 10 conditions -----
                for num_cond in range(1, choice(range(2, 11))):
                    rand_cond = choice(self.conditions)

                    rand_d_date = random_date(birth_date, death_date)
                    rand_c_date = None
                    if choice(range(1, 11, 1)) >= 2:  # 20% chance no cured
                        rand_c_date = random_date(rand_d_date, death_date)
                    if rand_c_date is None:
                        rand_c_date = str(rand_c_date)  # Saves 'None'
                    temp_pcond = PCondition(f'pc{num_cond}',
                                            rand_d_date, rand_c_date,
                                            rand_cond.id)
                    rand_patient.add_condition(temp_pcond)

                # Fill trials history -----------------------------------------
                for cond in rand_patient.conditions:
                    last_trial = None
                    cured = False
                    for trial_num in range(1, choice(range(2, 11))):  # max 10 trials
                        chosen_therapies_pool = []
                        if not cured:
                            start_cond = cond.diagnosed
                            end_cond = cond.cured
                            if end_cond == 'None':
                                end_cond = death_date

                            if last_trial is None:  # Initialize start of trial
                                last_trial = random_date(start_cond, end_cond)
                            end_trial = random_date(last_trial, end_cond)

                            temp_th = choice(self.therapies)

                            # Avoid duplicated therapies
                            if temp_th.id in chosen_therapies_pool:
                                while temp_th.id in chosen_therapies_pool:
                                    temp_th = choice(self.therapies)
                            chosen_therapies_pool.append(temp_th.id)

                            success = choice(range(0, 101, 1)) / 100

                            if success == 1:
                                if rand_c_date != 'None':
                                    cured = True
                                else:
                                    success = choice(range(0, 100, 1)) / 100
                            temp_trial = Trial(f'tr{trial_num}',
                                               last_trial, end_trial,
                                               cond.id, temp_th.id,
                                               success)
                            rand_patient.add_trial(temp_trial)
                            last_trial = end_trial  # No retroactive trials
                self.add_entry(rand_patient)
                pbar.update(1)
        pbar.close()

    def sample_not_cured(self, num_to_sample=1, save_to_json=False):
        '''
        This method is used to randomly sample N patients from the dataset
        which have at least one condition that is not cured yet.

        Arguments
        ---------
        num_to_sample: int
            Integer number representing the number of patients to sample
        save_to_json: bool
            Flags whether or not to save the sampled patients ids into as
            a list in a .json file.
        '''

        num_pat = len(self.patients)
        if num_pat == 0:
            print('No patients in the dataset.\n'
                  'Load a dataset or random fill it')
            return []
        if num_pat < num_to_sample:
            print('Not enough patients in the dataset')
            return []

        patients_not_cured = []
        for patient_data in self.patients:
            for cond in patient_data.conditions:
                if cond.cured == 'None':
                    patients_not_cured.append([patient_data.id, cond.id])
                    break  # Ensures one condion per patient

        random_sampled_patients = []
        for _ in range(num_to_sample):
            chosen = choice(patients_not_cured)
            random_sampled_patients.append(chosen)
        if save_to_json:
            target_dir = os.path.abspath('../data')
            file_name = f'{num_to_sample}_uncured_patients.json'
            with open(target_dir + os.sep + file_name, 'w', encoding="utf-8") as stream_out:
                json.dump(random_sampled_patients, stream_out, indent=4)
                stream_out.close()
        return random_sampled_patients

    def print(self, short=True):
        '''
        Print method for the Dataset class
        '''

        size = [len(self.conditions), len(self.therapies), len(self.patients)]
        print(f'Dataset Size: {size[0]} Conditions | {size[1]} Therapies | {size[2]} Patients\n')
        if not short:
            for key, value in self.__dict__.items():
                print('-'*80, f'\n{key}:\n')
                for entry in value:
                    entry.print()
                    print('\n')
