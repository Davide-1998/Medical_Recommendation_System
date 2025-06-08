'''
Main script for evaluating the medical recommendation system
'''

import argparse
from datetime import date, timedelta
import json
from math import sqrt
import os
from random import choice
from time import time

import pandas as pd
from tqdm import tqdm

from names_dataset import NameDataset
from sklearn.metrics import mean_squared_error as RMSE
import matplotlib.pyplot as plt

from extractor import extract_condition_webpage, extract_therapy_webpage


def exists_or_create_directory(directory_path: str, access_mode: int=0o777) -> None:
    '''
    Function that checks the existance of a directory,
    otherwise it creates it with the specified acess mode.
    '''
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, access_mode)
        print(f'Created: {directory_path}')


def stat_interaction(stats_directory_path, filename, what_to_save=None):
    '''
    Function to open a stream towards a file of statistics data.
    if the parameter 'what_to_save' if populated with an object that
    can be converted to string, then it will be appended to the contents 
    of the file, otherwise the file will be read
    '''

    data = []
    exists_or_create_directory(stats_directory_path)
    stream_mode = 'a' if what_to_save is not None else 'r'
    filepath = os.path.join(stats_directory_path, filename)
    with open(filepath, stream_mode, encoding="utf-8") as stream_stat:
        # Save stats file
        if what_to_save is not None:
            stream_stat.write(f'{what_to_save}\n')
        else:
            # Read from stats file
            for line in stream_stat:
                data.append(float(line.strip()))
        stream_stat.close()
    return data


class Entity():
    '''
    This is the base class that is inherited from the others.
    It provides methods to retireve an print the data.

    Attributes
    ----------
    id: str
        Unique identifier of the entity.
    name: str
        Is the string representing the name of the entity.
    type: str
        Is the string representing the type of entity.
        It's the only optional attribute.
   '''

    def __init__(self, _id=None, _name=None, _type=None):
        '''
        _id: str
        _name: str
        _type: str -> optional. Add it using .update_type(_type)
        '''

        self.id = str(_id)
        self.name = str(_name)
        self.type = _type

    def update_type(self, _type):
        '''
        _type: str
            Type of Therapy or Condition.
        '''

        if 'type' in self.__dict__:
            print(f'The type of {self.id} is changed from \'{self.type}\' to \'{_type}\'')
        self.type = str(_type)

    def to_dict(self):
        '''
        Returns the data contained in the class as a dictionary.

        Returns
        -------
        dict
        '''

        return self.__dict__

    def from_dict(self, loaded_dict):
        '''
        Method to load data from a Dicitonary into the class.

        loaded_dict: dict
            Its keys must be: id, name, type.
        '''

        for key, value in loaded_dict.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
            elif key == 'type':
                self.update_type(value)

    def print(self, indent=False):
        '''
        Prints the informations contained in the class.

        indent: bool
            Adds a tab spacing in the formatting of the string.
        '''

        for key, value in self.__dict__.items():
            print(f"{'\t' if indent else ''}{key:<6} : {value:<10}")


class Condition(Entity):
    '''
    Is the class representing the conditions.
    It inherits directly from the Entity class.

    Attributes
    ----------
    id: str
        Unique idetifier of the condition.
    name: str
        Name of the condition.
    type: str
        Type of condition.
    '''

    def __init__(self, _id=None, _name=None, _type=None):
        super().__init__(_id, _name, _type)


class Therapy(Entity):
    '''
    Is the class representing the conditions.
    It inherits directly from the Entity class

    Attributes
    ----------
    id: str
        Unique idetifier of the therapy.
    name: str
        Name of the therapy.
    type: str
        Type of therapy.
   '''

    def __init__(self, _id=None, _name=None, _type=None):
        super().__init__(_id, _name, _type)


class Trial():
    '''
    Is the class representing the trials of therapies made over the
    patients for a given condition.

    Attributes
    ---------
    id: str
        Unique identifier of the trial
    start: int
        An integer number representing the starting date of treatment.
        The format used is yyyymmdd .
    end: int
        An integer number representing the ending date of treatment.
        The format used is yyyymmdd .
    condition: str
        Is a string representing the condition's unique identifier for
        which the trial was assigned.
    therapy: str
        Is a string representing the therapy's unique identifier which
        is used in the trial.
    successful: str
        Is the percentage of success that the trial had.
    '''

    def __init__(self, _id=None, start=0, end=0, cond_id=None, th_id=None,
                 success=0.0):
        '''
        _id: str
            Unique identifier of the Trial
        start: int
            Date in which trial started. Format: yyyymmdd
        end: int
            Date in which trial ended. Format yyyymmdd.
        cond_id: str
            Unique identifier of the condition for which the trial is
            proposed.
        th_id: str
            Unique identifier of the therapy in use.
        success: float
            Rate of success of the therapy. Must be between 0 and 1.
        '''

        self.id = str(_id)
        self.start = str(start)
        self.end = str(end)
        self.condition = str(cond_id)
        self.therapy = str(th_id)

        self.add_success(success)

    def to_dict(self):
        '''
        Returns the class dictionary
        '''

        return self.__dict__

    def add_success(self, success):
        '''
        Add a success rate and converts it into a percentage.
        Values higher than 1 are shifted in the 0-1 interval.
        '''

        if isinstance(success, str):  # Avoid error during load of percentage
            success = float(success.replace('%', ''))
        if int(success) >= 1:
            success = int(success) / 100
        self.successful = f'{success:0.2%}'

    def from_dict(self, loaded_dict):
        '''
        Method to load data coming from a loaded dictionary into the
        class.

        loaded_dict: dict
            Its keys must be: id, start, end, condition, therapy and
            succesful.
        '''

        for key, value in loaded_dict:
            if key in self.__dict__.keys():
                self.__dict__[key] = value

    def print(self, indent=False):
        '''
        Prints the class attributes data.

        indent: bool
            Adds a tab spacing in the formatting of the string.
        '''

        for key, value in self.__dict__.items():
            print(f"{'\t' if indent else ''}{key:<10} : {value:<10}")


class PCondition():
    '''
    Is the class representing the condition that a patient has
    or had.

    Attributes
    ----------
    id: str
        Unique identifier of patient condition
    diagnosed: int
        The date in which the condition was first diagnosed
    cured: int
        The date in which the condition was cured
    kind: str
        Is the unique identifier of the Condition class to which the
        condition belongs to. (i.e. Condition.id )
    '''

    def __init__(self, _id=None, d_date='None', c_date='None', kind=None):
        '''
        _id: str
            Unique identifier of the specific condition.
            Condition - Patient relation.
        d_date: int
            Date of first diagnosed.
        c_date: int
            Date of cure.
        kind: Condition.id
            Id of condition. Condition - Condition pool relation.
        '''

        self.id = str(_id)
        self.diagnosed = str(d_date)
        self.cured = str(c_date)
        self.kind = str(kind)

    def set_cured(self, c_date=0):
        '''
        This method is used to set the date of when the condition is
        cured.

        c_date: int
            Date in which the condition is considered cured.
        '''

        self.cured = int(c_date)

    def to_dict(self):
        '''
        This method returns a json serializable version of the class.

        Return
        ------
        dict
        '''

        return self.__dict__

    def from_dict(self, loaded_dict):
        '''
        Method to load data coming from a loaded dictionary into the
        class.

        loaded_dict: dict
            Its keys must be: id, diagnosed, cured and kind.
        '''

        for key, value in loaded_dict:
            if key in self.__dict__.keys():
                self.__dict__[key] = str(value)

    def print(self, indent=False):
        '''
        Prints the elements in the class.

        indent: bool
            Adds a tab spacing in the formatting of the string.
        '''

        if not indent:
            for key, value in self.__dict__.items():
                print('{:<10} : {:<10}'.format(key, value))
        else:
            for key, value in self.__dict__.items():
                print('\t{:<10} : {:<10}'.format(key, value))


class Patient(Entity):
    '''
    Is the class reresenting the Patients.

    Attributes
    ----------
    id: str
        Unique identifier of the patient -> Inherited by Entity class.
    name: str
        Name of the patient -> Inherited by Entity class.
    conditions: list of PCondition
        Is the list containing the conditions to which the patient
        was subjected to. Is its medical history.
    trials: list of Trial
        Is the list of trials the patient undergone.
    '''

    def __init__(self, _id=None, _name=None):
        super().__init__(_id, _name)
        self.id = str(self.id)
        self.conditions = []
        self.trials = []

    def add_trial(self, trial):
        '''
        Adds a Trial element ot the trials list.

        trial: Trial
        '''

        self.trials.append(trial)

    def add_condition(self, pcondition):
        '''
        Adds a condition to the Condition list.

        condition: PCondition
        '''

        self.conditions.append(pcondition)

    def to_dict(self):
        '''
        Retrieve the data of the class gathering them in a dictionary.

        Returns
        -------
        dict
        '''

        data = self.__dict__
        data['conditions'] = [x.to_dict() for x in self.conditions]
        data['trials'] = [x.to_dict() for x in self.trials]
        return data

    def save_in_json(self, name_file=None):
        '''
        Method to save the Patient class in a .json file on the device.

        name_file: str
            FileName in which to save the class. If None a combination
            of the id and the name will be used as the name. If
            a string is given, if it doesn't contain a path, the local
            one is assumed: os.getcwd() .
        '''

        data = self.__dict__
        if name_file is None:
            name_file = os.getcwd() + os.sep
            name_file += f'{self.id}_{self.name}'
        else:
            if os.sep not in name_file:
                name_file = os.getcwd() + os.sep + name_file
            else:
                if not os.path.isfile(name_file):
                    print(f'No file \'{name_file}\' found.')
        if '.json' not in name_file:
            name_file += '.json'

        with open(name_file, 'w', encoding="utf-8") as stream_out:
            json.dump(data, stream_out, indent=4)
            stream_out.close()

    def from_json(self, name_file, path_to_file=None):
        '''
        This method allows to load a json file of a patient in the
        class.

        name_file: str
            Is the name of the file from which data will be collected.
        path_to_file: str
            Filepath in which the file is located. If None the local
            one is assumed: os.getcwd()
        '''

        if path_to_file is None:
            path_to_file = os.getcwd() + os.sep

        with open(path_to_file + name_file, 'r', encoding="utf-8") as json_stream_in:
            loaded_patient = json.load(json_stream_in)
        json_stream_in.close()

        self.from_dict(loaded_patient)

    def from_dict(self, loaded_dict):
        '''
        Loads the data coming from a loaded dictionary into the class.

        loaded_dict: dict
        '''

        self.id = str(loaded_dict['id'])
        self.name = str(loaded_dict['name'])

        for c in loaded_dict['conditions']:
            temp_c = PCondition(c['id'], c['diagnosed'], c['cured'], c['kind'])
            self.conditions.append(temp_c)
        for t in loaded_dict['trials']:
            temp_t = Trial(t['id'], t['start'], t['end'], t['condition'],
                           t['therapy'], t['successful'])
            self.trials.append(temp_t)

    def get_condition(self, pcondition_id) -> str:
        '''
        This method allows to return the Condition id as in the dataset

        Arguments
        ---------
        pcondition_id: str
            Unique identifier of the pcondition in the patient class
        '''

        i = 0
        while self.conditions[i].id != pcondition_id:
            i += 1
            if i == len(self.conditions):
                print(f'No condition {pcondition_id} in dataset')
                return ''
        return self.conditions[i].kind

    def get_pcondition(self, condition_id):
        '''
        This method is used to translate an input condition to a Pcondition id.

        Arguments
        ---------
        condition_id: str
            Is the unique identifier of the condition to search for
        '''

        for pc in self.conditions:
            if pc.kind == condition_id:
                return pc.id
        return

    def print(self):
        '''
        Method used to print all the informations in the patient class.
        '''

        for key, value in self.__dict__.items():
            if not isinstance(value, list):
                print(f'{key:<10} : {value:<10}')
            else:
                print(f'{key} :', '-'*31)
                for t in value:
                    t.print(indent=True)
                    print('\t', '-'*15, '*', '-'*15)


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
        self.Conditions = []
        self.Therapies = []
        self.Patients = []

    def make_utility_matrix(self, condition):
        utility_matrix = pd.DataFrame(columns=[x.id for x in self.Therapies])
        with tqdm(total=len(self.Patients)) as pbar:
            pbar.set_description('Analyzing patients medical_history')
            for patient_entry in self.Patients:  # For all patients in dataset
                medical_history = {}
                pcondition = patient_entry.get_pcondition(condition)
                if pcondition is not None:  # Patient had that condition
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

        for c in self.Conditions:
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

        for th in self.Therapies:
            if th.id == therapy_id:
                return [th.name, th.type]
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
        while self.Patients[i].id != patient_id:
            i += 1
            if i == len(self.Patients):
                print('No patient with ID {} in the dataset'
                      .format(patient_id))
                return None
        else:
            return self.Patients[i]

    def add_entry(self, entry):
        '''
        Adds an entry to the dataset.

        entry: Condition / Therapy / Patient
        '''

        if not isinstance(entry, (Condition, Therapy, Patient)):
            print('Invalid entry type.')
            return

        if isinstance(entry, Condition):
            self.Conditions.append(entry)
        elif isinstance(entry, Therapy):
            self.Therapies.append(entry)
        else:
            self.Patients.append(entry)

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
        data['Conditions'] = [x.to_dict() for x in self.Conditions]
        data['Therapies'] = [x.to_dict() for x in self.Therapies]
        data['Patients'] = [x.to_dict() for x in self.Patients]

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
                for el in value:
                    temp_c = Condition()
                    temp_c.from_dict(el)
                    self.Conditions.append(temp_c)
            elif key == 'Therapies':
                for el in value:
                    temp_t = Therapy()
                    temp_t.from_dict(el)
                    self.Therapies.append(temp_t)
            elif key == 'Patients':
                for el in value:
                    temp_p = Patient(el['id'], el['name'])
                    for c in el['conditions']:
                        temp_pc = PCondition(c['id'], c['diagnosed'],
                                             c['cured'], c['kind'])
                        temp_p.conditions.append(temp_pc)
                    for t in el['trials']:
                        temp_t = Trial(t['id'], t['start'], t['end'],
                                       t['condition'], t['therapy'],
                                       t['successful'])
                        temp_p.trials.append(temp_t)
                    # temp_p.from_dict(el)
                    self.Patients.append(temp_p)

        tot_entries = sum(len(x) for x in loaded_dataset.values())
        print(f'Added {tot_entries} entries to the dataset')

    def random_fill(self, result_save_path, patients_num=0):
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
        conditions_url = 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z'
        conditions_name_pool = extract_condition_webpage(conditions_url, False, conditions_file)

        therapies_file = target_dir + os.sep + 'Therapies.html'
        therapies_url = 'https://en.wikipedia.org/wiki/List_of_therapies'
        therapies_name_pool = extract_therapy_webpage(therapies_url, False, therapies_file)

        names_pool = NameDataset()
        firstN_pool = list(names_pool.first_names)
        lastN_pool = list(names_pool.last_names)
        nameID_pool = list(range(1, len(firstN_pool)))

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
            for i in range(len(therapies_name_pool)):
                ther_id = f'Th{i+1}'
                therapy = therapies_name_pool[i]
                temp_ther = Therapy(ther_id, therapy[0], therapy[1])
                self.add_entry(temp_ther)
                pbar.update(1)
        pbar.close()

        with tqdm(total=patients_num) as pbar:
            pbar.set_description('Adding patients to dataset')

            # Add the requested amount of patients
            for i in range(patients_num):
                # Name and ID generation --------------------------------------
                first_name = choice(firstN_pool)
                last_name = choice(lastN_pool)
                name_id = choice(nameID_pool)
                nameID_pool.remove(name_id)  # Ensures uniqueness of ID

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
                    rand_cond = choice(self.Conditions)

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

                            temp_th = choice(self.Therapies)

                            # Avoid duplicated therapies
                            if temp_th.id in chosen_therapies_pool:
                                while temp_th.id in chosen_therapies_pool:
                                    temp_th = choice(self.Therapies)
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

    def sample_not_cured(self, num_to_sample=1, saveToJson=False):
        '''
        This method is used to randomly sample N patients from the dataset
        which have at least one condition that is not cured yet.

        Arguments
        ---------
        num_to_sample: int
            Integer number representing the number of patients to sample
        saveToJson: bool
            Flags whether or not to save the sampled patients ids into as
            a list in a .json file.
        '''

        num_pat = len(self.Patients)
        if num_pat == 0:
            print('No patients in the dataset.\n'
                  'Load a dataset or random fill it')
            return
        if num_pat < num_to_sample:
            print('Not enough patients in the dataset')
            return

        patients_not_cured = []
        for patient in self.Patients:
            for cond in patient.conditions:
                if cond.cured == 'None':
                    patients_not_cured.append([patient.id, cond.id])
                    break  # Ensures one condion per patient

        random_sampled_patients = []
        for i in range(num_to_sample):
            chosen = choice(patients_not_cured)
            random_sampled_patients.append(chosen)
        if saveToJson:
            target_dir = os.path.abspath('../data')
            fileName = '%d_uncured_patients.json' % num_to_sample
            stream_out = open(target_dir + os.sep + fileName, 'w')
            json.dump(random_sampled_patients, stream_out, indent=4)
            stream_out.close()
        return random_sampled_patients

    def print(self, short=True):
        size = [len(self.Conditions), len(self.Therapies), len(self.Patients)]
        print(f'Dataset Size: {size[0]} Conditions | {size[1]} Therapies | {size[2]} Patients\n')
        if not short:
            for key, value in self.__dict__.items():
                print('-'*80, f'\n{key}:\n')
                for entry in value:
                    entry.print()
                    print('\n')


def strDate_to_iso(str_date: str='20010101'):
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
        start_date = strDate_to_iso(start_date)
    if isinstance(end_date, str):
        end_date = strDate_to_iso(end_date)

    delta = end_date - start_date
    rand_delta = choice(range(delta.days+1))
    rand_date = start_date + timedelta(days=rand_delta)
    return rand_date.strftime('%Y%m%d')


def similarity_computation(utility_matrix, query_patient):
    '''
    This method computes the similarity score among the patients
    contained in the dataset.

    Arguments
    ---------
    utility_matrix: pandas.DataFrame
        Is the DataFrame representing the available data
    query_patient: str
        Is the unique identifier of the patient for which the recommendations
        are made.

    Return
    ------
    (pandas.Series, pandas.DataFrame)
        The first is a Serie representing the similarity among the patients in
        the dataset. The latter is the dataframe containing the nearest
        neighbours for the given query patient.
    '''

    
    # Cosine similarity is inefficient in case of sparse data:
    # https://stackoverflow.com/questions/45387476/
    #     cosine-similarity-between-each-row-in-a-dataframe-in-python
    # 
    # The euclidean distance in N-dimensions is used. The NaN inside the
    # vectors are temporary filled with 0 in order to avoid mismatch in
    # shape.
    # N.B:
    # If the results of similarities are 0 this is due to the subtraction of
    # the mean, meaning that each patient having the condition and using
    # only one trial is similar to the query one. This may be correct
    # considering that other patients may have cured the desease with just
    # one trial. -> Must check and give it more weight next.
    

    patient_vec = utility_matrix.loc[query_patient].copy()
    similarities = pd.DataFrame(columns=['euclidean'])

    # Try N-dimensional euclidean distance
    query_dist = patient_vec.copy().fillna(0)
    for patient in utility_matrix.index:
        if patient != query_patient:
            patient_dist = utility_matrix.loc[patient].copy().fillna(0)
            square_distance = sum((patient_dist - query_dist)**2)

            # Cosine sim:
            # cos_sim = cosine_similarity([query_dist, patient_dist])[0, 1]

            similarities.loc[patient] = sqrt(square_distance)

    # Try cosine similarity
    # cosine_sim = cosine_similarity(utility_matrix.copy().fillna(0))
    # i = 0
    # while utility_matrix.index[i] != query_patient:
    #     i += 1
    # cosine = pd.Series(cosine_sim[i, :], index=utility_matrix.index)
    # print(list(cosine_sim[i, :]).count(0))
    # similarities['cosine'] = cosine

    # Retain nearest neighbours:
    euclidean_threshold = sqrt(sum(query_dist**2))/2
    euclidean_sim = similarities['euclidean'] <= euclidean_threshold
    nearest_neighbours = similarities[euclidean_sim].copy()
    nearest_neighbours = nearest_neighbours.dropna()

    # Find patients which answered as query if it had tried at least 1 th
    biology_vec = patient_vec.copy().dropna()
    response_tolerance = 0.1  # 10% tolerance in the values
    if len(biology_vec.index) > 0:
        same_biology = pd.DataFrame(columns=biology_vec.index)
        lb = biology_vec * (1 - response_tolerance)  # Lower bound of tolerance
        ub = biology_vec * (1 + response_tolerance)  # Upper bound of tolerance
        th_filter = biology_vec.index
        for patient in utility_matrix.index:
            if patient != query_patient:
                utility_patient = utility_matrix.loc[patient][th_filter].copy()
                retained = utility_patient[utility_patient.between(lb, ub)]
                same_biology.loc[patient] = retained
        # same_biology = same_biology[same_biology[:] == biology_vec[:]].dropna()
        same_biology = same_biology.dropna()
        print(f'Same biology matrix has size: {same_biology.shape}')

        intersection = nearest_neighbours.filter(same_biology.index,
                                                 axis=0).copy()
        # Increase (relatively) the similarity for those patients that has a
        # Similar biology
        for patient in intersection.index:
            nearest_neighbours.loc[patient]['euclidean'] -= 1
        # similarities.loc[patient]['cosine'] *= 2

    return nearest_neighbours


def check_fullna(obj, return_mean=False):
    '''
    Function to remove NaN elements from a pandas table
    '''


    to_return = None
    if isinstance(obj, pd.DataFrame):
        not_nan_shape = obj.dropna(how='all').shape
        if 0 in not_nan_shape:
            return pd.Series([0 for i in range(not_nan_shape[1])])
        to_return = obj

    elif isinstance(obj, pd.Series):
        if len(obj.dropna()) == 0:
            return 0
        to_return = obj

    if return_mean:
        return to_return.mean()
    return to_return


def baseline_computation(utility_matrix, query_patient, query_therapy):
    '''
    This method is used to compute the baseline for the ratings.

    Arguments
    ---------
    utility_matrix: pandas.DataFrame
        Is the DataFrame representing the dataset
    query_patient: str
        Is the unique identifier of the patient for which a recommendation is
        requested
    query_therapy: str
        Is the unique identifier of the therapy under analysis

    Returns
    -------
    baseline: float64
        Is the results of the computations.
    '''

    matrix_mean = check_fullna(utility_matrix, True)
    overall_mean = sum(matrix_mean)/len(utility_matrix.columns)

    q_patient = utility_matrix.loc[query_patient]
    query_patient_therapies_mean = check_fullna(q_patient, True)

    therapy_mean = check_fullna(utility_matrix[query_therapy], True)

    baseline = overall_mean + \
        (query_patient_therapies_mean - overall_mean) + \
        (therapy_mean - overall_mean)
    return baseline


def rating_computation(utility_matrix, query_patient, nearest_neighbours,
                       specific_therapy=None):
    '''
    This method is used to compute the ratings of therapies for a specific
    query patient.

    Arguments
    ---------
    utility_matrix: pandas.DataFrame
        Is the dataset of patients and therapies (rows and columns) available.
    query_patient: str
        Is the unique identifier of the patient for which the recommendation is
        made.
    nearest_neighbours: pandas.DataFrame
        Is the dataset containing the nearest neighbours for the given patient
        with their similarity scores dependant by the score used.
        On the rows are the id of the patients, on the columns the similarity
        scores.
    specific_therapy: str
        Is the ID of a specific therapy for which a rating is required.
        This option is only used during evaluation.

    Returns
    -------
    pandas.DataFrame
        Is the dataset containing the recommendations.
        On the rows are the therapies id, while on the columns are the name
        of the therapy, the type of therapy and their ratings.
        By guidelines, it has 5 rows.
    '''

    patient_vec = utility_matrix.loc[query_patient].copy()
    if specific_therapy is not None:
        patient_vec = patient_vec.filter(specific_therapy)
    rating_vector = patient_vec.copy()
    rating_vector_nan = rating_vector.isna()
    for therapy in patient_vec.index:
        if rating_vector_nan[therapy]:
            baseline = baseline_computation(utility_matrix,
                                            query_patient,
                                            therapy)
            rating = 0.0
            sim_sum = 0.0
            for patient_score in nearest_neighbours.index:
                loc_patient_nan = utility_matrix.loc[patient_score].isna()
                loc_patient_score = utility_matrix.loc[patient_score][therapy]
                loc_patient_sim = nearest_neighbours.loc[patient_score]['euclidean']
                if not loc_patient_nan[therapy]:
                    partial_rating = loc_patient_score - \
                                     baseline_computation(utility_matrix,
                                                          patient_score,
                                                          therapy)
                    # Absolute used due to euclidean distance
                    partial_rating *= (1 - abs(loc_patient_sim))
                    rating += partial_rating
                sim_sum = sum(1 - abs(nearest_neighbours['euclidean']))
            if sim_sum != 0:
                rating /= sim_sum
            else:
                rating /= 1
            rating_vector[therapy] = baseline + rating
    rating_vector = rating_vector.sort_values(ascending=False)

    # Remove already done therapies
    recommending_therapies = set(rating_vector.index)
    performed_therapies = set(patient_vec.dropna().index)
    intersection = recommending_therapies.intersection(performed_therapies)

    rating_vector = rating_vector.drop(labels=intersection)
    return rating_vector.iloc[:5].copy().fillna(0)


def medical_recommendation(dataset, patients_ids, query_condition, stats_directory_path):
    '''
    This method is used to compute the best recommendations for each patient
    given a specific condition.
    The workflow is:
        - Load the reference dataset
        - Generate the matrix of therapies vs patients
        - Normalize the results
        - Fill the gaps
        - Retain only similar patients
        - return the recommendations

    Arguments
    ---------
    dataset: string
        It is the path to the dataset in .json to load.
    patients_ids: list of strings
        It is the list of patients ids for which a recommendation must be done.
    query_condition: string
        It is the condition id for which the recommendation is required.

    Returns:
    therapy_ans: dict
        Is the dictionary containing the suggested therapies for the patient
        to the condition.
        It contains: {query_patient_id: pandas.DataFrame}
    '''

    if isinstance(dataset, str):  # A path is given
        path = os.path.dirname(dataset)
        name_file = os.path.basename(dataset)
        reference_dataset = Dataset()
        reference_dataset.from_json(path, name_file)
    elif isinstance(dataset, Dataset):
        print('A Dataset class has been passed. \nNo loading action.')
    else:
        print(f'The dataset type <{type(dataset)}> is not supported')
        return

    therapy_ans = {}

    if len(patients_ids) == 1 and os.sep in patients_ids[0]:  # Is file
        file_path = patients_ids[0]
        if '.txt' in file_path:
            with open(file_path, 'r', encoding="utf-8") as patients_stream:
                id_list = []
                pc_list = []
                for line in patients_stream:
                    data = line.split('\t')
                    if len(data) > 2:
                        id_list.append(data[0])
                        pc_list.append(data[-1].replace('\n', ''))
            patients_stream.close()
            patients_ids = id_list
            query_condition = pc_list

    for p_id in patients_ids:  # For all query patient
        start_time = time()

        query_patient = reference_dataset.get_patient(p_id)
        if query_patient is None:
            return
        idx = patients_ids.index(p_id)
        condition_id = query_patient.get_condition(query_condition[idx])

        condition_name = reference_dataset.get_condition(condition_id)

        if condition_id == '' or condition_name == '':
            continue

        print(f'Started Recommendation for patient \'{p_id}\' and condition \'{condition_name}\':')

        query_matrix = reference_dataset.make_utility_matrix(condition_id)

        # Compute similarity in the DataFrame
        nearest_neighbours = similarity_computation(query_matrix,
                                                    query_patient.id)

        # Start Ratings
        recommend = rating_computation(query_matrix, query_patient.id,
                                       nearest_neighbours)
        answer_cols = ['Name', 'Type', 'Rating']
        ans = pd.DataFrame(columns=answer_cols)

        for therapy_id, therapy_rating in recommend.items():
            therapy_name, therapy_kind = reference_dataset.get_therapy(therapy_id)
            ans.loc[therapy_id] = pd.Series([therapy_name, therapy_kind, therapy_rating],
                                            index=answer_cols)

        therapy_ans[query_patient.id] = ans

        # Saving time statistics
        tot_time = time() - start_time
        stat_interaction(stats_directory_path,
                         f'run_time_{len(reference_dataset.Patients)}_patients.txt',
                         tot_time)
        print(f'Recommendation time: {tot_time:0.4f}[s]')
    return therapy_ans


def random_sample_patient(utility_matrix):
    '''
    This method is used to randomly sample a patient from the ones contained
    inside the utility matrix generated for a specific disease.
    It uses a recursive approach to provide a patient which had underwent
    at least one trial to cure itself.

    Arguments
    ---------
    utility_matrix: pandas.DataFrame
        Is the utility matrix generated for a specific condition.

    Return
    ------
    pandas.Series
        Is the ratings vector of the randomly chosen patient.
    '''

    query_patient = choice([x for x in utility_matrix.index])
    patient_vec = utility_matrix.loc[query_patient].dropna()
    if patient_vec.empty:
        patient_vec = random_sample_patient(utility_matrix)
    return patient_vec


def evaluation(stats_save_directory, dataset_list='None', iterations=3, random_fill_save_path=os.getcwd()):
    '''
    This method is used to evaluate the recommendation system proposed.
    '''

    print('Evaluation Started')
    loaded_datasets = {}  # List of Dataset classes
    eval_results_dict = {}

    if dataset_list is None:
        patients_nums = [25, 50, 100, 150]
        modifier = 1000
        for patient_index in patients_nums:
            idx = patients_nums.index(patient_index) + 1
            print(f'Random filling of dataset {idx}/{len(patients_nums)}')
            temporary_dataset = Dataset()
            temporary_dataset.random_fill(os.path.join(random_fill_save_path, 'temp'),
                                          patient_index*modifier)
            key = f'rand_{idx}' + f'_{patient_index*modifier}'
            loaded_datasets[key] = temporary_dataset
            eval_results_dict[key] = {'time': [], 'rmse': []}
    else:
        for path in dataset_list:
            print(f'Loading dataset {dataset_list.index(path) + 1}/{len(dataset_list)}')
            path_to_dataset = os.path.dirname(path)
            namefile = os.path.basename(path)
            temporary_dataset = Dataset()
            temporary_dataset.from_json(path_to_dataset, namefile)
            key = namefile.replace('.json', '') + f'_{len(temporary_dataset.Patients)}'

            loaded_datasets[key] = temporary_dataset
            eval_results_dict[key] = {'time': [], 'rmse': []}

    # Start gathering evaluation data
    for dataset_name, dataset in loaded_datasets.items():
        for _ in range(int(iterations)):
            num_patients = len(dataset.Patients)
            condition_pool = [condition.id for condition in dataset.Conditions]

            # Chose random condition
            query_condition = choice(condition_pool)
            print(f'Condition in evaluation: {query_condition}')

            start_time = time()

            # Make utility matrix for that condition
            utility_matrix = dataset.make_utility_matrix(query_condition)

            # Chose random patient and therapy
            query_patient = random_sample_patient(utility_matrix)

            query_therapy = choice([[x, y] for x, y in query_patient.items()])
            true_value = query_therapy[1]

            # Compute Nearest Neighbours
            nn_matrix = similarity_computation(utility_matrix,
                                               query_patient.name)

            # Compute Ratings
            ghost_utility = utility_matrix.copy()
            ghost_patient = ghost_utility.loc[query_patient.name]
            ghost_patient = ghost_patient.drop(query_therapy[0])
            ghost_utility.loc[query_patient.name] = ghost_patient

            therapy_rating_prediction = rating_computation(ghost_utility,
                                                           query_patient.name,
                                                           nn_matrix,
                                                           query_therapy)

            pred_value = therapy_rating_prediction[query_therapy[0]]

            tot_time = time() - start_time
            stat_interaction(stats_save_directory,
                             f'run_time_{num_patients}_patients.txt',
                             tot_time)

            rmse = RMSE([true_value], [pred_value])
            stat_interaction(stats_save_directory,
                             f'rmse_scores_{num_patients}_patients.txt',
                             rmse)

            eval_results_dict[dataset_name]['time'].append(tot_time)
            eval_results_dict[dataset_name]['rmse'].append(rmse)

    # Make summary dataframe
    eval_results = pd.DataFrame(columns=['mean rmse', 'mean time'])
    for num_patients, eval_dict in eval_results_dict.items():
        len_time = len(eval_dict['time'])
        len_rmse = len(eval_dict['rmse'])

        means = {'mean time': sum(eval_dict['time']) / len_time,
                 'mean rmse': sum(eval_dict['rmse']) / len_rmse}
        eval_results.loc[num_patients] = pd.Series(means)

        # Make plots
        fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=120)
        fig.suptitle(f'{num_patients} Patients', fontsize=24)
        i = 0
        for metric in ['time', 'rmse']:
            ax[i].plot(range(1, len(eval_dict[metric]) + 1), eval_dict[metric])
            ax[i].set_title(f'Mean {metric}: {means[f'mean {metric}']:0.4f}',
                            fontsize=24)
            ax[i].set_xlabel('Iteration', fontsize=24)
            ax[i].set_ylabel(metric, fontsize=24)
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=20)
            ax[i].tick_params(axis='y', labelsize=20)
            i += 1
        fig.tight_layout()
        rmse_figure_name = f'{num_patients}_patients_rmse_and_time.png'
        fig.savefig(os.path.join(stats_save_directory, rmse_figure_name), dpi=120)

    # Make summary plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=120)
    fig.suptitle('Mean Evaluation scores', fontsize=24)
    i = 0
    for eval_score in eval_results.columns:
        ax[i].plot(eval_results[eval_score])
        ax[i].set_title(f'{eval_score.capitalize()}: {eval_results[eval_score].mean():0.4f}',
                        fontsize=24)
        ax[i].set_xlabel('Patients Datasets', fontsize=24)
        ax[i].set_ylabel(eval_score, fontsize=24)
        ax[i].grid(True)
        ax[i].tick_params(axis='x', labelsize=20)
        ax[i].tick_params(axis='y', labelsize=20)
        i += 1
    fig.tight_layout()
    evaluation_figure_name = 'Evaluation_metrics_summary.png'
    fig.savefig(os.path.join(stats_save_directory, evaluation_figure_name), dpi=120)

    print(eval_results)


if __name__ == '__main__':

    # Project folders used to store data and results
    PROJECT_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
    DATA_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, 'data')
    RESULTS_DIRECTORY = os.path.join(PROJECT_ROOT_DIRECTORY, 'results')
    STATS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'stats')

    # Argparse
    DESCRITPION = 'This module implements a medical recommendation system able to' \
            ' provide a suggested set of 5 therapies given a dataset and a ' \
            'patient in the dataset and the condition for which the '\
            'therapies must be suggested.\n' \
            'It is also able to generate a randomly filled dataset given ' \
            'the conditions and therapies pools or website from which to ' \
            'crawl the informations'
    parser = argparse.ArgumentParser(description=DESCRITPION)

    DATASET_HELP = 'Dataset used both to save and retrieve values.\n' \
                   'The save operation overwites the file.'
    parser.add_argument('-d', metavar='dataset', nargs='+',
                        default=None,
                        help=DATASET_HELP)
    parser.add_argument('--random-fill', metavar='num', default=None,
                        help='Fills the dataset specified by \'-d\' with '
                             'random \'num\' entries.')
    parser.add_argument('--recommend', metavar='patient_id', nargs='+',
                        help='Calls the recommendation system and provides a'
                             'set of recommended therapies for the condition '
                             'specified by -c. It takes as arguments the '
                             'ID of one or more patients', default=[])
    parser.add_argument('-c', metavar='conditions', nargs='+',
                        help='Use this flag to tell the IDs of the coditions '
                             'for which a recommendation is required.\n'
                             'This flag has to be used in conjunction with '
                             'the --recommend one. 1 condition for 1 patient '
                             'id.')
    parser.add_argument('--sample', metavar='n_samples', default=0,
                        help='Use this command to randomly sample from the '
                             'random filled dataset')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Flags whether or not to save the results')
    parser.add_argument('--evaluate', default=0, metavar='evaluation_steps',
                        help='Runs the evaluation of the recommendation'
                             'system over the dataset specified')
    parser.add_argument('--last_run', action='store_true', default=False,
                        help='Use this flag to print the results of the last '
                             'run if any')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='Use this flag to access the usage statistics of'
                             ' the recommendation system like rmse and time. '
                             'The available data will be read from the '
                             f'{STATS_DIRECTORY} directory')
    args = parser.parse_args()

    DEFAULT_DATASET_NAMEFILE = 'dataset.json'

    if args.random_fill is not None:
        fill_num = int(args.random_fill)
        random_dataset = Dataset()
        random_dataset.random_fill(os.path.join(DATA_DIRECTORY), fill_num)
        random_dataset.print()
        if args.save:
            if args.d is not None:
                DEFAULT_DATASET_NAMEFILE = os.path.basename(args.d[0])
            dataset_save_path = os.path.dirname(DEFAULT_DATASET_NAMEFILE)
            if len(dataset_save_path) == 0:
                dataset_save_path = os.path.join(DATA_DIRECTORY, 'temp')
            exists_or_create_directory(dataset_save_path)
            if int(args.sample) > 0:
                random_dataset.sample_not_cured(int(args.sample), saveToJson=True)
            random_dataset.save_in_json(dataset_save_path, DEFAULT_DATASET_NAMEFILE)

    if len(args.recommend) > 0:
        if args.c is None and os.sep not in args.recommend[0]:
            print('No condition provided.\n'
                  'Use -c followed by the condition id.\n'
                  'Or provide a .json or .txt file having the informations')
        else:
            exists_or_create_directory(RESULTS_DIRECTORY)
            if len(args.d) == 1:
                args.d = args.d[0]
            else:
                print('Too many dataset for a recommendation')

            th = medical_recommendation(args.d, args.recommend, args.c,
                                        args.save)
            if th is not None:
                for patient, recommendations in th.items():
                    fileName = RESULTS_DIRECTORY + os.sep + patient + '.json'
                    recommendations.to_json(fileName, indent=4)
                    print(f'Recommendation for Patient ID {patient}:')
                    print(recommendations)

    if args.last_run:
        last_results = {}
        exists_or_create_directory(RESULTS_DIRECTORY)
        for el in os.listdir(RESULTS_DIRECTORY):
            if '.json' in el:
                with open(RESULTS_DIRECTORY + os.sep + el, 'r', encoding="utf-8") as stream_in:
                    loaded_dataframe = pd.DataFrame(json.load(stream_in))
                    stream_in.close()
                el = el.replace('.json', '')
                last_results[el] = loaded_dataframe
                print(f'Recommendations for patient {el}:')
                print(loaded_dataframe, '\n')

    if args.stats:
        exists_or_create_directory(STATS_DIRECTORY)
        stats = {'run time': {}, 'rmse': {}}
        for el in os.listdir(STATS_DIRECTORY):
            if '.txt' in el:  # All stats saved in .txt files
                el_no_txt = el.replace('.txt', '')
                if 'time' in el:
                    stats['run time'][el_no_txt] = stat_interaction(STATS_DIRECTORY, el)
                if 'rmse' in el:
                    stats['rmse'][el_no_txt] = stat_interaction(STATS_DIRECTORY, el)

        # print('Run times:')
        for stat in stats:
            for el, item in stats[stat].items():
                kind = f'{el.split('_')[2]} {el.split('_')[3]}'
                iter_message = '-'*5 + f' {kind}:'
                print(stat.upper())
                print(iter_message, '-'*(40-len(iter_message)))
                for x in item:
                    print(x)
                out_message = f'Mean {stat}: {(sum(item)/len(item))}'
                print(out_message, '-'*(40-len(out_message)), '\n')

    if int(args.evaluate) > 0:
        exists_or_create_directory(STATS_DIRECTORY)
        evaluation(STATS_DIRECTORY, args.d, args.evaluate, random_fill_save_path=DATA_DIRECTORY)
