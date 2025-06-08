'''
This module contains the definition of the Patient and PCondition classes
'''

import json
import os

from classes.entities import Entity
from classes.trial import Trial

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

    def __init__(self, _id=None, d_date='None', c_date='None', c_kind=None):
        '''
        _id: str
            Unique identifier of the specific condition.
            Condition - Patient relation.
        d_date: int
            Date of first diagnosed.
        c_date: int
            Date of cure.
        c_kind: Condition.id
            Id of condition. Condition - Condition pool relation.
        '''

        self.id = str(_id)
        self.diagnosed = str(d_date)
        self.cured = str(c_date)
        self.kind = str(c_kind)

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

        self.__dict__.update(loaded_dict)

    def print(self, indent=False):
        '''
        Prints the elements in the class.

        indent: bool
            Adds a tab spacing in the formatting of the string.
        '''

        for key, value in self.__dict__.items():
            print(f"{'\t' if indent else ''}{key:<10} : {value:<10}")


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

        for condition in self.conditions:
            if condition.kind == condition_id:
                return condition.id
        return ''

    def print(self, indent=True):
        '''
        Method used to print all the informations in the patient class.
        '''

        for key, value in self.__dict__.items():
            if not isinstance(value, list):
                print(f'{key:<10} : {value:<10}')
            else:
                print(f'{key} :', '-'*31)
                for t in value:
                    t.print(indent)
                    print('\t', '-'*15, '*', '-'*15)
