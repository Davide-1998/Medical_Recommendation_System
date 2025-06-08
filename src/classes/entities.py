'''
This module contains the class representing the entities that will be manipulated by the
recommendation system and that will represent the data available and how they can be used.
'''

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
