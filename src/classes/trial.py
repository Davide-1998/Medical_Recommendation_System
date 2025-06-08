'''
Module defining the Trial Class
'''

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

        self.__dict__.update(loaded_dict)

    def print(self, indent=False):
        '''
        Prints the class attributes data.

        indent: bool
            Adds a tab spacing in the formatting of the string.
        '''

        for key, value in self.__dict__.items():
            print(f"{'\t' if indent else ''}{key:<10} : {value:<10}")
