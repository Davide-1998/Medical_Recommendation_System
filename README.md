# Python Medical Recommendation System

This project serves as an understanding example of the application of Hybrid Recommendations systems in the
medical field.  
The purpose is to investigate wether it is possible, given a dataset of medical histories and therapies, to
apply recommendation systems algorithms to suggest the next best therapy to any new patient under analysis.

## Requirements

This code has been written majorly with python 3.x, thus any machine having a pyhton interpreter can run it.  
Anyway, being developed on a Linux machine some requirements for the OS are necessary.

|   OS    |        Packages        |
| :-----: | :--------------------: |
|  Linux  | wget </br> python v3.x |
| Windows |      python v3.x       |

Python environment code requirements have been provided in the `requirements.txt` file in the root folder.  
It is warmly suggested to use a python environment to avoid conflicting packages with your underlying python
distribution.  
Here the commands to achieve a standalone python environment for this project; remember to not version the `.venv`
fodler.

```bash
# Create a python environment named .venv
python3 -m venv .venv

# Activate the python environment
## Linux
source ./venv/bin/activate
## Windows
.\venv\Scripts\activate.bat

# Install requirements
pip3 install -r requirements.txt
```

## Code Usage

### Easy Run

A script performing the recommendation and evaluation tasks has been
provided in the `src` folder under the name `easy_test.sh`.  
It is possible to run it from a linux machine by issuing the commands:

```bash
/bin/bash ./src/easy_test.sh

# Or in alternative
chmod +x easy_test.sh
./easy_test.sh
```

More detailed script commands are explained in the following sections.

### Provide recommendation

To make the code provide the recommendations for one or more patients it
is enough to run the command:

```bash
`python3 Medical_recommendations.py --recommend patient_id_1 patient_id_2 .. -c condition_id`
```

### Using a File as Input for Recommendations

The `patient_id` field may also be a '.txt' which has two columns:
one for the patients identifier and the other for the patient
condition for which to have a recommendation.  
As shown in `./data/datasetB_cases.txt` where the file content is:

| PatientID | Patient Condition |
| :-------: | :---------------: |
|     6     |       pc32        |
|   51345   |     pc277636      |
|    ...    |        ...        |

An usage example is:

```bash
python3 Medical_recommendations.py --recommend path/to/id_cond_couples.txt
```

### Dataset Creation

Use the `./src/extractor.py` script with the web urls to conditions and
therapies.  
It has already been setted to the one used for developing the system.  
Be careful that is an "hard-coded solution", so further adjustments may
be done on the source code.
In `./data/temp` there are already some downloaded .html files ready to
be processed to provide the dataset.

### Using a Different Dataset

These commands will generate the therapies recommended for each patient
for the specified condition.
The default dataset used is located in the 'data' folder.  
Any other dataset can be used if it is addressed by the flag `-d`.  
As an example:

```bash
python3 Medical_recommendations -d path/to/dataset.json --recommend patient_id -c pc_id
```

### Save Results

The flag `--s` must be passed to save the results.

### Random dataset generation

In order to make the code generate a randomly filled dataset with N
entries, it is sufficient to run:

```bash
python3 Medical_recommendations.py --random-fill N --s
```

The code syntax means:

- `N`: Is the number of patients to generate.
- `--s`: Flag to save the randomly generated dataset into the `./data`
  folder.  
   If any other location is desired the flag `-d` must be provided.

More detailed informations in how to use the code can be found by
issuing the command:

```bash
python3 Medical_recommendations.py -h
```

## Evaluation

In order to run the evaluation of the recommendation system it is enough
to call the script by passing the argument `--evaluate` followed by the
integer number of evaluation steps to perform.
In each evaluation step 10 iterations are performed over the input
dataset.
At the end of the procedure the mean RMSE and time will be printed in
output.

An example of the command is:

```bash
python3 Medical_recommendations.py --evaluate 1
```

If a specific dataset have to be used, it is possible to specify it
through the `-d` flag:

```bash
python3 Medical_recommendations.py -d path/to/my/dataset.json --evaluate 1
```

### Show results

The results of the recommendation task are shown by default at the end
of the run.
To show the last results provided after the code execution, it is
possible to run the following command:

```bash
python3 Medical_recommendations.py --last_run
```

### Show statistics of the program

The statistics of the program can be inspected through the flag
`--stats`.  
This will automatically read the data contained into the `./results/stats`
folder and return to the user the known performances of the program.  
An example of usage is:

```bash
python3 Medical_recommendations.py --stats
```

## Files structure

### src Folder

In the `src` folder there are many files, here it is a brief explanation
of their roles:

- `extractor.py`: It is the script responsible for the download and
  refinement of the data contained in the webpages from which the
  conditions and therapies have been taken.  
   It is an "hard-coded" solution that works only on systems in which
  the program `wget` is installed.  
   In case of linux systems it is possible to install it through the
  command:  
   `sudo apt install wget -y`
- `Medical_recommendations.py`: It is the file in which is implemented
  the recommendation system.

### data folder

In the data folder are contained the datasets, the test patients ids
file and a temp folder.

- `temp` folder: It contains the processed website downloaded by the
  functions contained in the 'extractor.py' module in the 'src' folder.
  The downloaded websites are saved in .html format.
- `100k_patients_dataset.json`: It is the dataset contatining 100
  thousand patients having random conditions for which trials with
  random therapies have been tried. This dataset is comprehensive of:

  - 322 Conditions processed from:  
    [https://www.nhsinform.scot/illnesses-and-conditions/a-to-z](https://www.nhsinform.scot/illnesses-and-conditions/a-to-z)
  - 204 Therapies processed from:  
    [https://en.wikipedia.org/wiki/List_of_therapies](https://en.wikipedia.org/wiki/List_of_therapies)
  - 100k Patients generated randomly

- `3_uncured_patients.json`: It is a .json file containing the ids of
  three patients, in the dataset, which suffers from some deseases.

### results folder

In the results folder are contained the output of the recommendation
system.  
Specifically, for each id given in input a .json file with the
suggestions is created in this folder.  
The subfolder `./results/Stats` is used to store several files
containing the statistics of the recommendation system: one for the
execution times and the other for the evaluation results dependant by
the number of patients contained in the dataset.

## Further Reading

A more extensive explanation of the provided work can be found in
[Agorithm_and_Tests.pdf](https://github.com/Davide-1998/Medical_Recommendation_System/blob/c10cc9167661eb0b84a8645b7db290e71af89f8a/doc/Algorithm_and_Tests.pdf).

## TODOs

- Add [urllib3](https://urllib3.readthedocs.io/en/stable/index.html) support to not require wget installation
