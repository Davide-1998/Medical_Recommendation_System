'''
Main script for evaluating the medical recommendation system
'''

import argparse
import json
from math import sqrt
import os
from random import choice
from time import time

import pandas as pd
from sklearn.metrics import mean_squared_error as RMSE
import matplotlib.pyplot as plt

from classes.dataset import Dataset


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
    for patient_id in utility_matrix.index:
        if patient_id != query_patient:
            patient_dist = utility_matrix.loc[patient_id].copy().fillna(0)
            square_distance = sum((patient_dist - query_dist)**2)

            # Cosine sim:
            # cos_sim = cosine_similarity([query_dist, patient_dist])[0, 1]

            similarities.loc[patient_id] = sqrt(square_distance)

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
        for patient_id in utility_matrix.index:
            if patient_id != query_patient:
                utility_patient = utility_matrix.loc[patient_id][th_filter].copy()
                retained = utility_patient[utility_patient.between(lb, ub)]
                same_biology.loc[patient_id] = retained
        # same_biology = same_biology[same_biology[:] == biology_vec[:]].dropna()
        same_biology = same_biology.dropna()
        print(f'Same biology matrix has size: {same_biology.shape}')

        intersection = nearest_neighbours.filter(same_biology.index,
                                                 axis=0).copy()
        # Increase (relatively) the similarity for those patients that has a
        # Similar biology
        for patient_id in intersection.index:
            nearest_neighbours.loc[patient_id]['euclidean'] -= 1
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
                         f'run_time_{len(reference_dataset.patients)}_patients.txt',
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
            key = namefile.replace('.json', '') + f'_{len(temporary_dataset.patients)}'

            loaded_datasets[key] = temporary_dataset
            eval_results_dict[key] = {'time': [], 'rmse': []}

    # Start gathering evaluation data
    for dataset_name, dataset in loaded_datasets.items():
        for _ in range(int(iterations)):
            num_patients = len(dataset.patients)
            condition_pool = [condition.id for condition in dataset.conditions]

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
                random_dataset.sample_not_cured(int(args.sample), save_to_json=True)
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
