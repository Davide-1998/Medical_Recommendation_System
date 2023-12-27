# This script works only on machine having wget available through shell command

import os
import subprocess


def download_webpage(url, targetFile='webpage.html', reDownload=False):
    '''
    This method is used to download a webpage and store it in a
    specific location on the device.

    url: str
        Is the webpage's url which will be downloaded.
    targetFile: str
        Is the file path in which the webpage will be saved.
        If no system separator is found on it, the local one will
        be used.
    '''

    if os.sep not in targetFile:
        targetFile += os.getcwd() + os.sep

    if not os.path.isdir(os.path.dirname(targetFile)):
        os.makedirs(os.path.dirname(targetFile), 0o777)

    if not os.path.isfile(targetFile):
        subprocess.call(['wget', '-k', '-O', targetFile, '%s' % url])
    elif reDownload:
        subprocess.call(['wget', '-k', '-O', targetFile, '%s' % url])


def extract_condition_webpage(url, targetFile='Condition_website.html'):
    '''
    This method is used to download and process the conditions coming
    from an input url.

    url: str
        Is the webpage url which will be downloaded
    targetFile: str
        Is the file name in which the webpage will be saved and from
        which data will be loaded.

    Returns
    -------
    list of lists
        Is the list of conditions name that have been found in the
        webpage followed by the type attributed to them
    '''

    download_webpage(url, targetFile)
    webpage = open(targetFile, 'r')

    conditions = []

    # for line in webpage:
    #     if url in line:
    #         conditions.append(line)

    nextSave = False
    for line in webpage:
        if nextSave:
            conditions.append(line)
            nextSave = False
        if url in line:
            nextSave = True

    webpage.close()

    for el in conditions:
        line = el.replace('\t', '')
        line = line.replace(':', '')
        line = line.replace('&#39;', "'")
        if '(' in line:
            idx_start = line.index('(')
            idx_stop = idx_start
            while line[idx_stop] != ')':
                idx_stop += 1
            line = line[:idx_start-1] + line[idx_stop+1:]

        '''
        line = el.split(' ')[1]
        line = line.replace('"', '')
        line = line.split('/')[-1]
        line = line.replace('-', ' ')
        if '(' in line:
            idx = line.index('(')
            line = line[:idx]
        '''
        conditions[conditions.index(el)] = line.strip()

    # cond_copy = conditions.copy()
    # singleWord = set()
    # doubleWord = set()
    # unexplained = set()

    types = set()
    for el in conditions:
        if len(el.split(' ')) == 1:
            types.add(el)
            # singleWord.add(el)
            # cond_copy.remove(el)

        elif len(el.split(' ')) == 2 or len(el.split(' ')) == 3:
            # doubleWord.add(el)
            single = el.split(' ')[-1]
            if single.casefold() in ['children', 'adults']:  # Flag list
                if len(el.split(' ')) == 2:
                    single = el.split(' ')[0].replace(':', '')
                elif len(el.split(' ')) == 3:
                    single = el.split(' ')[1].replace(':', '')
                    if single == 'in':
                        single = el.split(' ')[0]
                else:
                    single = el.split(' ')[-2].replace(':', '')
            types.add(single)
        # else:
        #     _type = el
        #     # singleWord.add(single)
        #     # cond_copy.remove(el)

        # conditions[conditions.index(el)] = [el, _type]

    condition_type = []
    for el in conditions:
        el = el.replace(':', '').split(' ')
        stop = False
        for item in types:
            if item in el and not stop:
                condition_type.append([' '.join(el[:]), item])
                stop = True
                break
        if not stop:
            el = ' '.join(el[:])
            condition_type.append([el, el])
    return condition_type


def extract_therapy_webpage(url, targetFile='Therapy_website.html'):
    '''
    This method is used to download and process the webpage used
    as source for therpaies names.

    url: str
        Is the webpage url which will be downloaded
    targetFile: str
        Is the file name in which the webpage will be saved and from
        which data will be loaded.

    Returns
    -------
    list of lists
        Is the list of conditions name that have been found in the
        webpage followed by the type of each therapy.
    '''

    download_webpage(url, targetFile)

    therapies = []
    webpage = open(targetFile, 'r')
    for line in webpage:
        # if 'therapy' in line:
        if 'li' in line:
            therapies.append(line)
    webpage.close()
    therapies = therapies[21:-35]  # Removes garbage

    therapies_type = []
    for el in therapies:
        temp = el.split('title=')[1]
        temp = temp.split('"')
        if '' in temp:
            temp.remove('')
        temp = temp[0]
        if '&' in temp:
            temp = temp.split('&')[0].replace('_', ' ')
        if '(' in temp:
            temp = temp.split('(')[0]
        temp = temp.strip()
        split_temp = temp.split(' ')

        # Type choice
        if len(split_temp) == 1:
            therapies_type.append([temp, temp])
        else:
            if 'therapy' in split_temp:
                idx = split_temp.index('therapy')
                rec_temp = ' '.join(split_temp[:idx])
                rec_temp.strip()
                therapies_type.append([temp, rec_temp])
            else:
                therapies_type.append([temp, split_temp[0].strip()])
    return therapies_type


if __name__ == '__main__':
    t_dir = '../data/temp'

    cond_url = 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z'
    condName = 'Condition_website.html'
    conditions = extract_condition_webpage(cond_url, t_dir + os.sep + condName)

    therapy_url = 'https://en.wikipedia.org/wiki/List_of_therapies'
    therName = 'Therapy_website.html'
    therapies = extract_therapy_webpage(therapy_url, t_dir + os.sep + therName)

    print('\nConditions list\n')

    for x in conditions:
        print(x)

    print('\nTherapies list\n')

    for x in therapies:
        print(x)
