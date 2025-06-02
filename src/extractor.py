'''
This module provide functionalities to donwload the index.html of webpages and parsing it.
Its purpose is to being used for extracting a list of diseases and therapies to use inside
the recommendation system algorithm.
'''

import os
import re
import certifi
import urllib3


def get_webpage_data(url: str, cache_response = False, cache_filepath='webpage.html') -> str:
    '''
    This method is used to download a webpage and store it in a
    specific location on the device.

    url: str
        Is the webpage's url which will be downloaded.
    cache_response: bool
        Flag stating whether or not to dump the HTTP response in an .html file
    cache_filepath: str
        Is the file path in which the webpage will be saved.

    returns a string containing the decoded HTTP response
    '''

    if cache_response and not os.path.isfile(cache_filepath):
        # Build absolute file path
        os.path.abspath(cache_filepath)

        # Build folder hierarchy whenever it is not available
        if not os.path.isdir(os.path.dirname(cache_filepath)):
            os.makedirs(os.path.dirname(cache_filepath), 0o777)

    # Perform the GET request to the web page and save it on file
    request_headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) '
                        'Chrome/23.0.1271.64 Safari/537.11',
        'Connection': 'keep-alive',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3'
    }

    pool_manager = urllib3.PoolManager(headers=request_headers,
                                        cert_reqs='CERT_REQUIRED',
                                        ca_certs=certifi.where())
    response = pool_manager.request("GET", url)

    if response.status == 200:
        decoded_response = response.data.decode()
        if cache_response:
            with open(cache_filepath, "w", encoding="utf-8") as file_stream:
                file_stream.write(decoded_response)
                file_stream.close()
    else:
        raise urllib3.exceptions.HTTPError(f"Response status is {response.status}")

    # If everything okay return the response in string format
    return decoded_response.split('\n') # Need the split otherwise entire text is one line


def extract_condition_webpage(url, cache_http_response=False, target_file='Condition_website.html'):
    '''
    This method is used to download and process the conditions coming
    from an input url.

    url: str
        Is the webpage url which will be downloaded
    target_file: str
        Is the file name in which the webpage will be saved and from
        which data will be loaded.

    Returns
    -------
    list of lists
        Is the list of conditions name that have been found in the
        webpage followed by the type attributed to them
    '''

    conditions = []
    conditions_regex = r'\>([aA-zZ|_|-|\s]*)\<\/a\>'

    try:
        webpage = get_webpage_data(url, cache_http_response, target_file)
    except urllib3.exceptions.HTTPError as e:
        print(f"Exception: {e}")

    found_conditions_list = False

    for line in webpage:
        if 'az_list_indivisual' in line:
            found_conditions_list = True
        if found_conditions_list is True and '<li>' in line:
            regex_result = re.search(conditions_regex, line)
            if regex_result is not None:
                conditions.append(regex_result.groups()[0])
        if '</ul>' in line: # End of list reached
            found_conditions_list = False

    condition_type = []
    for item in conditions:
        splitted_condition = item.replace(':', '').split(' ')

        # Some diseases are presented as seen in spefic kinds of patients
        if 'in' in splitted_condition:
            splitted_condition = splitted_condition[:splitted_condition.index('in')]

        if len(splitted_condition) == 1:
            condition_type.append([item, item])
        else:
            condition_type.append([item, splitted_condition[-1]])

    print(condition_type)
    return condition_type


def extract_therapy_webpage(url, cache_http_response=False, target_file='Therapy_website.html'):
    '''
    This method is used to download and process the webpage used
    as source for therpaies names.

    url: str
        Is the webpage url which will be downloaded
    target_file: str
        Is the file name in which the webpage will be saved and from
        which data will be loaded.

    Returns
    -------
    list of lists
        Is the list of conditions name that have been found in the
        webpage followed by the type of each therapy.
    '''

    therapies = []
    therapies_regex = r'\>([aA-zZ|_|-|\s]*)\<\/a\>'

    try:
        webpage = get_webpage_data(url, cache_http_response, target_file)
    except urllib3.exceptions.HTTPError as e:
        print(f"Exception: {e}")

    found_therapies = False
    for line in webpage:
        if found_therapies is False and 'title=\"Traditional medicine\"' in line:
            found_therapies = True
        if found_therapies is True and 'li' in line:
            regex_result = re.search(therapies_regex, line)
            if regex_result is not None:
                therapies.append(regex_result.groups()[0])

        # Detect end of list and stop list making
        if found_therapies is True and '</ul>' in line:
            found_therapies = False

    therapies = therapies[1:]  # Removes useless column labels line

    therapies_type = []
    for item in therapies:
        split_item = item.split(' ')

        # Type choice
        if len(split_item) == 1:
            therapies_type.append([item, item])
        else:
            if 'therapy' in split_item:
                idx = split_item.index('therapy')
                rec_item = ' '.join(split_item[:idx])
                rec_item.strip()
                therapies_type.append([item, rec_item])
            else:
                therapies_type.append([item, split_item[0].strip()])
    return therapies_type


if __name__ == '__main__':
    # Tell whether or not to print found conditions and therapies
    PRINT_FOUND_DATA = False

    # Tell whether or not to dump http response to .html file
    SAVE_WEBPAGES_RESPONSE = False

    # Location where to save the http responses if the SAVE_WEBPAGES_RESPONSE is True
    TEMP_DATA_DIR = os.path.join(os.getcwd(), 'data', 'temp')

    # Query and download conditions
    CONDITIONS_URL = 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z'
    CONDITIONS_FILENAME = 'Condition_website.html'
    CONDITIONS_FILEPATH = TEMP_DATA_DIR + os.sep + CONDITIONS_FILENAME
    extracted_conditions = extract_condition_webpage(CONDITIONS_URL,
                                                     SAVE_WEBPAGES_RESPONSE,
                                                     CONDITIONS_FILEPATH)

    # Query and download therapies
    THERAPIES_URL = 'https://en.wikipedia.org/wiki/List_of_therapies'
    THERAPIES_FILENAME = 'Therapy_website.html'
    THERAPIES_FILEPATH = TEMP_DATA_DIR + os.sep + THERAPIES_FILENAME
    extracted_therapies = extract_therapy_webpage(THERAPIES_URL,
                                                  SAVE_WEBPAGES_RESPONSE,
                                                  THERAPIES_FILEPATH)

    # Print crawling results if requested, otherwise just a brief summary
    print(f"Extractor found: {len(extracted_conditions)} Conditions "
          f"and {len(extracted_therapies)} Therapies")

    if PRINT_FOUND_DATA:
        print('\n----- Conditions list -----\n')
        for cond in extracted_conditions:
            print(cond)

        print('\n----- Therapies list -----\n')
        for ther in extracted_therapies:
            print(ther)
