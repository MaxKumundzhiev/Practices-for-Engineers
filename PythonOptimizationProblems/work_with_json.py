from random import choice
import json
import os

##############################################################################################################################################################################
        #IMPORTANT: json don't provide the support for UTF-8    
        #json.dump() --> this method is resposibke for disurialisation --> we would like to write data in json format
        # args: ({what we would like to write}, {where we would like to write}, {indent=value - padding, {ensure_ascii=True/Flase - whether check for utf-8 nor not}})

        #IMPORTANT: shorter version of reading existing json file:
        #json.load(open('path_to_file_to_read'))

        #json.load() --> this method is resposibke for surialisation --> we would like to read a data in json format 
        # args: ({paht_to_file_to_read}) 
##############################################################################################################################################################################        


def person_generator():
    'generate person name and number'
    'output: dictionary {"name": name, "number": number}'
    characters = ['a', 'b', 'c', 'd', 'e']
    numbers = ['1', '2', '3', '4', '5', '6', '7']

    name = ''
    number = ''

    while len(name) != 5:
        name += choice(characters)
    while len(number) != 7:
        number +=  choice(numbers)    

    person = {
        'name' : name,
        'number': number
    }
    return person

def transform_to_json():
    persons = []
    for i in range(5):
        persons.append(person_generator())
    with open('generated_persons.json', 'w') as file:
        json.dump(persons, file, indent=2, ensure_ascii=False)
    return persons   
    
def execute(persons_dict):
    try:
        #Check if we have our json file
        data = json.load(open('generated_persons.json')) 
    except:
        data = []

    data.append(persons_dict)
    with open('generated_persons.json', 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)        

def main():
    execute(transform_to_json())

#Entry point of script
if __name__ == '__main__':
    main() 