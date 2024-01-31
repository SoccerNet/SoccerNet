import json
import numpy as np

def evaluate(ground_truth_file, predictions_file):

    EVENT_DICTIONARY_action_class = {"Tackling":0,"Standing tackling":1,"High leg":2,"Holding":3,"Pushing":4,
                        "Elbowing":5, "Challenge":6, "Dive":7, "Dont know":8}


    labels = json.load(open(ground_truth_file))
    predictions = json.load(open(predictions_file))
        
    
    distribution_action_groundtruth = np.zeros((1, 8))
    distribution_action_prediction = np.zeros((1, 8))
    distribution_offence_severity_groundtruth = np.zeros((1, 4))
    distribution_offence_severity_prediction = np.zeros((1, 4))


    counter = 0

    for actions in labels["Actions"]:
        action_class = labels['Actions'][actions]['Action class']
        offence_class = labels['Actions'][actions]['Offence']
        severity_class = labels['Actions'][actions]['Severity']

        offence_severity_class_groundtruth = 0
        action_class_groundtruth = 0


        if action_class == '' or action_class == 'Dont know':
            continue

        if (offence_class == '' or offence_class == 'Between') and action_class != 'Dive':
            continue

        if (severity_class == '' or severity_class == '2.0' or severity_class == '4.0') and action_class != 'Dive' and offence_class != 'No offence' and offence_class != 'No Offence':
            continue

        if offence_class == '' or offence_class == 'Between':
            offence_class = 'Offence'

        if severity_class == '' or severity_class == '2.0' or severity_class == '4.0':
            severity_class = '1.0'

        if offence_class == 'No Offence' or offence_class == 'No offence':
           offence_class = 'No offence'

        if offence_class == 'No Offence' or offence_class == 'No offence':
            offence_severity_class_groundtruth = 0
        elif offence_class == 'Offence' and severity_class == '1.0':
            offence_severity_class_groundtruth = 1
        elif offence_class == 'Offence' and severity_class == '3.0':
            offence_severity_class_groundtruth = 2
        elif offence_class == 'Offence' and severity_class == '5.0':
            offence_severity_class_groundtruth = 3
        else:
            continue      

        counter += 1
        action_class_groundtruth = EVENT_DICTIONARY_action_class[action_class]

        if action_class == 'Dive':
            severity_class = '1.0'

        if offence_class == 'No offence':
            severity_class = ''
        
        distribution_action_groundtruth[0, action_class_groundtruth] += 1
        distribution_offence_severity_groundtruth[0, offence_severity_class_groundtruth] += 1
        if actions in predictions['Actions']:
            if labels['Actions'][actions]['Offence'] == predictions['Actions'][actions]['Offence'] and severity_class == predictions['Actions'][actions]['Severity']:
                distribution_offence_severity_prediction[0, offence_severity_class_groundtruth] += 1

            if labels['Actions'][actions]['Action class'] == predictions['Actions'][actions]['Action class']:
                distribution_action_prediction[0, action_class_groundtruth] += 1

        else:
            print("You did not predict the action: ", actions)

    accuracy_offence_severity = sum(distribution_offence_severity_prediction[0]) / sum(distribution_offence_severity_groundtruth[0])
    accuracy_action = sum(distribution_action_prediction[0]) / sum(distribution_action_groundtruth[0])
    
    print(type(distribution_offence_severity_prediction))

    balanced_accuracy_offence_severity = np.mean(distribution_offence_severity_prediction / distribution_offence_severity_groundtruth)
    balanced_accuracy_action = np.mean(distribution_action_prediction / distribution_action_groundtruth)

    leaderboard_value = balanced_accuracy_offence_severity * 0.5 + balanced_accuracy_action * 0.5

    results = {
        "accuracy_offence_severity": accuracy_offence_severity*100,
        "accuracy_action": accuracy_action*100,
        "balanced_accuracy_offence_severity": balanced_accuracy_offence_severity*100,
        "balanced_accuracy_action": balanced_accuracy_action*100,
        "leaderboard_value": leaderboard_value*100
    }

    return results
