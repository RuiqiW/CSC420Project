import pandas as pd
import sys
import os

def prep_data_index_file(file_name, train_file, valid_file, test_file): 
    """ Load the original index csv file from Google's landmark dataset v2
        and samples from the most famous 1000 classes, and make a 7:2:1
        split to train, valid, test data
    """
    index = pd.read_csv(file_name)
    landmark = index.groupby("landmark_id").count()
    
    most_famous = landmark.sort_values("id", ascending=False).iloc[: 1000]
    most_famous = most_famous.reset_index()['landmark_id']
    selected = pd.DataFrame()
    for lid in most_famous:
        find = index.loc[index.landmark_id == lid]
        selected = selected.append(find.sample(min(300, len(find))))

    selected = selected[['id', 'landmark_id']]
    land_map = selected.groupby("landmark_id").first().reset_index()[['landmark_id']]
    land_map['class_id'] = land_map.index
    selected = selected.merge(land_map)
    
    train = selected.sample(frac=0.7)
    valid_test = selected.drop(train.index)
    valid = valid_test.sample(frac=0.67)
    test = valid_test.drop(valid.index)
    
    
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    test.to_csv(test_file, index=False)
    

def adding_noise_to_test(test_file, noise_folder):
    """ Add the non-landmark images into the test dataset
    """
    test = pd.read_csv(test_file)
    noise = {
        'id': [],
        'landmark_id': [],
        'class_id': []
    }
    for filename in os.listdir(noise_folder):
        noise['id'].append(filename.replace(".jpg", ""))
        noise['landmark_id'].append(1000)
        noise['class_id'].append(1000)
    test = test.append(pd.DataFrame(noise))
    test.to_csv(test_file, index=False)
        
if __name__ == "__main__":
    file_name, train_file, valid_file, test_file = sys.argv[1:]
    prep_data_index_file(file_name, train_file, valid_file, test_file)
    
    
