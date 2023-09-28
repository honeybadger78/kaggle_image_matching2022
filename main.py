import time
from utils.io import load_yaml
from net.image_matcher import match_images
from net. keypoint_to_fundamental import SolutionHolder


def solution(samples, config, output):
    solution_holder = SolutionHolder()
    for i, row in enumerate(samples):
    
        sample_id, batch_id, image_1_id, image_2_id = row
        start_time = time.time()
  
        matching_key_points0, matching_key_points1 = match_images(config, sample_id, batch_id, image_1_id, image_2_id)
    
        solution_holder.add_solution(sample_id, matching_key_points0, matching_key_points1)
    
        end_time = time.time()
        print(f'Iter total: {end_time - start_time:.04f}s')
  
    solution_holder.dump(output)

if __name__ == "__main__":
    import csv
    
    config= load_yaml("/kaggle_matching/config/config.yaml")
    samples_data= config['Directory']['image'] + 'test.csv'
    with open(f'{samples_data}') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            samples_data += [row]
    solution(samples_data, config['Directory']['submissino_csv'], config['Directory']['output'])