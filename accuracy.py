import os 

original_path = 'test_images'
radiologist_path = 'radiologist_selection'
results_path = 'test_results_test_two'
# results_path = 'test_results_test_two'
# results_path = 'test_results_test_three'


patients = os.listdir('test_images')

num = len(patients)
correct_num = 0

for i, file in enumerate(patients):
    radiologist_select = os.listdir(os.path.join(radiologist_path, patients[i].replace('-', '_')))
    test_results = os.listdir(os.path.join(results_path, patients[i]))

    if list(set(radiologist_select) & set(test_results)):
        correct_num += 1

acc = correct_num / num
print(f'%.4f' % acc)