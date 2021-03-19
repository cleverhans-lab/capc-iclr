import os
files_dist = 'files'
input_data = 'input_data.npy'
input_labels = 'input_labels.npy'
predict_labels = 'predict_labels'
out_client_name = os.path.join(files_dist, 'logits')
out_server_name = os.path.join(files_dist, 'noise')
out_final_name = os.path.join(files_dist, 'output')
label_final_name = os.path.join(files_dist, 'final_label')
inference_times_name = os.path.join(files_dist, 'inference_times')
argmax_times_name = os.path.join(files_dist, 'argmax_times')
client_csp_times_name = os.path.join(files_dist, 'client_csp_times')
inference_no_network_times_name = os.path.join(files_dist, 'inference_no_network_times')