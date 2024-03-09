import csv
import statistics
import glob

# Get a list of all CSV files in the results directory
files = glob.glob('./results/*.csv')

# Initialize dictionary to store metrics for each dataset
dataset_metrics = {}

# Read and process each CSV file
for file in files:
    with open(file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            dataset = row['Dataset']
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = {
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1 Score': [],
                    'AUC': [],
                    'Specificity': [],
                    'Sensitivity': []
                }

            accuracy = float(row['Accuracy'])
            precision = float(row['Precision'])
            recall = float(row['Recall'])
            f1_score = float(row['F1 Score'])
            auc = float(row['AUC'])
            specificity = float(row['Specificity'])
            sensitivity = float(row['Sensitivity'])

            dataset_metrics[dataset]['Accuracy'].append(accuracy)
            dataset_metrics[dataset]['Precision'].append(precision)
            dataset_metrics[dataset]['Recall'].append(recall)
            dataset_metrics[dataset]['F1 Score'].append(f1_score)
            dataset_metrics[dataset]['AUC'].append(auc)
            dataset_metrics[dataset]['Specificity'].append(specificity)
            dataset_metrics[dataset]['Sensitivity'].append(sensitivity)

# Calculate mean and variance for each dataset and metric
mean_variance = {}

for dataset, metrics in dataset_metrics.items():
    mean_variance[dataset] = {}
    for metric, values in metrics.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        mean_variance[dataset][metric] = f'{mean}({stdev})'

# Create a new CSV file with the mean and variance for each dataset and metric
headers = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity', 'Sensitivity']
new_rows = []

for dataset, metrics in mean_variance.items():
    new_row = [dataset]
    for metric, value in metrics.items():
        new_row.append(value)
    new_rows.append(new_row)

# Write the new CSV file
with open('./results/statistics.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    writer.writerows(new_rows)