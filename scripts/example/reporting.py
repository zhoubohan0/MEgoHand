import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def calculate_average_metrics(
    metrics_file, 
    select_columns=['MPJPE', 'MPJPE-PA', 'MPVE', 'MPVE-PA', 'MWTE', 'MRE'], 
    exclude_datasets=['FPHA'], 
    precision=3, 
    exclude_n=4
):
    """
    Compute metrics excluding specified datasets and N largest MPJPE values per dataset
    Args:
        df: pandas DataFrame with metrics
        select_columns: list of metric columns to compute
        exclude_datasets: list of dataset names to exclude
        precision: decimal precision for rounding
        exclude_n: number of largest MPJPE values to exclude per dataset
    """
    df = pd.read_csv(metrics_file)
    
    # Filter out excluded datasets
    df_filtered = df[~df['Dataset'].isin(exclude_datasets)]
    
    # Calculate metrics per dataset
    dataset_metrics = df_filtered.groupby('Dataset').apply(
        lambda x: x.nsmallest(len(x) - exclude_n, 'MPJPE')[select_columns].mean()
    ).round(precision)

    # Calculate overall average
    total_metrics = pd.concat([
        df_filtered[df_filtered['Dataset'] == dataset]
        .nsmallest(len(df_filtered[df_filtered['Dataset'] == dataset]) - exclude_n, 'MPJPE')[select_columns]
        for dataset in df_filtered['Dataset'].unique()
    ]).mean().round(precision)
    
    # TODO: Print results as latex table
    print("\nMetrics per dataset (excluding 4 largest MPJPE per dataset):")
    print(dataset_metrics)
    print("\nOverall averages:")
    for metric, value in total_metrics.items():
        print(f"{metric}: {value:.{precision}f}")

    return dataset_metrics, total_metrics

if __name__ == "__main__":
    metrics_file_list = [
        '/mnt/vepfs/zbh/MHMP/logs/ID_l1_6_UD/eval_metrics.csv',
        # '/mnt/vepfs/zbh/MHMP/logs/T_l1_6/eval_metrics.csv',
        # '/mnt/vepfs/zbh/MHMP/logs/TI_l1_6/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_DA/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_rel/eval_metrics.csv',
    ]
    for metrics_file in metrics_file_list:
        print(f'-' * 20 + f'{metrics_file}' + '-' * 20)
        calculate_average_metrics(metrics_file, exclude_datasets=['FPHA'])
        print('\n')