import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

def preprocess_csv(metrics_file, exclude_datasets=['FPHA']):
    df = pd.read_csv(metrics_file)
    # Filter out excluded datasets
    df_filtered = df[~df['Dataset'].isin(exclude_datasets)]
    return df_filtered

def get_visualization_sequence(metrics_file, top_n=4):
    df = preprocess_csv(metrics_file)
    # group by dataset, for each dataset, sort by MPJPE and keep the top_n candidates, and output Dataset Trajectory MPJPE
    df_grouped = df.groupby('Dataset').apply(lambda x: x.sort_values(by='MPJPE', ascending=True).head(top_n).sort_values(by='Trajectory', ascending=True))
    output_file = metrics_file.replace('metrics.csv', 'seq.txt')
    with open(output_file, 'w') as f:
        for i, row in df_grouped.iterrows():
            f.write(f'{row["Dataset"]}, {row["Trajectory"]}, {row["MPJPE"]}\n')


def calculate_average_metrics_for_single_file(
    metrics_file, 
    select_columns=['MPJPE', 'MPJPE-PA', 'MPVE', 'MPVE-PA', 'MWTE', 'MRE'], 
    precision=3, 
    exclude_n=4,
    log=False
):
    """
    Compute metrics excluding specified datasets and N largest MPJPE values per dataset
    Args:
        df: pandas DataFrame with metrics
        select_columns: list of metric columns to compute
        precision: decimal precision for rounding
        exclude_n: number of largest MPJPE values to exclude per dataset
    """
    df_filtered = preprocess_csv(metrics_file)
    
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
    
    if log:
        print(f'-' * 20 + f'{metrics_file}' + '-' * 20)
        print(dataset_metrics)
        print("\nOverall averages:")
        for metric, value in total_metrics.items():
            print(f"{metric}: {value:.{precision}f}")
        print('\n')

    return dataset_metrics, total_metrics


'''
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/eval_metrics.csv',4)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/test_metrics.csv',10)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/T_l1_6/eval_metrics.csv',4)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/T_l1_6/test_metrics.csv',10)
breakpoint()
'''

if __name__ == "__main__":
    base_dir = '/mnt/vepfs/zbh/MHMP/logs'
    data_mapping = {
        # 'LatentAct': 'baseline-latentact',
        # r'\quad– no concat map': 'baseline-latentact_noconcat',
        # 'LatentAct-Diff': 'baseline-latentact_diff',
        # r'\quad– no concat map': 'baseline-latentact_diff_noconcat',
        r'\algoabbr-T': 'T_l1_6',
        r'\algoabbr-ID': 'TI_l1_6',
        r'\algoabbr-TI': 'TID_l1_6_DA',
        r'\textbf{\algo(ours)}': 'TID_l1_6_UD_norealdep',
        r'\quad– no depth supervision': 'TID_l1_6_UD_rel',
        r'\quad– depthanythingv2': 'TID_l1_6_UD',
        r'\quad– relative depth': 'ID_l1_6_UD',
    }
    '''
    name_mapping = {
        r'\algoabbr-T': 'MEgo-T',
        r'\algoabbr-ID': 'MEgo-ID',
        r'\algoabbr-TI': 'MEgo-TI',
        r'\textbf{\algo(ours)}': 'MEgoHand',
        r'\quad– no depth supervision': '– no depth supervision',
        r'\quad– depthanythingv2': '– depthanythingv2',
        r'\quad– relative depth': '– relative depth',
    }
    '''
    format_inv_mapping = {
        'MEgo_T': r'\algoabbr-T',
        'MEgo_ID': r'\algoabbr-ID',
        'MEgo_TI': r'\algoabbr-TI',
        'MEgoHand': r'\textbf{\algo(ours)}',
        'MEgoHand_nodepth': r'\quad– no depth supervision',
        'MEgoHand_depthanythingv2': r'\quad– depthanythingv2',
        'MEgoHand_rel': r'\quad– relative depth',
    }
    color_mapping = {
        'baseline_color': 'lightred',
        'TID_modality_color': 'lightblue',
        'TID_variant_color': 'lightgreen',
    }
    
    write_latex_line = lambda row, algo_name: f'& {algo_name} & {row["MPJPE"]:.3f} & {row["MPJPE-PA"]:.3f} & {row["MPVE"]:.3f} & {row["MPVE-PA"]:.3f} & {row["MWTE"]:.3f} & {row["MRE"]:.3f} \\\\'
    write_color_line = lambda color: f'\\rowcolor{{{color}}}'
    
    eval_metrics_list = [
        '/mnt/vepfs/zbh/MHMP/logs/T_l1_6/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/ID_l1_6_UD/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TI_l1_6/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_DA/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_rel/eval_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/eval_metrics.csv'
    ]
    metrics_row = dict()
    for algo_name in data_mapping:
        csv_file = os.path.join(base_dir, data_mapping[algo_name], 'test_metrics.csv')
        dataset_metrics, total_metrics = calculate_average_metrics_for_single_file(csv_file, exclude_n=0)
        metrics_row[algo_name] = total_metrics
    
    latex_line_dict = {k: write_latex_line(metrics_row[format_inv_mapping[k]], format_inv_mapping[k]) for k in format_inv_mapping}
    color_line_dict = {k: write_color_line(color_mapping[k]) for k in color_mapping}
    eval_latex_line = r'''
% ===== Baselines =====
{baseline_color}
{LatentAct}
{baseline_color}
{LatentAct_noconcat}
{baseline_color}
{LatentAct_diff}
{baseline_color}
{LatentAct_diff_noconcat}

\addlinespace[0.2em]

% ===== T, ID, TI =====
{TID_modality_color}
{MEgo_T}
{TID_modality_color}
{MEgo_ID}
{TID_modality_color}
{MEgo_TI}

\addlinespace[0.2em]

% ===== TID variants =====

{TID_variant_color}
{MEgoHand}
{TID_variant_color}
{MEgoHand_nodepth}
{TID_variant_color}
{MEgoHand_depthanythingv2}
{TID_variant_color}
{MEgoHand_rel}
    '''.format(**latex_line_dict, **color_line_dict)
    print(eval_latex_line)
    breakpoint()
    
    ########################################################################################3
    
    
    test_metrics_list = [
        '/mnt/vepfs/zbh/MHMP/logs/T_l1_6/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/ID_l1_6_UD/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TI_l1_6/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_DA/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_rel/test_metrics.csv',
        '/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/test_metrics.csv'
    ]
    

        
    for metrics_file in test_metrics_list:
        calculate_average_metrics_for_single_file(metrics_file)