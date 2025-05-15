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
    include_datasets=None,
    log=False
):
    """
    Compute metrics for specified datasets and N largest MPJPE values per dataset
    Args:
        metrics_file: path to metrics CSV file
        select_columns: list of metric columns to compute
        precision: decimal precision for rounding
        exclude_n: number of largest MPJPE values to exclude per dataset
        include_datasets: list of datasets to include (if None, includes all)
        log: whether to print results
    """
    df = pd.read_csv(metrics_file)
    df_filtered = df[df['Dataset'].isin(include_datasets)] if include_datasets is not None else df

    # Calculate overall average
    total_metrics = pd.concat([
        df_filtered[df_filtered['Dataset'] == dataset]
        .nsmallest(len(df_filtered[df_filtered['Dataset'] == dataset]) - exclude_n, 'MPJPE')[select_columns]
        for dataset in df_filtered['Dataset'].unique()
    ]).mean().round(precision)
    
    if log:
        print(f'-' * 20 + f'{metrics_file}' + '-' * 20)
        print(total_metrics)
        print("\nOverall averages:")
        for metric, value in total_metrics.items():
            print(f"{metric}: {value:.{precision}f}")
        print('\n')

    return total_metrics


'''
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/eval_metrics.csv',4)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/TID_l1_6_UD_norealdep/test_metrics.csv',10)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/T_l1_6/eval_metrics.csv',4)
get_visualization_sequence('/mnt/vepfs/zbh/MHMP/logs/T_l1_6/test_metrics.csv',10)
breakpoint()

get_visualization_sequence('/home/zbh/Desktop/zbh/hand-motion-pretraining/MHMP/logs/TID_l1_6_UD_norealdep/test_metrics.csv',20)
get_visualization_sequence('/home/zbh/Desktop/zbh/hand-motion-pretraining/MHMP/logs/T_l1_6/test_metrics.csv',20)
breakpoint()
'''


if __name__ == "__main__":
    data_mapping = {
        # 'LatentAct': 'baseline-latentact',
        # r'\quad– no concat map': 'baseline-latentact_noconcat',
        # 'LatentAct-Diff': 'baseline-latentact_diff',
        # r'\quad– no concat map': 'baseline-latentact_diff_noconcat',
        r'\algo-T': 'T_l1_6',
        r'\algo-I': 'TI_l1_6',
        r'\algo-ID': 'TI_l1_6',
        r'\algo-TI': 'TID_l1_6_DA',
        r'\algo~(ours)': 'TID_l1_6_UD_norealdep',
        r'\quad– no depth supervision': 'TID_l1_6_UD_rel',
        r'\quad– depthanythingv2': 'TID_l1_6_UD',
        r'\quad– relative depth': 'ID_l1_6_UD',
    }
    format_inv_mapping = {
        'MEgo_T': r'\algo-T',
        'MEgo_I': r'\algo-I',
        'MEgo_ID': r'\algo-ID',
        'MEgo_TI': r'\algo-TI',
        'MEgoHand': r'\algo~(ours)',
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
    write_latex_bold_line = lambda row, algo_name: f' & \\textbf{{{algo_name}}} & \\textbf{{{row["MPJPE"]:.3f}}} & \\textbf{{{row["MPJPE-PA"]:.3f}}} & \\textbf{{{row["MPVE"]:.3f}}} & \\textbf{{{row["MPVE-PA"]:.3f}}} & \\textbf{{{row["MWTE"]:.3f}}} & \\textbf{{{row["MRE"]:.3f}}} \\\\'
    write_color_line = lambda color: f'&\\rowcolor{{{color}}}'
    
    # base_dir = '/mnt/vepfs/zbh/MHMP/logs'
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    eval_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['H2O', 'HOI4D', 'HOT3D', 'TACO', 'OAKINK2']) for algo_name in data_mapping}
    arctic_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'test_metrics.csv'), include_datasets=['ARCTIC']) for algo_name in data_mapping}
    holo_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'test_metrics.csv'), include_datasets=['HOLO']) for algo_name in data_mapping}

    h2o_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['H2O'], exclude_n=0) for algo_name in data_mapping}
    hoi4d_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['HOI4D'], exclude_n=0) for algo_name in data_mapping}
    hot3d_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['HOT3D'], exclude_n=0) for algo_name in data_mapping}
    taco_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['TACO'], exclude_n=0) for algo_name in data_mapping}
    oakink2_metrics_row = {algo_name: calculate_average_metrics_for_single_file(os.path.join(base_dir, data_mapping[algo_name], f'eval_metrics.csv'), include_datasets=['OAKINK2'], exclude_n=0) for algo_name in data_mapping}
    for algo in [r'\algo~(ours)', r'\algo-TI']:
        print(','.join([str(h2o_metrics_row[algo]['MPJPE']), str(hoi4d_metrics_row[algo]['MPJPE']), str(hot3d_metrics_row[algo]['MPJPE']), str(oakink2_metrics_row[algo]['MPJPE']), str(taco_metrics_row[algo]['MPJPE']), str(arctic_metrics_row[algo]['MPJPE']), str(holo_metrics_row[algo]['MPJPE'])]))
    breakpoint()
    write_fn = lambda k: write_latex_line if k != 'MEgoHand' else write_latex_bold_line
    eval_latex_line = {k: write_fn(k)(eval_metrics_row[format_inv_mapping[k]], format_inv_mapping[k]) for k in format_inv_mapping}
    arctic_latex_line = {k: write_fn(k)(arctic_metrics_row[format_inv_mapping[k]], format_inv_mapping[k]) for k in format_inv_mapping}
    holo_latex_line = {k: write_fn(k)(holo_metrics_row[format_inv_mapping[k]], format_inv_mapping[k]) for k in format_inv_mapping}
    color_line_dict = {k: write_color_line(color_mapping[k]) for k in color_mapping}
    
    """
    print(r'''
{MEgoHand}
{MEgoHand_nodepth}
{MEgoHand_depthanythingv2}
{MEgoHand_rel}     
    '''.format(**eval_latex_line,**color_line_dict))
    
    print(r'''
{MEgoHand}
{MEgoHand_nodepth}
{MEgoHand_depthanythingv2}
{MEgoHand_rel}     
    '''.format(**arctic_latex_line,**color_line_dict))
    
    print(r'''
{MEgoHand}
{MEgoHand_nodepth}
{MEgoHand_depthanythingv2}
{MEgoHand_rel}     
    '''.format(**holo_latex_line,**color_line_dict))
    """
    
    
    
    print(r'''
{TID_variant_color}
{MEgo_T}          
{TID_variant_color}
{MEgo_I}
{TID_variant_color}
{MEgo_ID}
{TID_variant_color}
{MEgo_TI}
{TID_variant_color}
{MEgoHand}
    '''.format(**arctic_latex_line,**color_line_dict))
    
    print(r'''
{TID_variant_color}
{MEgo_T}          
{TID_variant_color}
{MEgo_I}
{TID_variant_color}
{MEgo_ID}
{TID_variant_color}
{MEgo_TI}
{TID_variant_color}
{MEgoHand}
    '''.format(**holo_latex_line,**color_line_dict))
    breakpoint()