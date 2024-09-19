import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plotSchedule(schedule_csv):
    
    df = pd.read_csv(schedule_csv)
    # schedule_csv = "_"+schedule_csv
    plt.plot(df['Cycle'], df['memory usage %'], marker='o', markersize=1, linestyle='-', color='b')
    plt.title('Memory Usage % Across Cycles')
    plt.xlabel('Cycle')
    plt.ylabel('Memory Usage %')
    plt.grid(True)
    plt.savefig(f'{schedule_csv}.memory_usage.png')
    
    # plot decode effective bsz
    plt.figure()
    plt.plot(df['Cycle'], df['decode effective bsz'], marker='o', linestyle='-', color='g')
    plt.title('Decode Effective Batch Size Across Cycles')
    plt.xlabel('Cycle')
    plt.ylabel('Decode Effective Batch Size')
    plt.grid(True)
    plt.savefig(f'{schedule_csv}.decode_effective_bsz.png')
    
    # plot prefill effective bsz
    plt.figure()
    plt.plot(df['Cycle'], df['prefill effective bsz'], marker='o', linestyle='-', color='r')
    plt.title('Prefill Effective Batch Size Across Cycles')
    plt.xlabel('Cycle')
    plt.ylabel('Prefill Effective Batch Size')
    plt.grid(True)
    plt.savefig(f'{schedule_csv}.prefill_effective_bsz.png')
    
    # plot both decode and prefill effective bsz in one plot
    plt.figure()
    plt.plot(df['Cycle'], df['decode effective bsz'], marker='o', linestyle='-', color='g', label='Decode Effective Batch Size')
    plt.plot(df['Cycle'], df['prefill effective bsz'], marker='o', linestyle='-', color='r', label='Prefill Effective Batch Size')
    plt.title('Decode and Prefill Effective Batch Size Across Cycles')
    plt.xlabel('Cycle')
    plt.ylabel('Effective Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{schedule_csv}.effective_bsz.png')
    

if __name__ == '__main__':
    # plotSchedule('512_large_kv_cache.schedule.csv')
    # plotSchedule('512.schedule.csv')
    # plotSchedule('740.schedule.csv')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--schedule_csv", type=str, help="path to schedule csv file")
    args = arg_parser.parse_args()
    plotSchedule(args.schedule_csv)