import pandas as pd
import os
from functools import lru_cache
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import copy
import argparse

# Define base paths
networkBasePath = "./trace/networkData"
gemmBasePath = "./trace"
gemvBasePath = "./trace"

# Read the original GEMM CSV file
gemmFileName = os.path.join(gemmBasePath, "save.csv")
df_original = pd.read_csv(gemmFileName)

# GEMM format:
# M, N, K, cta_m, cta_n, cta_k, warp_m, warp_n, warp_k, split, stage, A_layout, B_layout, D_layout, Runtime, GFLOPs


def getGEMMShape(batch_size):
    return [
        (batch_size, 1024, 8192),
        (batch_size, 7168, 8192),
        (batch_size, 8192, 3584),
        (batch_size, 3072, 8192),
    ]


def getGEMMOptimalSpeed(df):
    df.sort_values(by='Runtime', ascending=True, inplace=True)
    return {'time': df.iloc[0]['Runtime'], 'GFLOPs': df.iloc[0]['GFLOPs']}


def getGEMMOptimalConfig(df):
    df.sort_values(by='Runtime', ascending=True, inplace=True)
    return df.iloc[0]


def getBlockNums(gemmConfig):
    m = gemmConfig["M"]
    n = gemmConfig["N"]
    cta_m, cta_n = gemmConfig["cta_m"], gemmConfig["cta_n"]
    split_k = gemmConfig["split"]
    return m * n / cta_m / cta_n * split_k


def keepFastestPerBlockLimit(df, blockName, timeName):
    df = df.sort_values(by=[blockName, timeName]).groupby(blockName).head(1)
    df = df.sort_values(by=blockName, ascending=True).copy()
    df.reset_index(drop=True, inplace=True)
    return df


def getRestrainedGEMMSpeed(df):
    # **Correction:** Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Assign 'blocks' using .loc to be explicit
    df.loc[:, 'blocks'] = df.apply(getBlockNums, axis=1)
    
    # Filter and make a copy to avoid warnings
    df = df[df['blocks'] < 1080].copy()
    if df.empty:
        return None
    
    # Assign 'allSMTflops' using .loc
    df.loc[:, 'allSMTflops'] = df['GFLOPs'] * 108 / df['blocks']
    
    # Keep the fastest per block limit
    df = keepFastestPerBlockLimit(df, "blocks", "Runtime")
    
    # Assign 'tag' using .loc
    df.loc[:, 'tag'] = df.apply(genTag, axis=1)
    
    return df


def genTag(row):
    # Example tag format:
    # 128_256_32_64_64_32_1_2_ColumnMajor_RowMajor_ColumnMajor
    return f"{row['cta_m']}_{row['cta_n']}_{row['cta_k']}_{row['warp_m']}_{row['warp_n']}_{row['warp_k']}_{row['split']}_{row['stage']}_{row['A_layout']}_{row['B_layout']}_{row['D_layout']}"


@lru_cache(maxsize=1024)
def getAllGemmBlockSpeed(gemmShape, layoutRequirement=(None, None, None)):
    configs = []
    m, n, k = gemmShape
    all_df = pd.DataFrame()
    
    # **Correction:** Make a copy of the original DataFrame
    filterdf = df_original.copy()
    
    layoutList = ['A_layout', 'B_layout', 'D_layout']
    for i in range(3):
        if layoutRequirement[i] is not None:
            # **Correction:** Make a copy after each filtering operation
            filterdf = filterdf[filterdf[layoutList[i]] == layoutRequirement[i]].copy()
    
    for nsplit in range(0, 1):
        factor = 2 ** nsplit
        shape = (m, n // factor, k)
        df = filterdf[
            (filterdf['M'] == m) &
            (filterdf['N'] == n // factor) &
            (filterdf['K'] == k)
        ].copy()  # **Correction:** Make a copy after subsetting
        
        df = getRestrainedGEMMSpeed(df)
        if df is not None:
            all_df = pd.concat([all_df, df], ignore_index=True)
    
    # Get the optimal configuration without any factor
    df_optimal = filterdf[
        (filterdf['M'] == m) &
        (filterdf['N'] == n) &
        (filterdf['K'] == k)
    ].copy()  # **Correction:** Make a copy after subsetting
    if not df_optimal.empty:
        optimal = getGEMMOptimalConfig(df_optimal)
        optimal = optimal.copy()  # **Ensure optimal is a copy to avoid modifying original DataFrame
        
        # Assign 'blocks', 'tag', and 'allSMTflops' using .loc
        optimal['blocks'] = getBlockNums(optimal)
        optimal['tag'] = genTag(optimal)
        optimal['allSMTflops'] = optimal['GFLOPs'] * 108 / optimal['blocks']
        
        # Convert the Series to a DataFrame before concatenation
        optimal_df = pd.DataFrame([optimal])
        all_df = pd.concat([all_df, optimal_df], ignore_index=True)

    return all_df


def greedyOptimize(df):
    # **Correction:** Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    df.sort_values(by='blocks', ascending=True, inplace=True)
    
    # Initialize 'drop' column using .loc
    df.loc[:, 'drop'] = False
    initSize = df.shape[0]
    df.reset_index(drop=True, inplace=True)
    
    while True:
        previous_gflops = 0
        for index, row in df.iterrows():
            if row['GFLOPs'] < previous_gflops * 1.01:
                df.loc[index, 'drop'] = True  # **Correction:** Use .loc for assignment
            else:
                previous_gflops = row['GFLOPs']
        
        # **Correction:** Make a copy after filtering
        df = df[df['drop'] != True].copy()
        df.reset_index(drop=True, inplace=True)
        
        if df.shape[0] == initSize:
            break
        initSize = df.shape[0]
    
    return df.drop(columns=['drop'])


@lru_cache(maxsize=1024)
def getGemmProfile(gemmShape, layoutRequirement=(None, None, None)):
    # print(gemmShape, layoutRequirement)
    all_gemm_speed = getAllGemmBlockSpeed(gemmShape, layoutRequirement)
    # print(all_gemm_speed)
    optimized_df = greedyOptimize(all_gemm_speed)
    # print(optimized_df)
    optimized_df = optimized_df.rename(columns={'Runtime': 'time'})
    return optimized_df


global_search_space = range(512, 4096 + 128, 128)


def getNetworkExtrapolateData(file, size):
    df = pd.read_csv(file)
    x_df = df["size"].values
    y_df = df["avg_time"].values
    interpolation_function = interp1d(x_df, y_df, kind='linear', fill_value='extrapolate')
    y_values = interpolation_function(size)
    return y_values


def getNetProfile(sizeMB, filename):
    size = sizeMB * 1024 * 1024
    allgatherBase = os.path.join(networkBasePath, filename)
    files = [f for f in os.listdir(allgatherBase) if f.endswith('.csv')]
    combined_df = pd.DataFrame()
    
    for file in files:
        time = getNetworkExtrapolateData(os.path.join(allgatherBase, file), size)
        blocks = int(file.split('.')[0])
        d = {'size': size, 'blocks': blocks, 'time': time / 1000}
        combined_df = pd.concat([combined_df, pd.DataFrame(data=d, index=[0])], ignore_index=True)
    
    combined_df.sort_values(by='blocks', ascending=True, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


@lru_cache(maxsize=1024)
def getAllgatherProfile(sizeMB):
    return getNetProfile(sizeMB, "allgather")


@lru_cache(maxsize=1024)
def getAllgatherAsyncProfile(sizeMB):
    return getNetProfile(sizeMB, "allgatherAsync")


@lru_cache(maxsize=1024)
def getReduceScatterProfile(sizeMB):
    return getNetProfile(sizeMB, "reduceScatter")


@lru_cache(maxsize=1024)
def getReduceScatterAsyncProfile(sizeMB):
    return getNetProfile(sizeMB, "allgatherAsync")


@lru_cache(maxsize=1024)
def getAllReduceProfile(sizeMB):
    df1 = getNetProfile(sizeMB, "allgather")
    df2 = getNetProfile(sizeMB, "reduceScatter")
    
    def find_min_time(df1, row):
        filtered_df = df1[df1['blocks'] <= row['blocks']]
        if not filtered_df.empty:
            return filtered_df['time'].min()
        else:
            return None
    
    df2['time2'] = df2.apply(lambda row: find_min_time(df1, row), axis=1)
    df2['time_all'] = df2['time'] + df2['time2']
    df2.drop(columns=['time2', "time"], inplace=True)
    df2.rename(columns={'time_all': 'time'}, inplace=True)
    return df2


def getSizeofComm(batch_size):
    return 2 * batch_size * 8192 / 1024 / 1024


def getSizeofCommShape(shape):
    return 2 * shape[0] * shape[1] / 1024 / 1024


# Read the GEMV CSV file
df_gemv = pd.read_csv(os.path.join(gemvBasePath, 'gemv.csv'))


@lru_cache(maxsize=1024)
def getGemvTime(batch_size, seq_len, block_num):
    df_sorted = df_gemv.sort_values(by=['batch_size', 'seqlen', 'blocks']).copy()
    batch_sizes = df_sorted['batch_size'].unique()
    seqlens = df_sorted['seqlen'].unique()
    blocks = df_sorted['blocks'].unique()
    B, S, K = np.meshgrid(batch_sizes, seqlens, blocks, indexing='ij')
    times = df_sorted['time'].values.reshape(len(batch_sizes), len(seqlens), len(blocks))
    interpolator = RegularGridInterpolator(
        (batch_sizes, seqlens, blocks),
        times,
        method='linear',
        bounds_error=False,  # Allows extrapolation
        fill_value=None
    )
    newpoint = np.array([batch_size, seq_len, block_num])
    estimated_time = interpolator(newpoint)
    return estimated_time * 1000


def getGemvProfile(block_num, seq_len, time):
    # Binary search for batch size
    low = 1
    high = 4096
    while low < high:
        mid = (low + high) // 2
        current_time = getGemvTime(mid, seq_len, block_num)
        if current_time is None:
            # If interpolation couldn't estimate, adjust search range
            low = mid + 1
            continue
        if current_time < time:
            low = mid + 1
        else:
            high = mid
    return low


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", type=int, default=128, help="m dim")
    arg_parser.add_argument("-n", type=int, default=1024, help="n dim")
    arg_parser.add_argument("-k", type=int, default=8192, help="k dim")
    layout_choices = ["R", "C", "B"]

    arg_parser.add_argument("-a", choices=layout_choices, default="B", help="A layout, (R)ow, (C)olumn, or (B)oth")
    arg_parser.add_argument("-b", choices=layout_choices, default="B", help="B layout, (R)ow, (C)olumn, or (B)oth")
    arg_parser.add_argument("-d", choices=layout_choices, default="B", help="D layout, (R)ow, (C)olumn, or (B)oth")

    arg_parser.add_argument("--save_path", type=str, default="./out.txt", help="The path to save the results")
    args = arg_parser.parse_args()
    
    layout_name = {
        "R": "RowMajor",
        "C": "ColumnMajor",
        "B": None
    }

    # Fetch GEMM profile based on input arguments
    df = getGemmProfile(
        (args.m, args.n, args.k),
        (layout_name[args.a], layout_name[args.b], layout_name[args.d])
    )
    
    # Save the resulting DataFrame to the specified path
    df.to_csv(args.save_path, index=False)
    
    # Example usage (commented out)
    # print(getGemmProfile((640, 1024, 8192), (None, None, None)))
    # print(getAllReduceProfile(getSizeofComm(1024)))
    # print(getGemvProfile(4, 1024, 0.10))
