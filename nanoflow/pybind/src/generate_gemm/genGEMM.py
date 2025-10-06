import os
import math
import re

def getCutlassMajorName(x):
    if (x == 1):
        return "RowMajor"
    else :
        return "ColumnMajor"
    
def getTensorMajorName(x):
    if (x == 1):
        return "row"
    else :
        return "col"

def getGemmCanonicalName(cta, warp, split, stage, major, id):
    return f"{cta[0]}_{cta[1]}_{cta[2]}_{warp[0]}_{warp[1]}_{warp[2]}_{split}_{stage}_{getCutlassMajorName(major[0])}_{getCutlassMajorName(major[1])}_{getCutlassMajorName(major[2])}"

def getH100GemmCanonicalName(cta, cluster, split, schedule, major, id):
    return f"SM90_{cta[0]}_{cta[1]}_{cta[2]}_{cluster[0]}_{cluster[1]}_{cluster[2]}_{split}_{getCutlassMajorName(major[0])}_{getCutlassMajorName(major[1])}_{getCutlassMajorName(major[2])}_{schedule}"

schedule_map = {
    "stream_k_warpspecialized_cooperative_epi_nosmem": ("cutlass::epilogue::NoSmemWarpSpecialized",             "cutlass::gemm::KernelTmaWarpSpecializedCooperative", "cutlass::gemm::StreamKScheduler"),
    "stream_k_warpspecialized_cooperative_epi_tma":    ("cutlass::epilogue::TmaWarpSpecializedCooperative",     "cutlass::gemm::KernelTmaWarpSpecializedCooperative", "cutlass::gemm::StreamKScheduler"),
    "warpspecialized_cooperative_epi_nosmem":          ("cutlass::epilogue::NoSmemWarpSpecialized",             "cutlass::gemm::KernelTmaWarpSpecializedCooperative", "cutlass::gemm::PersistentScheduler"),
    "warpspecialized_cooperative_epi_tma":             ("cutlass::epilogue::TmaWarpSpecializedCooperative",     "cutlass::gemm::KernelTmaWarpSpecializedCooperative", "cutlass::gemm::PersistentScheduler"),
    "warpspecialized_epi_nosmem":                      ("cutlass::epilogue::NoSmemWarpSpecialized",             "cutlass::gemm::KernelTmaWarpSpecialized",            "cutlass::gemm::PersistentScheduler"),
    "warpspecialized_pingpong_epi_nosmem":             ("cutlass::epilogue::NoSmemWarpSpecialized",             "cutlass::gemm::KernelTmaWarpSpecializedPingpong",    "cutlass::gemm::PersistentScheduler"),
    "warpspecialized_pingpong_epi_tma":                ("cutlass::epilogue::TmaWarpSpecialized",                "cutlass::gemm::KernelTmaWarpSpecializedPingpong",    "cutlass::gemm::PersistentScheduler"),
    "auto":                                            ("cutlass::epilogue::collective::EpilogueScheduleAuto",  "cutlass::gemm::collective::KernelScheduleAuto",      "cutlass::gemm::PersistentScheduler"),
}
def parse_H100_cutlass_signiture(signature):
    original_signature = signature
    signature = signature.split("_")
    cta_mnk = signature[9].split("x")
    cluster_mnk = signature[10].split("x")
    feature = re.split("align8", original_signature)[1]
    # if start with "_", remove "_"
    feature = re.sub("^_", "", feature)
    if feature == "":
        feature = "auto"
    
    return cta_mnk, cluster_mnk, feature


def constructInstantiation(cta, warp, split, stage, major, id):

    content = f"""template class CutlassGEMMWrapper<{cta[0]:4},{cta[1]:4},{cta[2]:4},{warp[0]:4},{warp[1]:4},{warp[2]:4},{split:2},{stage:2},\
 cutlass::layout::{getCutlassMajorName(major[0])}, cutlass::layout::{getCutlassMajorName(major[1])}, cutlass::layout::{getCutlassMajorName(major[2])}>;
""" 
    return content

def constructExternDeclaration(cta, warp, split, stage, major, id):
    tmpl_content = f"""extern template class CutlassGEMMWrapper<{cta[0]:4},{cta[1]:4},{cta[2]:4},{warp[0]:4},{warp[1]:4},{warp[2]:4},{split:2},{stage:2},\
 cutlass::layout::{getCutlassMajorName(major[0])}, cutlass::layout::{getCutlassMajorName(major[1])}, cutlass::layout::{getCutlassMajorName(major[2])}>;
"""
    gen_func = f"""extern BaseGEMMWrapper * gen_{getGemmCanonicalName(cta, warp, split, stage, major, id)}();"""
    #return tmpl_content + gen_func
    return gen_func + '\n'

def constructGemmFactory(cta, warp, split, stage, major, id):
    content = f"""BaseGEMMWrapper * gen_{getGemmCanonicalName(cta, warp, split, stage, major, id)}() {{
        return new CutlassGEMMWrapper<{cta[0]},{cta[1]},{cta[2]},{warp[0]},{warp[1]},{warp[2]},{split},{stage},\
cutlass::layout::{getCutlassMajorName(major[0])}, cutlass::layout::{getCutlassMajorName(major[1])}, cutlass::layout::{getCutlassMajorName(major[2])}>;
    }}
"""
    return content + constructInstantiation(cta, warp, split, stage, major, id);

# Iterate through all configs and execute a generic callback
def genCfg(func):
    for c in configs:
        cta, warp, stage = parseConfigs(c)
        for split in split_list:
            for major in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
                func(cta, warp, split, stage, major, count)

def genCfgH100(func):
    for c in h100_configs:
        cta, cluster, schedule = parse_H100_cutlass_signiture(c)
        for split in [1]:
            for major in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
                func(cta, cluster, split, schedule, major, count)

def H100_declearation(cta, cluster, split, schedule, major, id):
    content = f"""CutlassH100GEMMWrapper<{cta[0]:4},{cta[1]:4},{cta[2]:4},{cluster[0]:2},{cluster[1]:2},{cluster[2]:2}, {split:2}, \
cutlass::layout::{getCutlassMajorName(major[0])}, cutlass::layout::{getCutlassMajorName(major[1])}, cutlass::layout::{getCutlassMajorName(major[2])},\
{schedule_map[schedule][0]}, {schedule_map[schedule][1]}, {schedule_map[schedule][2]}>"""
    return content

def constructH100Instantiation(cta, cluster, split, schedule, major, id):
    content = f"""template class {H100_declearation(cta, cluster, split, schedule, major, id)};
"""
    return content

def constructH100ExternDeclaration(cta, cluster, split, schedule, major, id):
    gen_func = f"""extern BaseGEMMWrapper * gen_{getH100GemmCanonicalName(cta, cluster, split, schedule, major, id)}();"""
    return gen_func + '\n'

def constructH100GemmFactory(cta, cluster, split, schedule, major, id):
    content = f"""BaseGEMMWrapper * gen_{getH100GemmCanonicalName(cta, cluster, split, schedule, major, id)}() {{
        return new {H100_declearation(cta, cluster, split, schedule, major, id)}();
    }}
"""
    return content + constructH100Instantiation(cta, cluster, split, schedule, major, id);

# Iterate through all configs and generate code to a file
def genCode(filename, func, mode='w'):
    with open(filename, mode) as f:
        #genCfg(lambda cta, warp, split, stage, major, id: f.write(func(cta, warp, split, stage, major, id)))
        genCfg(lambda *args: f.write(func(*args)))
def genH100Code(filename, func, mode='w'):
    with open(filename, mode) as f:
        genCfgH100(lambda *args: f.write(func(*args)))

def parseConfigs(s):
    segments = s.split('_')
    cta_m = int(segments[5].split('x')[0])
    cta_n = int(segments[5].split('x')[1])
    cta_k = int(segments[6].split('x')[0])
    stage = int(segments[6].split('x')[1])
    warp_m = cta_m // int(segments[9])
    warp_n = cta_n // int(segments[10])
    warp_k = cta_k // int(segments[11])
    return [cta_m, cta_n, cta_k], [warp_m, warp_n, warp_k], stage

def GetAllH100GemmCanonicalNames():
    names = []
    genCfgH100(lambda *args: names.append(getH100GemmCanonicalName(*args)))
    return names

count = 0
configs = ['cutlass_tensorop_f16_s16816gemm_f16_128x128_32x3_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x128_64x3_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x128_64x4_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x256_32x3_nn_align8_2_4_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x256_64x3_nn_align8_2_4_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x64_32x6_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_128x64_64x3_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align8_4_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x128_64x3_nn_align8_4_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x64_32x3_nn_align8_4_1_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x64_32x4_nn_align8_4_1_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x64_64x3_nn_align8_4_1_1', 'cutlass_tensorop_f16_s16816gemm_f16_256x64_64x4_nn_align8_4_1_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x128_32x6_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x128_64x3_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x256_32x4_nn_align8_1_4_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x256_64x3_nn_align8_1_4_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x256_64x4_nn_align8_1_4_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x64_32x10_nn_align8_2_2_1', 'cutlass_tensorop_f16_s16816gemm_f16_64x64_64x5_nn_align8_2_2_1', 'cutlass_tensorop_f16_s1688gemm_f16_128x128_32x2_nn_align8_2_2_1', 'cutlass_tensorop_f16_s1688gemm_f16_128x256_32x2_nn_align8_2_4_1', 'cutlass_tensorop_f16_s1688gemm_f16_128x64_32x2_nn_align8_2_2_1', 'cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8_4_2_1', 'cutlass_tensorop_f16_s1688gemm_f16_256x64_32x2_nn_align8_4_1_1', 'cutlass_tensorop_f16_s1688gemm_f16_64x128_32x2_nn_align8_2_2_1', 'cutlass_tensorop_f16_s1688gemm_f16_64x128_64x2_nn_align8_1_2_2', 'cutlass_tensorop_f16_s1688gemm_f16_64x256_32x2_nn_align8_1_4_1', 'cutlass_tensorop_f16_s1688gemm_f16_64x64_32x2_nn_align8_2_2_1']
h100_configs = ['cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_warpspecialized_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_1x2x1_0_nnn_align8_warpspecialized_pingpong_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_warpspecialized_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_128x128x64_2x1x1_0_nnn_align8_warpspecialized_pingpong_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x1x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x2x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16_f32_f16_f16_256x128x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x1x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x2x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x2x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_1x2x1_0_nnn_align8_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_2x1x1_0_nnn_align8',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_2x1x1_0_nnn_align8_stream_k_warpspecialized_cooperative_epi_tma',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_nosmem',
'cutlass3x_sm90_tensorop_s64x256x16gemm_f16_f16_f32_f16_f16_128x256x64_2x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma']
split_list = [1, 2, 3, 4, 5, 6]

if __name__ == "__main__":
    genCode("cutlassGemmExternDeclearation.cuh", constructExternDeclaration)
    genH100Code("cutlassGemmExternDeclearation.cuh", constructH100ExternDeclaration, 'a')
    genCode("cutlassGemmFactory.gen", constructGemmFactory)
    genH100Code("cutlassGemmFactory.gen", constructH100GemmFactory, 'a')

    # LINE_LIMIT is determined by how many lines of code were generated per GEMM config and how many GEMM configs we want to put in one file
    LINE_LIMIT = 100
    with open("cutlassGemmFactory.gen" , 'r') as f:
        content = f.readlines()
        gen_id_width = math.ceil(math.log10(math.ceil(len(content) / LINE_LIMIT)))
        for i in range(0, len(content), LINE_LIMIT):
            with open(f"cutlassGemmFactory_{i//LINE_LIMIT:0{gen_id_width}d}.cu", 'w') as f:
                str_content = "".join(content[i:min(i+LINE_LIMIT, len(content))])
                f.write("#include \"gemmFactory.cuh\"\n")
                f.write(str_content + '\n')

    with open("gemmFactory.cu", 'w') as f:
        map_entries = []
        genCfg(lambda *args: map_entries.append(
            f"""{{"{getGemmCanonicalName(*args)}", &gen_{getGemmCanonicalName(*args)}}}"""
        ))
        genCfgH100(lambda *args: map_entries.append(
            f"""{{"{getH100GemmCanonicalName(*args)}", &gen_{getH100GemmCanonicalName(*args)}}}"""
        ))
        templateContent = open("gemmFactory.in", 'r').read()
        f.write(templateContent.replace("@map_entries@", ',\n'.join(map_entries)))
