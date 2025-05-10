# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
from dataclasses import dataclass
import fnmatch
import itertools
from pathlib import Path
from typing import List, Optional, Tuple

from codegen.cmake_config import *
from codegen.cpp_symbol_map import *

from codegen.ops.fmha_fwd import (
    FmhaFwdApiTrait,
    DTYPE_BITS,
    FMHA_FWD_KERNEL_HEADER,
    FMHA_FWD_API_PER_DTYPE,
    FMHA_FWD_API_PER_HDIM_CASE,
)


FMHA_FWD_APPENDKV_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_trait_{F_idx} = ck_tile::TileFmhaFwdAppendKVTraits<{F_spad},
                                                    {F_skpad},
                                                    {F_dpad},
                                                    {F_dvpad},
                                                    {F_occupancy}>;

using fmha_pipeline_problem_{F_idx} = ck_tile::BlockFmhaFwdAppendKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    {F_bs},
    {F_bsk},
    {F_bd},
    {F_bdv},
    {F_vlayout},
    {F_rope},
    {F_pagedkv},
    fmha_trait_{F_idx}>;

using fmha_pipeline_{F_idx} = ck_tile::BlockFmhaFwdAppendKVPipeline<
    fmha_pipeline_problem_{F_idx}>;

using fmha_kernel_{F_idx} = ck_tile::FmhaFwdAppendKVKernel<fmha_pipeline_{F_idx}>;

using trait_{F_idx} = fmha_fwd_appendkv_traits_<{F_hdim}, {F_dtype}, {F_bs}, {F_bsk}, {F_bd}, {F_bdv}, {F_vlayout},
                        {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_rope}, {F_pagedkv}>;

#include <iostream>

template<>
float fmha_fwd_appendkv_<trait_{F_idx}>(const ck_tile::stream_config& s, fmha_fwd_appendkv_args a)
{{
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids] = fmha_fwd_appendkv_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
}}
"""

FMHA_FWD_APPENDKV_API_FILENAME="fmha_fwd_appendkv_api.cpp"
FMHA_FWD_APPENDKV_API="""
float fmha_fwd_appendkv(fmha_fwd_appendkv_traits t, fmha_fwd_appendkv_args a, const ck_tile::stream_config& s){{
    float r = -1;
{F_dispatch}
    return r;
}}
"""

FMHA_FWD_APPENDKV_API_INNER_DISPATCH="""            {F_if}((t.is_v_rowmajor == {F_vlayout}) &&
                        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck}) && (t.rope_type == {F_rope_check}) &&
                        ((a.block_table_ptr != nullptr) == {F_pagedkv})) {{
                using trait_ = fmha_fwd_appendkv_traits_<{F_hdim}, {F_dtype}, {F_bs}, {F_bsk}, {F_bd}, {F_bdv}, {F_vlayout}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_rope}, {F_pagedkv}>;
                return fmha_fwd_appendkv_<trait_>(s, a);
            }}
"""

@dataclass
class FmhaFwdAppendKVApiTrait:
    # sync with fmha_fwd_traits<>, to generate fallback calls
    hdim      : str
    dtype     : str  # data type
    bs        : int  # tile size along q seqlen
    bsk       : int  # tile size along k seqlen
    bd        : int  # tile size along qk gemm unroll
    bdv       : int  # tile size along kv gemm unroll
    vlayout   : str
    spad      : str
    skpad     : str
    dpad      : str
    dvpad     : str
    rope      : str # key from ROPE_MAP
    pagedkv   : str

    @property
    def name(self) -> str:
        return f'{self.hdim}-{self.dtype}-{self.bs}-{self.bsk}-{self.bd}-{self.bdv}-{self.vlayout}-'+\
               f'{self.spad}-{self.skpad}-{self.dpad}-{self.dvpad}-{self.rope}-{self.pagedkv}'

    @property
    def scheck(self) -> str:
        if self.spad == 't' : return f'true /*a.seqlen_q % {self.bs} != 0*/'
        else :                return f'a.seqlen_q % {self.bs} == 0'

    @property
    def skcheck(self) -> str:
        # we do not check all the values in a.seqlen_k_ptr
        return 'true'

    @property
    def dcheck(self) -> str:
        if self.dpad == 't': return f'true /*a.hdim_q % {self.bd} != 0*/' # TODO: order of get_pipelines() matters! (ugly)
        else :               return f'a.hdim_q % {self.bd} == 0'

    @property
    def dvcheck(self) -> str:
        if self.dvpad == 't': return f'true /*a.hdim_v % {self.bdv} != 0*/' # TODO: order of get_pipelines() matters! (ugly)
        else :                return f'a.hdim_v % {self.bdv} == 0'

@dataclass
class FmhaFwdAppendKVPipeline:
    F_vlayout   : str  # row/col
    F_spad      : str  # true/false
    F_skpad     : str  #
    F_dpad      : str  #
    F_dvpad     : str  #
    F_rope      : str  # key from ROPE_MAP
    F_pagedkv   : str  # t/f

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_skpad == 't' : n += 'sk'
            if self.F_dpad == 't' : n += 'd'
            if self.F_dvpad == 't' : n += 'dv'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f'v{self.F_vlayout[0]}'
        if pn != '' : n += f'_{pn}'
        if self.F_rope != 'no': n += f'_{self.F_rope}'
        if self.F_pagedkv == 't': n += '_pagedkv'
        return n

class FmhaFwdAppendKVApiPool:
    def __init__(self, mask_impl):
        self.pool = dict()
        self.mask_impl = mask_impl

    def register_traits(self, trait : FmhaFwdApiTrait) -> None:
        # TODO: do we need to check duplication?
        if trait.dtype not in self.pool.keys():
            self.pool[trait.dtype] = dict()
        if trait.hdim not in self.pool[trait.dtype].keys():
            self.pool[trait.dtype][trait.hdim] = list()

        self.pool[trait.dtype][trait.hdim].append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_dtypes=str()
        for i, dtype in enumerate(self.pool.keys()):
            per_hdim_case=str()
            for j, hdim in enumerate(self.pool[dtype].keys()):
                traits=self.pool[dtype][hdim]
                inners=str()
                for k, trait in enumerate(traits):
                    if_k = 'if' if k == 0 else 'else if'
                    inners = inners + FMHA_FWD_APPENDKV_API_INNER_DISPATCH.format(F_if=if_k, F_vlayout=LAYOUT_MAP[trait.vlayout],
                                   F_scheck=trait.scheck, F_skcheck=trait.skcheck, F_dcheck=trait.dcheck, F_dvcheck=trait.dvcheck, F_rope_check=ROPE_CHECK_MAP[trait.rope],
                                   F_pagedkv=BOOL_MAP[trait.pagedkv], F_spad=BOOL_MAP[trait.spad], F_skpad=BOOL_MAP[trait.skpad], F_dpad=BOOL_MAP[trait.dpad], F_dvpad=BOOL_MAP[trait.dvpad],
                                   F_rope=ROPE_MAP[trait.rope], F_bs=trait.bs, F_bsk=trait.bsk, F_bd=trait.bd, F_bdv=trait.bdv, F_hdim=hdim, F_dtype=FWD_DTYPE_MAP[dtype])
                if_j = 'if' if j == 0 else 'else if'
                per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_inner_dispatch=inners)
            if_i = 'if' if i == 0 else 'else if'
            per_dtypes = per_dtypes + FMHA_FWD_API_PER_DTYPE.format(F_if=if_i, F_dtype=dtype, F_hdim_case=per_hdim_case)
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_APPENDKV_API.format(F_dispatch = per_dtypes)

@dataclass
class FmhaFwdAppendKVTileSize:
    F_bs        : int  # tile size along q seqlen
    F_bsk       : int  # tile size along k seqlen
    F_bd        : int  # tile size along qk gemm unroll
    F_bdv       : int  # tile size along kv gemm unroll
    F_occupancy : int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy
    @property
    def name(self) -> str:
        return f"b{self.F_bs}x{self.F_bsk}x{self.F_bd}x{self.F_bdv}" +\
            ("" if self.F_occupancy == -1 else f"_o{self.F_occupancy}")

@dataclass
class FmhaFwdAppendKVKernel:
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_tile          : FmhaFwdAppendKVTileSize
    F_pipeline      : FmhaFwdAppendKVPipeline
    mask_impl       : str

    @property
    def template(self) -> str:
        kernel_body = str()
        return FMHA_FWD_KERNEL_HEADER + \
            FMHA_FWD_APPENDKV_KERNEL_BODY.format(
                F_idx           = self.F_idx,
                F_hdim          = self.F_hdim,
                F_dtype         = FWD_DTYPE_MAP[self.F_dtype],
                F_bs            = self.F_tile.F_bs,
                F_bsk           = self.F_tile.F_bsk,
                F_bd            = self.F_tile.F_bd,
                F_bdv           = self.F_tile.F_bdv,
                F_vlayout       = LAYOUT_MAP[self.F_pipeline.F_vlayout],
                F_spad          = BOOL_MAP[self.F_pipeline.F_spad],
                F_skpad         = BOOL_MAP[self.F_pipeline.F_skpad],
                F_dpad          = BOOL_MAP[self.F_pipeline.F_dpad],
                F_dvpad         = BOOL_MAP[self.F_pipeline.F_dvpad],
                F_rope          = ROPE_MAP[self.F_pipeline.F_rope],
                F_pagedkv       = BOOL_MAP[self.F_pipeline.F_pagedkv],
                F_occupancy     = self.F_tile.F_occupancy)

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return f"fmha_fwd_appendkv_d{self.F_hdim}_{self.F_dtype}_" + \
                self.F_tile.name + '_' + self.F_pipeline.name

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

    def api_trait(self) -> FmhaFwdAppendKVApiTrait:
        return FmhaFwdAppendKVApiTrait(
                hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                bs=self.F_tile.F_bs,
                bsk=self.F_tile.F_bsk,
                bd=self.F_tile.F_bd,
                bdv=self.F_tile.F_bdv,
                vlayout=self.F_pipeline.F_vlayout,
                spad=self.F_pipeline.F_spad,
                skpad=self.F_pipeline.F_skpad,
                dpad=self.F_pipeline.F_dpad,
                dvpad=self.F_pipeline.F_dvpad,
                rope=self.F_pipeline.F_rope,
                pagedkv=self.F_pipeline.F_pagedkv)

# TODO: design a more practical way to do it
# this is current supported tile size per hdim
def get_fmha_fwd_appendkv_tile_dict_from_dtype(dtype : str) -> Optional[dict]:
    if dtype == 'fp16' or dtype == 'bf16':
        return {
            '32'  : FmhaFwdAppendKVTileSize(64, 64,  32,  32, -1),
            '64'  : FmhaFwdAppendKVTileSize(64, 64,  64,  64, -1),
            '128' : FmhaFwdAppendKVTileSize(64, 64, 128, 128, -1),
            '256' : FmhaFwdAppendKVTileSize(64, 64, 256, 256, -1),
        }
    elif dtype == 'fp8' or dtype == 'bf8':
        return {
            '64'  : FmhaFwdAppendKVTileSize(64, 64,  64,  64, -1),
            '128' : FmhaFwdAppendKVTileSize(64, 64, 128, 128, -1),
            '256' : FmhaFwdAppendKVTileSize(64, 64, 256, 256, -1)
        }
    else:
        return None

def get_fwd_appendkv_blobs(kernel_filter : Optional[str], receipt, mask_impl) -> Tuple[FmhaFwdAppendKVApiPool, List[FmhaFwdAppendKVKernel]]:
    # TODO: we don't support tuning yet, so pick up one value for vlayout/pipeline/pad
    #       support this in future
    def get_pipelines(dtype, hdim) -> List[FmhaFwdAppendKVPipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr pipeline, let 't' padding to appear later!!
        # TODO: how to design this more generic?
        squant = 't' if dtype == 'fp8' else 'f'
        pipelines = []
        if dtype in ['fp16', 'bf16']:
            # NOTICE: it will be very complicated if we consider all the hdim_q padding cases while
            #         applying rotary embedding, so I just use 't' in inter/half pipelines
            for vlayout in ['row', 'col']:
                for pagedkv in ["t", "f"]:
                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 'f', 't', 'f', 'f', 'no', pagedkv))
                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 't', 't', 't', 't', 'no', pagedkv))

                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 'f', 't', 't', 'f', 'inter', pagedkv))
                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 't', 't', 't', 't', 'inter', pagedkv))

                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 'f', 't', 't', 'f', 'half', pagedkv))
                    pipelines.append(FmhaFwdAppendKVPipeline(vlayout, 't', 't', 't', 't', 'half', pagedkv))
        elif dtype in ['fp8', 'bf8']:
            # rope/paged-kv is not supported
            pipelines.append(FmhaFwdAppendKVPipeline('col', 't', 't', 't', 't', 'no', 'f'))
        elif dtype in ['fp8fp16', 'fp8bf16']:
            # TODO
            None
        else:
            assert False
        return pipelines

    gen = list()
    api_pool = FmhaFwdAppendKVApiPool(mask_impl)

    for dtype in FWD_DTYPE_MAP.keys():
        d = get_fmha_fwd_appendkv_tile_dict_from_dtype(dtype)
        if d == None:
            continue
        for hdim_str in d.keys():
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in get_pipelines(dtype, hdim):
                k = FmhaFwdAppendKVKernel(F_idx=0,
                                  F_hdim=hdim,
                                  F_dtype=dtype,
                                  F_tile=tile,
                                  F_pipeline=pipeline,
                                  mask_impl=mask_impl)
                if kernel_filter != None:
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                if receipt == 2:
                    cond = dtype in ['fp16', 'bf16']
                    cond &= pipeline.F_vlayout == 'row'
                    if not cond:
                        continue
                api_pool.register_traits(k.api_trait())
                gen.append(k)

    return (api_pool, gen)

def write_single_kernel(kernel: FmhaFwdAppendKVKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_fwd_appendkv_api(api_pool : FmhaFwdAppendKVApiPool, autogen_dir: Path) -> None:
    (autogen_dir / FMHA_FWD_APPENDKV_API_FILENAME).write_text(api_pool.api)

def write_blobs(output_dir : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    api_pool, kernels = get_fwd_appendkv_blobs(kernel_filter, receipt, mask_impl)
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    write_fwd_appendkv_api(api_pool, output_dir)

def list_blobs(file_path : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    with file_path.open('a') as f:
        _, kernels = get_fwd_appendkv_blobs(kernel_filter, receipt, mask_impl)
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_APPENDKV_API_FILENAME) + "\n")
