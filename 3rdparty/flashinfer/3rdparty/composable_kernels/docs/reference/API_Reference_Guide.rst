.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _api-reference:

********************************************************************
API reference guide
********************************************************************


This document contains details of the APIs for the Composable Kernel (CK) library and introduces
some of the key design principles that are used to write new classes that extend CK functionality.

=================
Using CK API
=================

This section describes how to use the CK library API.

=================
CK Datatypes
=================

-----------------
DeviceMem
-----------------

.. doxygenstruct:: DeviceMem

---------------------------
Kernels For Flashattention
---------------------------

The Flashattention algorithm is defined in :cite:t:`dao2022flashattention`. This section lists
the classes that are used in the CK GPU implementation of Flashattention.

**Gridwise classes**

.. doxygenstruct:: ck::GridwiseBatchedGemmSoftmaxGemm_Xdl_CShuffle

**Blockwise classes**

.. doxygenstruct:: ck::ThreadGroupTensorSliceTransfer_v4r1

.. doxygenstruct:: ck::BlockwiseGemmXdlops_v2

.. doxygenstruct:: ck::BlockwiseSoftmax

**Threadwise classes**

.. doxygenstruct:: ck::ThreadwiseTensorSliceTransfer_StaticToStatic

.. bibliography::
