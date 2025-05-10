.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _what-is-ck:

********************************************************************
What is the Composable Kernel library
********************************************************************


Methodology
===========

The Composable Kernel (CK) library provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs and CPUs, through general purpose kernel languages like HIP C++.

CK utilizes two concepts to achieve performance portability and code maintainability:

* A tile-based programming model
* Algorithm complexity reduction for complex ML operators using an innovative technique called
  "Tensor Coordinate Transformation".

.. image:: ../data/ck_component.png
   :alt: CK Components


Code Structure
==============

The CK library is structured into 4 layers:

* "Templated Tile Operators" layer
* "Templated Kernel and Invoker" layer
* "Instantiated Kernel and Invoker" layer
* "Client API" layer

It also includes a simple wrapper component used to perform tensor transform operations more easily and with fewer lines of code.

.. image:: ../data/ck_layer.png
   :alt: CK Layers
   