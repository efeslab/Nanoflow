.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _docker-hub:

********************************************************************
CK Docker Hub
********************************************************************

Why do I need this?
===================

To make things simpler, and bring Composable Kernel and its dependencies together, 
docker images can be found on `Docker Hub <https://hub.docker.com/r/rocm/composable_kernel/tags>`_. Docker images provide a complete image of the OS, the Composable Kernel library, and its dependencies in a single downloadable file. 

Refer to `Docker Overview <https://docs.docker.com/get-started/overview/>`_ for more information on Docker images and containers.

Which image is right for me?
============================

The image naming includes information related to the docker image. 
For example ``ck_ub20.04_rocm6.0`` indicates the following:

* ``ck`` - made for running Composable Kernel;
* ``ub20.04`` - based on Ubuntu 20.04;
* ``rocm6.0`` - ROCm platform version 6.0.

Download a docker image suitable for your OS and ROCm release, run or start the docker container, and then resume the tutorial from this point. Use the ``docker pull`` command to download the file::

    docker pull rocm/composable_kernel:ck_ub20.04_rocm6.0


What is inside the image?
-------------------------

The docker images have everything you need for running CK including:

* `ROCm <https://rocm.docs.amd.com/en/latest/index.html>`_
* `CMake <https://cmake.org/getting-started/>`_
* `Compiler <https://github.com/ROCm/llvm-project>`_
* `Composable Kernel library <https://github.com/ROCm/composable_kernel>`_

Running the docker container
============================

After downloading the docker image, you can start the container using one of a number of commands. Start with the ``docker run`` command as shown below::

    docker run                                                            \
    -it                                                                   \
    --privileged                                                          \
    --group-add sudo                                                      \
    -w /root/workspace                                                    \
    -v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace                         \
    rocm/composable_kernel:ck_ub20.04_rocm6.0                             \
    /bin/bash

After starting the bash shell, the docker container current folder is `~/workspace`. The library path is ``~/workspace/composable_kernel``. Navigate to the library to begin the tutorial as explained in :ref:`hello-world`:

.. note::

    If your current folder is different from `${HOME}`, adjust the line ``-v ${HOME}:/root/workspace`` in the ``docker run`` command to fit your folder structure.

Stop and restart the docker image
=================================

After finishing the tutorial, or just when you have completed your work session, you can close the docker container, or stop the docker container to restart it at another time. Closing the docker container means that it is still in the active state, and can be resumed from where you left it. Stopping the container closes it, and returns the image to its initial state. 

Use the ``Ctrl-D`` option to exit the container, while leaving it active, so you can return to the container in its current state to resume the tutorial, or pickup your project where you left off. 

To restart the active container use the ``docker exec`` command to specify the container name and options as follows::

    docker exec -it <container_name> bash

Where: 

* `exec` is the docker command
* `-it` is the interactive option for `exec`
* `<container_name>` specifies an active container on the system
* `bash` specifies the command to run in the interactive shell

.. note::

    You can use the ``docker container ls`` command to list the active containers on the system.

To start a container from the image, use the ``docker start`` command::

    docker start <container_name>

Then use the docker exec command as shown above to start the bash shell. 

Use the ``docker stop`` command to stop the container and restore the image to its initial state::

    docker stop <container_name>
    
Editing the docker image
=======================

If you want to customize the docker image, edit the
`Dockerfile <https://github.com/ROCm/composable_kernel/blob/develop/Dockerfile>`_
from the GitHub repository to suit your needs.
