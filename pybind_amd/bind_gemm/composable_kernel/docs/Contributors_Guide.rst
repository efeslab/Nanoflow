.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _contributing-to:

********************************************************************
Contributor's guide
********************************************************************

This chapter explains the rules for contributing to the Composable Kernel project, and how to contribute.

Getting started
===============

#. **Documentation:** Before contributing to the library, familiarize yourself with the
   `Composable Kernel User Guide <https://rocm.docs.amd.com/projects/composable_kernel/en/latest/>`_.
   It provides insight into the core concepts, environment configuration, and steps to obtain or
   build the library. You can also find some of this information in the
   `README file <https://github.com/ROCm/composable_kernel/blob/develop/README.md>`_
   on the project's GitHub page.
#. **Additional reading:** The blog post `AMD Composable Kernel library: efficient fused kernels for AI apps with just a few lines of code <https://community.amd.com/t5/instinct-accelerators/amd-composable-kernel-library-efficient-fused-kernels-for-ai/ba-p/553224>`_ provides a deeper understanding of the CK library and showcases its performance capabilities.
   <https://community.amd.com/t5/instinct-accelerators/amd-composable-kernel-library-efficient-fused-kernels-for-ai/ba-p/553224>`_
   from the AMD Community portal. It offers a deeper understanding of the library's objectives and showcases its performance capabilities.
#. **General information:** For broader information about AMD products, consider exploring the
   `AMD Developer Central portal <https://www.amd.com/en/developer.html>`_.

How to contribute
===================

You can make an impact by reporting issues or proposing code enhancements through pull requests.

Reporting issues
----------------

Use `Github issues <https://github.com/ROCm/composable_kernel/issues>`_
to track public bugs and enhancement requests.

If you encounter an issue with the library, please check if the problem has already been
reported by searching existing issues on GitHub. If your issue seems unique, please submit a new
issue. All reported issues must include:

* A comprehensive description of the problem, including:

  * What did you observe?
  * Why do you think it is a bug (if it seems like one)?
  * What did you expect to happen? What would indicate the resolution of the problem?
  * Are there any known workarounds?

* Your configuration details, including:

  * Which GPU are you using?
  * Which OS version are you on?
  * Which ROCm version are you using?
  * Are you using a Docker image? If so, which one?

* Steps to reproduce the issue, including:

  * What actions trigger the issue? What are the reproduction steps?

    * If you build the library from scratch, what CMake command did you use?

  * How frequently does this issue happen? Does it reproduce every time? Or is it a sporadic issue?

Before submitting any issue, ensure you have addressed all relevant questions from the checklist.

Creating Pull Requests
----------------------

You can submit `Pull Requests (PR) on GitHub
<https://github.com/ROCm/composable_kernel/pulls>`_.

All contributors are required to develop their changes on a separate branch and then create a
pull request to merge their changes into the `develop` branch, which is the default
development branch in the Composable Kernel project. All external contributors must use their own
forks of the project to develop their changes.

When submitting a Pull Request you should:

* Describe the change providing information about the motivation for the change and a general
  description of all code modifications.

* Verify and test the change:

  * Run any relevant existing tests.
  * Write new tests if added functionality is not covered by current tests.

* Ensure your changes align with the coding style defined in the ``.clang-format`` file located in
  the project's root directory. We leverage `pre-commit` to run `clang-format` automatically. We
  highly recommend contributors utilize this method to maintain consistent code formatting.
  Instructions on setting up `pre-commit` can be found in the project's
  `README file <https://github.com/ROCm/composable_kernel/blob/develop/README.md>`_

* Link your PR to any related issues:

  * If there is an issue that is resolved by your change, please provide a link to the issue in
    the description of your pull request.

* For larger contributions, structure your change into a sequence of smaller, focused commits, each
  addressing a particular aspect or fix.

Following the above guidelines ensures a seamless review process and faster assistance from our
end.

Thank you for your commitment to enhancing the Composable Kernel project! 
