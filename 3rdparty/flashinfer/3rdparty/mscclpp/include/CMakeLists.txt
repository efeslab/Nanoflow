# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS *.hpp)
target_sources(mscclpp_obj PUBLIC FILE_SET HEADERS FILES ${HEADERS})
