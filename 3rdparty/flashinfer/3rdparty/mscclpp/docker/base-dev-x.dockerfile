ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source https://github.com/microsoft/mscclpp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        htop \
        lcov \
        vim \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install cmake 3.26.4
ENV CMAKE_VERSION="3.26.4"
ENV CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64" \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
RUN curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local && \
    rm -rf ${CMAKE_HOME}.tar.gz
ENV PATH="/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"

# Install Python dependencies
ADD . /tmp/mscclpp
WORKDIR /tmp/mscclpp
ARG TARGET="cuda12.1"
RUN target_type=$(echo $TARGET | sed 's/\.[0-9]*$//') && \
    python3 -m pip install --no-cache-dir -r python/requirements_${target_type}.txt

# Set PATH
RUN echo PATH="${PATH}" > /etc/environment

# Cleanup
RUN rm -rf /tmp/mscclpp
WORKDIR /
