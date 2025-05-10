FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ARG ROCMVERSION=6.1
ARG compiler_version=""
ARG compiler_commit=""
ARG CK_SCCACHE=""

RUN set -xe

ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
RUN useradd -rm -d /home/jenkins -s /bin/bash -u 1004 jenkins
# Add rocm repository
RUN chmod 1777 /tmp
RUN apt-get update
RUN apt-get install -y --allow-unauthenticated apt-utils wget gnupg2 curl

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
RUN curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

RUN if [ "$ROCMVERSION" != "6.2" ]; then \
        sh -c "wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb  --no-check-certificate" && \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ./amdgpu-install_6.1.60100-1_all.deb && \
        wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
        sh -c "echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] $DEB_ROCM_REPO focal main > /etc/apt/sources.list.d/rocm.list" && \
        sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCMVERSION/ubuntu focal main > /etc/apt/sources.list.d/amdgpu.list'; \
    elif [ "$ROCMVERSION" = "6.2" ] && [ "$compiler_version" = "rc2" ]; then \
        sh -c "wget http://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_6.1-20.04-1_all.deb --no-check-certificate" && \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install dialog && DEBIAN_FRONTEND=noninteractive apt-get install ./amdgpu-install-internal_6.1-20.04-1_all.deb && \
        sh -c 'echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-release-archive-20.04-deb/ 6.1 rel-48 > /etc/apt/sources.list.d/rocm-build.list' && \
        amdgpu-repo --amdgpu-build=1736298; \
    fi

RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu focal main universe | tee -a /etc/apt/sources.list"
RUN amdgpu-install -y --usecase=rocm --no-dkms

## Sccache binary built from source for ROCm, only install if CK_SCCACHE is defined
ARG SCCACHE_REPO_URL=http://compute-artifactory.amd.com/artifactory/rocm-generic-experimental/rocm-sccache
ENV SCCACHE_INSTALL_LOCATION=/usr/local/.cargo/bin
ENV PATH=$PATH:${SCCACHE_INSTALL_LOCATION}
ENV CK_SCCACHE=$CK_SCCACHE
RUN if [ "$CK_SCCACHE" != "" ]; then \
        mkdir -p ${SCCACHE_INSTALL_LOCATION} && \
        curl ${SCCACHE_REPO_URL}/portable/0.2.16/sccache-0.2.16-alpha.1-rocm --output ${SCCACHE_INSTALL_LOCATION}/sccache && \
        chmod +x ${SCCACHE_INSTALL_LOCATION}/sccache; \
    fi

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    git \
    hip-rocclr \
    iputils-ping \
    jq \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    llvm-amdgpu \
    net-tools \
    pkg-config \
    python \
    python3 \
    python3-dev \
    python3-pip \
    redis \
    sshpass \
    stunnel \
    software-properties-common \
    vim \
    nano \
    zlib1g-dev \
    zip \
    openssh-server \
    clang-format-12 \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# hipTensor requires rocm-llvm-dev for rocm versions > 6.0.1
RUN if [ "$ROCMVERSION" = "6.1" ]; then \
        sh -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated rocm-llvm-dev"; \
    fi
# Update the cmake to version 3.27.5
RUN pip install --upgrade cmake==3.27.5

#Install latest ccache
RUN git clone https://github.com/ccache/ccache.git && \
    cd ccache && mkdir build && cd build && cmake .. && make install

#Install ninja build tracing tools
RUN wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
RUN gunzip /usr/local/bin/ninja.gz
RUN chmod a+x /usr/local/bin/ninja
RUN git clone https://github.com/nico/ninjatracing.git

#Install latest cppcheck
RUN git clone https://github.com/danmar/cppcheck.git && \
    cd cppcheck && mkdir build && cd build && cmake .. && cmake --build .
WORKDIR /

# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-3.8 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

ARG PREFIX=/opt/rocm
# Install packages for processing the performance results
RUN pip3 install --upgrade pip
RUN pip3 install sqlalchemy==1.4.46
RUN pip3 install pymysql
RUN pip3 install pandas==2.0.3
RUN pip3 install setuptools-rust
RUN pip3 install sshtunnel==0.4.0
# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN groupadd -f render

# Install the new rocm-cmake version
RUN git clone -b master https://github.com/ROCm/rocm-cmake.git  && \
  cd rocm-cmake && mkdir build && cd build && \
  cmake  .. && cmake --build . && cmake --build . --target install

WORKDIR /

ENV compiler_version=$compiler_version
ENV compiler_commit=$compiler_commit
RUN sh -c "echo compiler version = '$compiler_version'"
RUN sh -c "echo compiler commit = '$compiler_commit'"

RUN if ( [ "$compiler_version" = "amd-staging" ] || [ "$compiler_version" = "amd-mainline-open" ] ) && [ "$compiler_commit" = "" ]; then \
        git clone -b "$compiler_version" https://github.com/ROCm/llvm-project.git && \
        cd llvm-project && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi

RUN if ( [ "$compiler_version" = "amd-staging" ] || [ "$compiler_version" = "amd-mainline-open" ] ) && [ "$compiler_commit" != "" ]; then \
        git clone -b "$compiler_version" https://github.com/ROCm/llvm-project.git && \
        cd llvm-project && git checkout "$compiler_commit" && echo "checking out commit $compiler_commit" && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi

#clean-up the deb package
RUN sh -c "rm -rf amdgpu-install*"

#ENV HIP_CLANG_PATH='/llvm-project/build/bin'
#RUN sh -c "echo HIP_CLANG_PATH = '$HIP_CLANG_PATH'"
