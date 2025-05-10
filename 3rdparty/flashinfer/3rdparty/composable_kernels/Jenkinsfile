def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

def nthreads() {
    def nproc = sh(returnStdout: true, script: 'nproc')
    echo "Number of cores: ${nproc}"
    def n = nproc.toInteger()
    if (n > 32){
        n /= 2
    }
    if (n > 64){
        n = 64
    }
    echo "Number of threads used for building: ${n}"
    return n
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    return (output != "")
}

def getDockerImageName(){
    def img
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
    if (params.ROCMVERSION != "6.2"){
       if (params.COMPILER_VERSION == "") {
           img = "${env.CK_DOCKERHUB}:ck_ub20.04_rocm${params.ROCMVERSION}"
       }
       else{
          if (params.COMPILER_COMMIT == ""){
             img = "${env.CK_DOCKERHUB}:ck_ub20.04_rocm${params.ROCMVERSION}_${params.COMPILER_VERSION}"
          }
          else{
             def commit = "${params.COMPILER_COMMIT}"[0..6]
             img = "${env.CK_DOCKERHUB}:ck_ub20.04_rocm${params.ROCMVERSION}_${params.COMPILER_VERSION}_${commit}"
          }
       }
    }
    else{
       if (params.COMPILER_VERSION == "") {
           img = "${env.CK_DOCKERHUB_PRIVATE}:ck_ub20.04_rocm${params.ROCMVERSION}"
       }
       else{
          if (params.COMPILER_COMMIT == ""){
             img = "${env.CK_DOCKERHUB_PRIVATE}:ck_ub20.04_rocm${params.ROCMVERSION}_${params.COMPILER_VERSION}"
          }
          else{
             def commit = "${params.COMPILER_COMMIT}"[0..6]
             img = "${env.CK_DOCKERHUB_PRIVATE}:ck_ub20.04_rocm${params.ROCMVERSION}_${params.COMPILER_VERSION}_${commit}"
          }
       }
    }
    }
    return img
}

def check_host() {
    if ("${env.CK_SCCACHE}" != "null"){
        def SCCACHE_SERVER="${env.CK_SCCACHE.split(':')[0]}"
        echo "sccache server: ${SCCACHE_SERVER}"
        sh '''ping -c 1 -p 6379 "${SCCACHE_SERVER}" | echo $? > tmp.txt'''
        def output = readFile(file: "tmp.txt")
        echo "tmp.txt contents: \$output"
        return (output != "0")
    }
    else{
        return 1
    }
}

def build_compiler(){
    def compiler
    if (params.BUILD_COMPILER == "hipcc"){
        compiler = '/opt/rocm/bin/hipcc'
    }
    else{
        if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline-open" || params.COMPILER_COMMIT != ""){
            compiler = "/llvm-project/build/bin/clang++"
        }
        else{
            compiler = "/opt/rocm/llvm/bin/clang++"
        }        
    }
    return compiler
}

def getDockerImage(Map conf=[:]){
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm")
    def no_cache = conf.get("no_cache", false)
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
    if(no_cache)
    {
        dockerArgs = dockerArgs + " --no-cache "
    }
    echo "Docker Args: ${dockerArgs}"
    def image = getDockerImageName()
    //Check if image exists 
    def retimage
    try 
    {
        echo "Pulling down image: ${image}"
        retimage = docker.image("${image}")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            retimage.pull()
        }
    }
    catch(Exception ex)
    {
        error "Unable to locate image: ${image}"
    }
    return [retimage, image]
}

def buildDocker(install_prefix){
    show_node_info()
    env.DOCKER_BUILDKIT=1
    checkout scm
    def image_name = getDockerImageName()
    echo "Building Docker for ${image_name}"
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${install_prefix} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "

    echo "Build Args: ${dockerArgs}"
    try{
        if(params.BUILD_DOCKER){
            //force building the new docker if that parameter is true
            echo "Building image: ${image_name}"
            retimage = docker.build("${image_name}", dockerArgs + ' .')
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                retimage.push()
            }
            sh 'docker images -q -f dangling=true | xargs --no-run-if-empty docker rmi'
        }
        else{
            echo "Checking for image: ${image_name}"
            sh "docker manifest inspect --insecure ${image_name}"
            echo "Image: ${image_name} found! Skipping building image"
        }
    }
    catch(Exception ex){
        echo "Unable to locate image: ${image_name}. Building image now"
        retimage = docker.build("${image_name}", dockerArgs + ' .')
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            retimage.push()
        }
    }
}

def cmake_build(Map conf=[:]){

    def compiler = build_compiler()
    def config_targets = conf.get("config_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def setup_args = conf.get("setup_args","")

    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        config_targets = "package"
    }

    if(conf.get("build_install","") == "true")
    {
        config_targets = 'install ' + config_targets
        setup_args = ' -DBUILD_DEV=On -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }
    if (params.DL_KERNELS){
        setup_args = setup_args + " -DDL_KERNELS=ON "
    }

    if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    def pre_setup_cmd = """
            #!/bin/bash
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            cd build
        """
    def invocation_tag=""
    if (setup_args.contains("gfx11")){
        invocation_tag="gfx11"
    }
    if (setup_args.contains("gfx10")){
        invocation_tag="gfx10"
    }
    if (setup_args.contains("gfx90")){
        invocation_tag="gfx90"
    }
    if (setup_args.contains("gfx94")){
        invocation_tag="gfx94"
    }
    echo "invocation tag: ${invocation_tag}"
    def redis_pre_setup_cmd = pre_setup_cmd
    if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
        redis_pre_setup_cmd = pre_setup_cmd + """
            #!/bin/bash
            export ROCM_PATH=/opt/rocm
            export SCCACHE_ENABLED=true
            export SCCACHE_LOG_LEVEL=debug
            export SCCACHE_IDLE_TIMEOUT=14400
            export COMPILERS_HASH_DIR=/tmp/.sccache
            export SCCACHE_BIN=/usr/local/.cargo/bin/sccache
            export SCCACHE_EXTRAFILES=/tmp/.sccache/rocm_compilers_hash_file
            export SCCACHE_REDIS="redis://${env.CK_SCCACHE}"
            echo "connect = ${env.CK_SCCACHE}" >> ../script/redis-cli.conf
            export SCCACHE_C_CUSTOM_CACHE_BUSTER="${invocation_tag}"
            echo \$SCCACHE_C_CUSTOM_CACHE_BUSTER
            stunnel ../script/redis-cli.conf
            ../script/sccache_wrapper.sh --enforce_redis
        """
        try {
            def cmd1 = conf.get("cmd1", """
                    ${redis_pre_setup_cmd}
                """)
            sh cmd1
            setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache " + setup_args
        }
        catch(Exception err){
            echo "could not connect to redis server: ${err.getMessage()}. will not use sccache."
            def cmd2 = conf.get("cmd2", """
                    ${pre_setup_cmd}
                """)
            sh cmd2
        }
    }
    else{
        def cmd3 = conf.get("cmd3",  """
                ${pre_setup_cmd}
            """)
        sh cmd3
    }
    // reduce parallelism when compiling, clang uses too much memory
    def nt = nthreads()
    def cmd
    def execute_cmd = conf.get("execute_cmd", "")
    if(!setup_args.contains("NO_CK_BUILD")){
        def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args}   .. ")
        def build_cmd = conf.get("build_cmd", "${build_envs} dumb-init make  -j${nt} ${config_targets}")
        cmd = conf.get("cmd", """
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)
    }
    else{
        cmd = conf.get("cmd", """
            ${execute_cmd}
        """)
    }

    echo cmd

    dir("build"){
        sh cmd
    }

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "amd-master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm

        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
        if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline-open" || params.COMPILER_COMMIT != ""){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }

        def variant = env.STAGE_NAME

        def retimage
        (retimage, image) = getDockerImage(conf)

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 48, unit: 'HOURS')
                {
                    cmake_build(conf)
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def runCKProfiler(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm

        def image = getDockerImageName()
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "

        def variant = env.STAGE_NAME
        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'rocminfo | tee rocminfo.log'
                        if ( !runShell('grep -n "gfx" rocminfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 24, unit: 'HOURS')
                {
                    sh """
                        rm -rf build
                        mkdir build
                    """
                    dir("build"){
                        unstash 'ckProfiler.tar.gz'
                        sh 'tar -xvf ckProfiler.tar.gz'
                    }

					dir("script"){
                        if (params.RUN_FULL_QA){
                            sh "./run_full_performance_tests.sh 0 QA_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm.log"
                            archiveArtifacts "perf_resnet50_N256.log"
                            archiveArtifacts "perf_resnet50_N4.log"
                            archiveArtifacts "perf_batched_gemm.log"
                            archiveArtifacts "perf_grouped_gemm.log"
                            archiveArtifacts "perf_conv_fwd.log"
                            archiveArtifacts "perf_conv_bwd_data.log"
                            archiveArtifacts "perf_gemm_bilinear.log"
                            archiveArtifacts "perf_reduction.log"
                            archiveArtifacts "perf_splitK_gemm.log"
                            archiveArtifacts "perf_onnx_gemm.log"
                            archiveArtifacts "perf_mixed_gemm.log"
                           // stash perf files to master
                            stash name: "perf_gemm.log"
                            stash name: "perf_resnet50_N256.log"
                            stash name: "perf_resnet50_N4.log"
                            stash name: "perf_batched_gemm.log"
                            stash name: "perf_grouped_gemm.log"
                            stash name: "perf_conv_fwd.log"
                            stash name: "perf_conv_bwd_data.log"
                            stash name: "perf_gemm_bilinear.log"
                            stash name: "perf_reduction.log"
                            stash name: "perf_splitK_gemm.log"
                            stash name: "perf_onnx_gemm.log"
                            stash name: "perf_mixed_gemm.log"
                            //we will process results on the master node
                        }
                        else{
                            sh "./run_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm.log"
                            archiveArtifacts "perf_resnet50_N256.log"
                            archiveArtifacts "perf_resnet50_N4.log"
                            // stash perf files to master
                            stash name: "perf_gemm.log"
                            stash name: "perf_resnet50_N256.log"
                            stash name: "perf_resnet50_N4.log"
                            //we will process the results on the master node
                        }
					}
                }
            }
        }
        return retimage
}

def runPerfTest(Map conf=[:]){
    try{
        runCKProfiler(conf)
    }
    catch(e){
        echo "throwing error exception in performance tests"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def Build_CK(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        checkout scm

        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
        if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline-open" || params.COMPILER_COMMIT != ""){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }
        def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
        def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
        dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
        echo "Docker flags: ${dockerOpts}"

        def variant = env.STAGE_NAME
        def retimage
        gitStatusWrapper(credentialsId: "${env.status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'rocminfo | tee rocminfo.log'
                        if ( !runShell('grep -n "gfx" rocminfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 24, unit: 'HOURS')
                {
                    //check whether to run performance tests on this node
                    def do_perf_tests = 0
                    sh 'rocminfo | tee rocminfo.log'
                    if ( runShell('grep -n "gfx1030" rocminfo.log') || runShell('grep -n "gfx1101" rocminfo.log') || runShell('grep -n "gfx942" rocminfo.log') ){
                        do_perf_tests = 1
                        echo "Stash profiler and run performance tests"
                    }
                    cmake_build(conf)
                    dir("build"){
                        //run tests and examples
                        sh 'make -j check'
                        if (params.RUN_PERFORMANCE_TESTS && do_perf_tests == 0 ){
                            //we only need the ckProfiler to run the performance tests, so we pack and stash it
                            //do not stash profiler on nodes where we don't need to run performance tests
                            sh 'tar -zcvf ckProfiler.tar.gz bin/ckProfiler'
                            stash name: "ckProfiler.tar.gz"
                        }
                        if (params.RUN_FULL_QA && do_perf_tests == 0 ){
                            // build deb packages for all gfx9 targets and prepare to export
                            sh 'make -j package'
                            archiveArtifacts artifacts: 'composablekernel-ckprofiler_*.deb'
                            archiveArtifacts artifacts: 'composablekernel-tests_*.deb'
                            sh 'mv composablekernel-ckprofiler_*.deb ckprofiler_0.2.0_amd64.deb'
                            stash name: "ckprofiler_0.2.0_amd64.deb"
                        }
                    }
                    if (params.hipTensor_test && do_perf_tests == 0 ){
                        //build and test hipTensor
                        sh """#!/bin/bash
                            rm -rf "${params.hipTensor_branch}".zip
                            rm -rf hipTensor-"${params.hipTensor_branch}"
                            wget https://github.com/ROCm/hipTensor/archive/refs/heads/"${params.hipTensor_branch}".zip
                            unzip -o "${params.hipTensor_branch}".zip
                        """
                        dir("hipTensor-${params.hipTensor_branch}"){
                            sh """#!/bin/bash
                                mkdir -p build
                                ls -ltr
                                CC=hipcc CXX=hipcc cmake -Bbuild . -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install"
                                cmake --build build -- -j
                            """
                        }
                        dir("hipTensor-${params.hipTensor_branch}/build"){
                            sh 'ctest'
                        }
                    }
                }
            }
        }
        return retimage
}

def Build_CK_and_Reboot(Map conf=[:]){
    try{
        Build_CK(conf)
    }
    catch(e){
        echo "throwing error exception while building CK"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def process_results(Map conf=[:]){
    env.HSA_ENABLE_SDMA=0
    checkout scm
    def image = getDockerImageName() 
    def prefixpath = "/opt/rocm"

    // Jenkins is complaining about the render group 
    def dockerOpts="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if (conf.get("enforce_xnack_on", false)) {
        dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
    }

    def variant = env.STAGE_NAME
    def retimage

    gitStatusWrapper(credentialsId: "${env.status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
        try {
            (retimage, image) = getDockerImage(conf)
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
            echo "The job was cancelled or aborted"
            throw e
        }
    }

    withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 1, unit: 'HOURS'){
            try{
                dir("script"){
                    if (params.RUN_FULL_QA){
                        // unstash perf files to master
                        unstash "ckprofiler_0.2.0_amd64.deb"
                        sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no ckprofiler_0.2.0_amd64.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        unstash "perf_gemm.log"
                        unstash "perf_resnet50_N256.log"
                        unstash "perf_resnet50_N4.log"
                        unstash "perf_batched_gemm.log"
                        unstash "perf_grouped_gemm.log"
                        unstash "perf_conv_fwd.log"
                        unstash "perf_conv_bwd_data.log"
                        unstash "perf_gemm_bilinear.log"
                        unstash "perf_reduction.log"
                        unstash "perf_splitK_gemm.log"
                        unstash "perf_onnx_gemm.log"
                        unstash "perf_mixed_gemm.log"
                        sh "./process_qa_data.sh"
                    }
                    else{
                        // unstash perf files to master
                        unstash "perf_gemm.log"
                        unstash "perf_resnet50_N256.log"
                        unstash "perf_resnet50_N4.log"
                        sh "./process_perf_data.sh"
                    }
                }
            }
            catch(e){
                echo "Throwing error exception while processing performance test results"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
            finally{
                echo "Finished processing performance test results"
            }
        }
    }
}

//launch develop branch daily at 23:00 UT in FULL_QA mode and at 19:00 UT with latest staging compiler version
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true;ROCMVERSION=6.1;COMPILER_VERSION=
                                              0 21 * * * % ROCMVERSION=6.1;COMPILER_VERSION=;COMPILER_COMMIT=
                                              0 19 * * * % BUILD_DOCKER=true;DL_KERNELS=true;COMPILER_VERSION=amd-staging;COMPILER_COMMIT=;USE_SCCACHE=false
                                              0 17 * * * % BUILD_DOCKER=true;DL_KERNELS=true;COMPILER_VERSION=amd-mainline-open;COMPILER_COMMIT=;USE_SCCACHE=false
                                              0 15 * * * % BUILD_INSTANCES_ONLY=true;RUN_CODEGEN_TESTS=false;RUN_PERFORMANCE_TESTS=false;USE_SCCACHE=false''' : ""

pipeline {
    agent none
    triggers {
        parameterizedCron(CRON_SETTINGS)
    }
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
            defaultValue: false,
            description: "Force building docker image (default: false), set to true if docker image needs to be updated.")
        string(
            name: 'USE_CUSTOM_DOCKER',
            defaultValue: '',
            description: 'If you want to use a custom docker image, please specify it here (default: leave blank).')
        string(
            name: 'ROCMVERSION', 
            defaultValue: '6.1', 
            description: 'Specify which ROCM version to use: 6.1 (default).')
        string(
            name: 'COMPILER_VERSION', 
            defaultValue: '', 
            description: 'Specify which version of compiler to use: release, amd-staging, amd-mainline-open, or leave blank (default).')
        string(
            name: 'COMPILER_COMMIT', 
            defaultValue: '', 
            description: 'Specify which commit of compiler branch to use: leave blank to use the latest commit (default), or use some specific commit of llvm-project branch.')
        string(
            name: 'BUILD_COMPILER', 
            defaultValue: 'clang', 
            description: 'Specify whether to build CK with hipcc or with clang (default).')
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
        booleanParam(
            name: "DL_KERNELS",
            defaultValue: false,
            description: "Select whether to build DL kernels (default: OFF)")
        booleanParam(
            name: "hipTensor_test",
            defaultValue: true,
            description: "Use the CK build to verify hipTensor build and tests (default: ON)")
        string(
            name: 'hipTensor_branch',
            defaultValue: 'mainline',
            description: 'Specify which branch of hipTensor to use (default: mainline)')
        booleanParam(
            name: "USE_SCCACHE",
            defaultValue: true,
            description: "Use the sccache for building CK (default: ON)")
        booleanParam(
            name: "RUN_CPPCHECK",
            defaultValue: false,
            description: "Run the cppcheck static analysis (default: OFF)")
        booleanParam(
            name: "RUN_PERFORMANCE_TESTS",
            defaultValue: true,
            description: "Run the performance tests (default: ON)")
        booleanParam(
            name: "RUN_CODEGEN_TESTS",
            defaultValue: true,
            description: "Run the codegen tests (default: ON)")
        booleanParam(
            name: "BUILD_INSTANCES_ONLY",
            defaultValue: false,
            description: "Test building instances for various architectures simultaneously (default: OFF)")
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        status_wrapper_creds = "${status_wrapper_creds}"
        gerrit_cred="${gerrit_cred}"
        DOCKER_BUILDKIT = "1"
    }
    stages{
        stage("Build Docker"){
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/opt/rocm')
                        cleanWs()
                    }
                }
            }
        }
        stage("Static checks") {
            parallel{
                stage('Clang Format and Cppcheck') {
                    when {
                        beforeAgent true
                        expression { params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find .. -not -path \'*.git*\' -iname \'*.h\' \
                                -o -not -path \'*.git*\' -iname \'*.hpp\' \
                                -o -not -path \'*.git*\' -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-12 -style=file {} | diff - {}\' && \
                                /cppcheck/build/bin/cppcheck ../* -v -j \$(nproc) -I ../include -I ../profiler/include -I ../library/include \
                                -D CK_ENABLE_FP64 -D CK_ENABLE_FP32 -D CK_ENABLE_FP16 -D CK_ENABLE_FP8 -D CK_ENABLE_BF16 -D CK_ENABLE_BF8 -D CK_ENABLE_INT8 -D DL_KERNELS \
                                -D __gfx908__ -D __gfx90a__ -D __gfx940__ -D __gfx941__ -D __gfx942__ -D __gfx1030__ -D __gfx1100__ -D __gfx1101__ -D __gfx1102__ \
                                -U __gfx803__ -U __gfx900__ -U __gfx906__ -U CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 \
                                --file-filter=*.cpp --force --enable=all --output-file=ck_cppcheck.log"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
                        archiveArtifacts "build/ck_cppcheck.log"
                        cleanWs()
                    }
                }
                stage('Clang Format') {
                    when {
                        beforeAgent true
                        expression { !params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find .. -not -path \'*.git*\' -iname \'*.h\' \
                                -o -not -path \'*.git*\' -iname \'*.hpp\' \
                                -o -not -path \'*.git*\' -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-12 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Codegen Tests")
        {
            parallel
            {
                stage("Run Codegen Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CODEGEN_TESTS.toBoolean() }
                    }
                    options { retry(2) }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cd ../codegen && rm -rf build && mkdir build && cd build && \
                                           cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
                                           -D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
                                           -D CMAKE_BUILD_TYPE=Release \
                                           -D GPU_TARGETS="gfx90a" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j check"""
                   }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
		stage("Build CK and run Tests")
        {
            parallel
            {
                stage("Build CK for all gfx9 targets")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install \
                                         -DGPU_TARGETS="gfx908;gfx90a;gfx940;gfx941;gfx942" \
                                         -DCMAKE_EXE_LINKER_FLAGS=" -L ${env.WORKSPACE}/script -T hip_fatbin_insert " \
                                         -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx908;gfx90a;gfx940;gfx941;gfx942" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx942" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx942" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx908;gfx90a" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx908;gfx90a" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK instances for different targets")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_INSTANCES_ONLY.toBoolean() && !params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        execute_args = """ cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
                                           -D CMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -D CMAKE_BUILD_TYPE=Release \
                                           -D GPU_TARGETS="gfx90a;gfx1030;gfx1101" \
                                           -D INSTANCES_ONLY=ON \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j32 """
                   }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "",  build_cmd: "", no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1030")
                {
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1030") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx1030" -DDL_KERNELS=ON -DCMAKE_CXX_FLAGS=" -O3 " """ 
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx1030" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1101")
                {
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1101") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx1101" -DDL_KERNELS=ON -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx1101" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
            }
        }

        stage("Performance Tests")
        {
            parallel
            {
                stage("Run ckProfiler: gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_PERFORMANCE_TESTS.toBoolean() }
                    }
                    options { retry(2) }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = """ -DGPU_TARGETS="gfx90a" -DBUILD_DEV=On """
                    }
                    steps{
                        runPerfTest(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Release')
                        cleanWs()
                    }
                }
            }
        }
        stage("Process Performance Test Results")
        {
            parallel
            {
                stage("Process results"){
                    when {
                        beforeAgent true
                        expression { params.RUN_PERFORMANCE_TESTS.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results()
                        cleanWs()
                    }
                }
            }
        }
    }
}
