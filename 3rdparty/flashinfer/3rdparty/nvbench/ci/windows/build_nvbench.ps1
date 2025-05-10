
Param(
    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [ValidateNotNullOrEmpty()]
    [string]$ARG_CMAKE_OPTIONS = ""
)

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

Remove-Module -Name build_common
Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList 17

$PRESET = "nvbench-ci"
$CMAKE_OPTIONS = ""

# Append any arguments pass in on the command line
If($ARG_CMAKE_OPTIONS -ne "") {
    $CMAKE_OPTIONS += " $ARG_CMAKE_OPTIONS"
}

configure_and_build_preset "NVBench" "$PRESET" "$CMAKE_OPTIONS"

If($CURRENT_PATH -ne "ci") {
    popd
}
