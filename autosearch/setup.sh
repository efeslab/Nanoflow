wget https://packages.gurobi.com/12.0/gurobi12.0.1_linux64.tar.gz
tar xvfz gurobi12.0.1_linux64.tar.gz
export GUROBI_HOME=$(pwd)/gurobi1201/linux64
export PATH="${GUROBI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
grbgetkey
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv 
source .venv/bin/activate
uv pip install ipykernel
uv pip install pandas matplotlib seaborn scikit-learn gurobipy